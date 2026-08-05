#pragma once
// carrier_recovery.hpp: Costas loop for carrier frequency and phase recovery.
//
// Removes the residual frequency and phase offset left by the down-converter
// so the constellation stands still and can be demapped coherently. It is
// the timing loop's sibling — same PI filter, same second-order dynamics —
// and shares its loop filter through <sw/dsp/sdr/loop_filter.hpp>.
//
// THE PHASE ACCUMULATOR IS WRAPPED, ALWAYS. It is tempting to let phase run
// as an unbounded sum of per-sample increments, but that value grows without
// limit: at 0.01 radians per sample it passes 1e4 within a million samples,
// where a 16-bit float's ULP exceeds the increment entirely and the
// oscillator stops turning. Wrapping to [-pi, pi) every sample keeps it
// where every scalar type has resolution to spare. This is the same lesson
// the timing loop's symbol clock carries, in a different disguise.
//
// DETECTORS
//
//   bpsk                e = Im(y) * sign(Re(y)). One decision axis, so it
//                       tolerates any phase within +/-90 degrees but has a
//                       180-degree ambiguity.
//   qpsk                e = Im(y)*sign(Re(y)) - Re(y)*sign(Im(y)). Two axes,
//                       90-degree ambiguity.
//   decision_directed   e = Im(y * conj(a)) against the sliced constellation
//                       point a. Needed for anything denser than QPSK, where
//                       the sign of a coordinate no longer determines the
//                       decision, but it needs the loop to be close already.
//
// All three leave a phase AMBIGUITY — a Costas loop cannot tell which
// constellation rotation it locked to, because the signal is symmetric under
// exactly that rotation. Resolving it is the job of a known preamble or of
// differential encoding, both outside this class.
//
// AFC. A phase detector's pull-in range is roughly the loop bandwidth, so a
// frequency offset much larger than that never acquires. Enabling AFC adds a
// frequency-error term, Im(y[k] * conj(y[k-1])), which measures the rotation
// BETWEEN consecutive symbols and so responds to frequency directly with a
// far wider capture range. It is fed to the integrator alongside the phase
// error and fades out as the loop locks, since near lock it contributes only
// noise.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/sdr/agc.hpp>          // detail::ComplexSample
#include <sw/dsp/sdr/loop_filter.hpp>

namespace sw::dsp::sdr {

enum class CarrierDetector { bpsk, qpsk, decision_directed };

template <DspField T = double>
struct CarrierRecoveryConfig {
	T loop_bandwidth = T(0.01);    // Bn*T, normalized to the symbol rate
	T damping        = T(0.707);
	T detector_gain  = T(1);       // Kp
	// Largest frequency the loop may track, radians per symbol. Also bounds
	// the integrator, so a runaway cannot spin the oscillator arbitrarily.
	T max_frequency  = T(0.5);
	CarrierDetector detector = CarrierDetector::qpsk;
	// AFC widens acquisition well beyond the phase detector's pull-in range.
	bool enable_afc  = true;
	T    afc_gain    = T(0.01);
	// Lock is declared once the smoothed |phase error| falls below this.
	T lock_threshold      = T(0.1);
	T lock_average_symbols = T(64);
};

// ============================================================================
// CarrierRecovery
// ============================================================================

template <DspField StateScalar = double,
          typename SampleScalar = complex_for_t<StateScalar>>
class CarrierRecovery {
public:
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	using config_type   = CarrierRecoveryConfig<StateScalar>;

	static_assert(detail::ComplexSample<SampleScalar>,
	              "CarrierRecovery needs complex samples: a real signal carries "
	              "no phase for the loop to correct.");

	// Invariants, established here and maintained by every process() call:
	//   I1  phase_ in [-pi, pi)
	//   I2  |frequency()| <= max_frequency
	//   I3  lock_error_ >= 0
	//   I4  loop gains finite and > 0
	//   I5  config validated
	explicit CarrierRecovery(const config_type& cfg = config_type{})
		: cfg_(cfg) {
		validate(cfg_);
		LoopFilterConfig<StateScalar> lf;
		lf.bandwidth     = cfg_.loop_bandwidth;
		lf.damping       = cfg_.damping;
		lf.detector_gain = cfg_.detector_gain;
		loop_.configure(lf);
		lock_alpha_ = StateScalar{1} /
		              ((cfg_.lock_average_symbols > StateScalar{})
		               ? cfg_.lock_average_symbols : StateScalar{1});
		reset();
	}

	const config_type& config() const { return cfg_; }

	// --- observable state ---------------------------------------------

	// Current NCO phase, radians, wrapped to [-pi, pi).
	StateScalar phase() const { return phase_; }
	// Tracked frequency offset, radians per symbol.
	StateScalar frequency() const { return loop_.integral(); }
	// Most recent phase-detector output.
	StateScalar phase_error() const { return last_error_; }
	// Smoothed |phase error| — the lock metric.
	StateScalar lock_metric() const { return lock_error_; }
	bool is_locked() const { return locked_; }
	std::size_t symbols_processed() const { return symbols_; }

	void reset() {
		phase_ = StateScalar{};
		loop_.reset();
		last_error_ = StateScalar{};
		lock_error_ = StateScalar{};
		locked_     = false;
		symbols_    = 0;
		have_prev_  = false;
		prev_sr_    = StateScalar{};
		prev_si_    = StateScalar{};
	}

	// --- processing ----------------------------------------------------

	// De-rotate one symbol and update the loop.
	//
	// Pre:  none.
	// Post: returns the input rotated by -phase_ as it stood on entry; the
	//       loop has advanced by one step; all invariants hold.
	SampleScalar process(SampleScalar in) {
		using std::cos;
		using std::sin;
		const StateScalar c = cos(-phase_), s = sin(-phase_);
		const SampleScalar y = rotate(in, c, s);

		const StateScalar e = phase_detect(y);
		last_error_ = e;

		// AFC contributes a frequency error, which the integrator absorbs
		// directly. It fades as the loop locks: near lock the cross-symbol
		// rotation is dominated by noise and modulation, so keeping it in
		// would only add jitter.
		// The frequency detector must be MODULATION-STRIPPED. The obvious
		// form, Im(y[k] * conj(y[k-1])), measures the rotation between
		// consecutive symbols — but on QPSK data consecutive symbols already
		// differ by a random multiple of 90 degrees, so that quantity is
		// dominated by the data and swamps the frequency it is meant to
		// measure. Feeding it to the integrator drives the loop off entirely:
		// measured, it turned an exact lock into a 0.76 error vector at every
		// offset tested.
		//
		// Stripping first fixes it. d = y * conj(a) with a the sliced
		// decision leaves only the phase error, so Im(d[k] * conj(d[k-1])) is
		// the rotation BETWEEN ERRORS — frequency, with the modulation gone.
		StateScalar integral_bump{};
		const StateScalar sr = strip_re(y), si = strip_im(y);
		if (cfg_.enable_afc && have_prev_) {
			const StateScalar ef = si * prev_sr_ - sr * prev_si_;
			const StateScalar fade = locked_ ? StateScalar{} : StateScalar{1};
			integral_bump = cfg_.afc_gain * ef * fade;
		}

		const StateScalar advance = loop_.advance(e);
		loop_.set_integral(loop_.integral() + integral_bump);
		loop_.clamp_integral(cfg_.max_frequency);          // I2

		phase_ = wrap(phase_ + advance);                   // I1

		// Lock metric: smoothed magnitude of the phase error.
		using std::abs;
		lock_error_ = lock_error_ + lock_alpha_ * (abs(e) - lock_error_);
		if (lock_error_ < StateScalar{}) lock_error_ = StateScalar{};   // I3
		locked_ = symbols_ > 4 * static_cast<std::size_t>(
		              static_cast<double>(cfg_.lock_average_symbols)) &&
		          lock_error_ < cfg_.lock_threshold;

		prev_sr_ = sr;
		prev_si_ = si;
		have_prev_ = true;
		++symbols_;
		return y;
	}

	bool invariants_hold() const {
		using std::abs;
		const StateScalar slack = static_cast<StateScalar>(1e-9);
		if (phase_ < -static_cast<StateScalar>(pi) - slack ||
		    phase_ >  static_cast<StateScalar>(pi) + slack) return false;    // I1
		if (abs(loop_.integral()) > cfg_.max_frequency + slack) return false; // I2
		if (lock_error_ < StateScalar{}) return false;                        // I3
		if (!(loop_.proportional_gain() > StateScalar{}) ||
		    !(loop_.integral_gain() > StateScalar{})) return false;           // I4
		return true;
	}

private:
	static void validate(const config_type& c) {
		if (!(c.max_frequency > StateScalar{}))
			throw std::invalid_argument("CarrierRecovery: max_frequency must be > 0");
		if (c.enable_afc && !(c.afc_gain > StateScalar{}))
			throw std::invalid_argument("CarrierRecovery: afc_gain must be > 0 when AFC is enabled");
		if (!(c.lock_threshold > StateScalar{}))
			throw std::invalid_argument("CarrierRecovery: lock_threshold must be > 0");
		// bandwidth / damping / detector_gain are validated by PiLoopFilter.
	}

	static StateScalar wrap(StateScalar p) {
		const StateScalar tp = static_cast<StateScalar>(two_pi);
		const StateScalar p_ = static_cast<StateScalar>(pi);
		while (p >=  p_) p -= tp;
		while (p <  -p_) p += tp;
		return p;
	}

	static SampleScalar rotate(const SampleScalar& x, StateScalar c, StateScalar s) {
		using C = std::remove_cvref_t<decltype(x.real())>;
		const C cc = static_cast<C>(c), ss = static_cast<C>(s);
		return SampleScalar(x.real() * cc - x.imag() * ss,
		                    x.real() * ss + x.imag() * cc);
	}

	static StateScalar re(const SampleScalar& x) { return static_cast<StateScalar>(x.real()); }
	static StateScalar im(const SampleScalar& x) { return static_cast<StateScalar>(x.imag()); }
	static StateScalar sgn(StateScalar v) { return (v < StateScalar{}) ? StateScalar{-1} : StateScalar{1}; }

	// y * conj(a), normalized: the residual error vector with the data
	// rotation removed. Shared by the AFC for both axes.
	StateScalar strip_re(const SampleScalar& y) const {
		StateScalar ar, ai;
		decision(y, ar, ai);
		const StateScalar m = ar * ar + ai * ai;
		if (!(m > StateScalar{})) return StateScalar{};
		return (re(y) * ar + im(y) * ai) / m;
	}
	StateScalar strip_im(const SampleScalar& y) const {
		StateScalar ar, ai;
		decision(y, ar, ai);
		const StateScalar m = ar * ar + ai * ai;
		if (!(m > StateScalar{})) return StateScalar{};
		return (im(y) * ar - re(y) * ai) / m;
	}

	// The sliced decision for the configured detector.
	void decision(const SampleScalar& y, StateScalar& ar, StateScalar& ai) const {
		switch (cfg_.detector) {
			case CarrierDetector::bpsk:
				ar = sgn(re(y)); ai = StateScalar{}; break;
			case CarrierDetector::qpsk:
				ar = sgn(re(y)); ai = sgn(im(y)); break;
			default:
				ar = sgn(re(y)) * quantize(std::abs(static_cast<double>(re(y))));
				ai = sgn(im(y)) * quantize(std::abs(static_cast<double>(im(y))));
				break;
		}
	}

	StateScalar phase_detect(const SampleScalar& y) const {
		switch (cfg_.detector) {
			case CarrierDetector::bpsk:
				return im(y) * sgn(re(y));
			case CarrierDetector::qpsk:
				return im(y) * sgn(re(y)) - re(y) * sgn(im(y));
			default: {
				// Decision-directed: Im(y * conj(a)) with a the sliced point.
				// Normalized by |a|^2 so an outer 16-QAM point does not
				// dominate an inner one and change the effective loop gain
				// with the data.
				const StateScalar ar = sgn(re(y)) * quantize(std::abs(static_cast<double>(re(y))));
				const StateScalar ai = sgn(im(y)) * quantize(std::abs(static_cast<double>(im(y))));
				const StateScalar mag2 = ar * ar + ai * ai;
				if (!(mag2 > StateScalar{})) return StateScalar{};
				return (im(y) * ar - re(y) * ai) / mag2;
			}
		}
	}

	// Slice one axis of a unit-average-power square QAM constellation onto
	// the odd-integer grid the mapper uses, scaled the same way.
	static StateScalar quantize(double v) {
		// Levels for the supported square QAMs after unit-power scaling; the
		// nearest is a good enough decision for a tracking loop.
		static const double levels[] = {1.0 / 3.0, 1.0, 5.0 / 3.0, 7.0 / 3.0};
		double best = levels[0], bd = std::abs(v - levels[0]);
		for (double l : levels) {
			const double d = std::abs(v - l);
			if (d < bd) { bd = d; best = l; }
		}
		return static_cast<StateScalar>(best);
	}

	config_type cfg_;
	PiLoopFilter<StateScalar> loop_;
	StateScalar lock_alpha_{};

	StateScalar phase_{};
	StateScalar last_error_{};
	StateScalar lock_error_{};
	bool        locked_ = false;
	std::size_t symbols_ = 0;

	// Previous MODULATION-STRIPPED error vector, y * conj(decision).
	StateScalar prev_sr_{}, prev_si_{};
	bool        have_prev_ = false;
};

} // namespace sw::dsp::sdr
