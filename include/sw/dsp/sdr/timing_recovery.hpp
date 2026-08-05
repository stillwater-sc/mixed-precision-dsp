#pragma once
// timing_recovery.hpp: symbol timing recovery.
//
// Finds the sampling instant inside each symbol period and tracks it as the
// transmitter's and receiver's clocks drift apart. Structurally this is the
// AGC's sibling — a feedback loop whose precision shows up as jitter and
// lock time rather than as a static error — but with two differences that
// make it harder: the loop is SECOND ORDER, so it can be made unstable by
// its own arithmetic, and it drives an INTERPOLATOR, so the quantity being
// corrected is a continuous position between samples rather than a scalar
// gain.
//
// STRUCTURE
//
//   input at N samples/symbol
//        |
//   cubic Farrow interpolator  <-- fractional position from the loop
//        |
//   timing error detector (Gardner or Mueller-Muller)
//        |
//   proportional-integral loop filter
//        |
//   symbol clock: next symbol is omega samples after this one
//
// INTERPOLATION is a four-point cubic Lagrange (Farrow) rather than the
// library's polyphase FractionalDelay. FractionalDelay quantizes the delay
// to 1/L, and that quantization would appear directly as timing jitter — the
// very quantity this module exists to measure. The Farrow form takes a
// continuous fractional position, is exact for cubic signals, and costs four
// taps.
//
// LOOP FILTER gains follow the standard second-order mapping from a
// normalized noise bandwidth Bn*T and damping factor zeta (Rice, "Digital
// Communications: A Discrete-Time Approach", sec. 7.2):
//
//   theta = (Bn*T) / (zeta + 1/(4*zeta))
//   k_p   = 4*zeta*theta / (1 + 2*zeta*theta + theta^2) / Kp
//   k_i   = 4*theta^2    / (1 + 2*zeta*theta + theta^2) / Kp
//
// so a caller asks for a bandwidth and a damping rather than for two opaque
// gains. Kp is the detector's slope, which depends on the signal; it is
// exposed as `detector_gain` for calibration and defaults to 1.
//
// DETECTORS
//
//   Gardner          non-data-aided, needs 2 samples/symbol, and is blind to
//                    carrier phase — usable before carrier recovery, which
//                    is why it usually comes first in a receiver.
//   Mueller-Muller   decision-directed, works at 1 sample/symbol, but needs
//                    the constellation to be roughly correct already, so it
//                    belongs after carrier recovery.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/sdr/agc.hpp>   // detail::ComplexSample

namespace sw::dsp::sdr {

enum class TimingDetector {
	// Gardner: e = Re{ (y[k] - y[k-1]) * conj(y[k-1/2]) }. Non-data-aided,
	// requires the midpoint sample and therefore >= 2 samples/symbol.
	gardner,
	// Mueller-Muller: e = Re{ conj(a[k-1])*y[k] - conj(a[k])*y[k-1] } with
	// a[] the sliced decisions. Decision-directed, 1 sample/symbol.
	mueller_muller
};

template <DspField T = double>
struct TimingRecoveryConfig {
	T samples_per_symbol = T(2);
	T loop_bandwidth     = T(0.01);   // Bn*T, normalized to the symbol rate
	T damping            = T(0.707);  // critically damped-ish
	// Kp, the detector's slope at the zero crossing. It scales with SIGNAL
	// POWER, because both detectors form a product of two sample values —
	// which is why an AGC belongs upstream of a timing loop. Measured ~0.7
	// per symbol for unit-power BPSK through an alpha=0.35 RRC pair. Leaving
	// this at 1 simply scales the achieved loop bandwidth by 1/Kp.
	T detector_gain      = T(1);
	T max_deviation      = T(0.05);   // omega clamp, as a fraction of nominal
	TimingDetector detector = TimingDetector::gardner;
	// Lock is declared when the smoothed RMS timing error falls below this.
	T lock_threshold     = T(0.05);
	T lock_average_symbols = T(64);   // smoothing window for the lock metric
};

namespace detail {

// Four-point cubic Lagrange. s[] are consecutive in TIME, s[0] oldest; the
// result is the signal at fraction `mu` of the way from s[1] to s[2].
// Verified exact for cubic inputs and correct at both endpoints.
template <DspField State, typename S>
S farrow_cubic(const S s[4], State mu) {
	const State one = State{1}, two = State{2};
	const State h0 = -mu * (mu - one) * (mu - two) / State(6);
	const State h1 =  (mu + one) * (mu - one) * (mu - two) / two;
	const State h2 = -(mu + one) * mu * (mu - two) / two;
	const State h3 =  (mu + one) * mu * (mu - one) / State(6);
	if constexpr (ComplexSample<S>) {
		using C = std::remove_cvref_t<decltype(s[0].real())>;
		return S(s[0].real() * static_cast<C>(h0) + s[1].real() * static_cast<C>(h1)
		       + s[2].real() * static_cast<C>(h2) + s[3].real() * static_cast<C>(h3),
		         s[0].imag() * static_cast<C>(h0) + s[1].imag() * static_cast<C>(h1)
		       + s[2].imag() * static_cast<C>(h2) + s[3].imag() * static_cast<C>(h3));
	} else {
		return static_cast<S>(s[0] * static_cast<S>(h0) + s[1] * static_cast<S>(h1)
		                    + s[2] * static_cast<S>(h2) + s[3] * static_cast<S>(h3));
	}
}

// Real part of a * conj(b), for real or complex samples. This is the only
// combination the two detectors need.
template <DspField State, typename S>
State real_of_a_conj_b(const S& a, const S& b) {
	if constexpr (ComplexSample<S>) {
		return static_cast<State>(a.real()) * static_cast<State>(b.real())
		     + static_cast<State>(a.imag()) * static_cast<State>(b.imag());
	} else {
		return static_cast<State>(a) * static_cast<State>(b);
	}
}

// Hard slicer to +/-1 on each axis — the decision Mueller-Muller needs.
// Sufficient for any Gray-coded square constellation, whose decision
// boundaries on the outer bits are the axes.
template <typename S>
S unit_slice(const S& x) {
	auto sgn = [](auto v) { return (v < decltype(v){}) ? decltype(v){-1} : decltype(v){1}; };
	if constexpr (ComplexSample<S>) {
		return S(sgn(x.real()), sgn(x.imag()));
	} else {
		return sgn(x);
	}
}

} // namespace detail

// ============================================================================
// TimingRecovery
// ============================================================================

template <DspField StateScalar = double, typename SampleScalar = StateScalar>
class TimingRecovery {
public:
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	using config_type   = TimingRecoveryConfig<StateScalar>;

	// Invariants established here and maintained by every process() call:
	//   I1  omega_ within [nominal*(1-max_dev), nominal*(1+max_dev)]
	//   I2  the interpolation delay stays inside the ring buffer
	//   I3  lock_metric_ >= 0
	//   I4  k_p_, k_i_ finite and > 0
	//   I5  config validated (rates, bandwidth, damping all positive)
	explicit TimingRecovery(const config_type& cfg = config_type{})
		: cfg_(cfg) {
		validate(cfg_);
		compute_gains();
		// The ring must hold the oldest point either detector can ask for:
		// the mid-symbol sample sits omega/2 further back than the symbol
		// itself, and the cubic needs one sample either side of that.
		const double sps_max = static_cast<double>(cfg_.samples_per_symbol) *
		                       (1.0 + static_cast<double>(cfg_.max_deviation));
		ring_size_ = static_cast<std::size_t>(2.0 * sps_max) + 8;
		ring_.assign(ring_size_, SampleScalar{});
		lock_alpha_ = StateScalar{1} /
		              ((cfg_.lock_average_symbols > StateScalar{})
		               ? cfg_.lock_average_symbols : StateScalar{1});
		reset();
	}

	const config_type& config() const { return cfg_; }

	// --- observable state ---------------------------------------------

	// Current estimate of samples per symbol. Departs from nominal by
	// exactly the clock offset the loop has tracked out.
	StateScalar omega() const { return cfg_.samples_per_symbol + omega_dev_; }
	// The tracked deviation on its own — this is what the integrator holds.
	StateScalar omega_deviation() const { return omega_dev_; }
	// Fractional interpolation position of the most recent symbol, in [0,1).
	StateScalar mu() const { return mu_; }
	// Most recent timing error detector output.
	StateScalar timing_error() const { return last_error_; }
	// Smoothed signal power seen at the symbol instants.
	StateScalar signal_power() const { return sig_power_; }
	// Lock metric: the smoothed MEAN timing error, normalized by signal
	// power, which makes it dimensionless and amplitude-independent.
	//
	// The mean and not the RMS. A locked Gardner loop still produces a
	// sizeable instantaneous error — about 0.37 RMS for unit-power BPSK —
	// because the detector output depends on the data pattern as well as on
	// the timing. That self-noise never goes away; what the loop actually
	// drives to zero is the AVERAGE. Thresholding the RMS would mean lock is
	// never declared no matter how well the loop is tracking.
	//
	// Since the detector output runs as Kp * tau * power, dividing by power
	// leaves roughly Kp * tau — so the metric reads directly as residual
	// timing offset in symbols, scaled by the detector slope.
	StateScalar lock_metric() const {
		using std::abs;
		const StateScalar p = (sig_power_ > tiny_power()) ? sig_power_ : tiny_power();
		return abs(err_mean_) / p;
	}
	// True once the smoothed error has settled below lock_threshold.
	bool is_locked() const { return locked_; }
	// Symbols emitted since construction or reset().
	std::size_t symbols_emitted() const { return symbols_; }

	void reset() {
		std::fill(ring_.begin(), ring_.end(), SampleScalar{});
		newest_      = 0;
		total_       = 0;
		omega_dev_   = StateScalar{};
		mu_          = StateScalar{};
		// Position of the next symbol RELATIVE to the newest sample; the
		// first is due once enough history exists for the interpolator and
		// the mid-symbol tap.
		ahead_       = static_cast<StateScalar>(warmup());
		last_error_  = StateScalar{};
		err_mean_    = StateScalar{};
		sig_power_   = StateScalar{};
		locked_      = false;
		symbols_     = 0;
		have_prev_   = false;
	}

	// --- processing ----------------------------------------------------

	// Feed one input sample. Returns {true, symbol} on the samples where a
	// symbol instant falls, {false, _} otherwise.
	//
	// Pre:  none.
	// Post: all invariants hold; on a strobe the symbol is the interpolated
	//       value at the loop's current estimate of the ideal instant.
	std::pair<bool, SampleScalar> process(SampleScalar in) {
		push(in);
		// Every new sample moves the pending symbol one sample further into
		// the past.
		ahead_ = ahead_ - StateScalar{1};

		// A symbol is due once it is at least one full sample behind the
		// newest — the cubic needs a neighbour on the future side.
		if (total_ <= warmup() || ahead_ > -StateScalar{1})
			return {false, SampleScalar{}};

		const StateScalar delay = -ahead_;               // in [1, 2)
		const SampleScalar y = interpolate(delay);

		const StateScalar e = detect(y, delay);
		last_error_ = e;

		// Second-order loop: proportional term nudges this symbol's
		// position, integral term corrects the rate.
		//
		// Both terms SUBTRACT. Measured open-loop, the Gardner S-curve
		// crosses zero at the correct instant with a POSITIVE slope of about
		// +0.7 per symbol, so a positive error means the loop is sampling
		// LATE and the next symbol has to come sooner. Adding here instead
		// makes the loop drive itself away from lock — it is stable-looking
		// arithmetic that silently never converges, so the direction is
		// worth stating rather than leaving to inspection.
		omega_dev_ = omega_dev_ - k_i_ * e;
		clamp_omega();                                   // I1
		const StateScalar step = cfg_.samples_per_symbol + omega_dev_ - k_p_ * e;

		ahead_ = ahead_ + step;
		// Fractional part of the interpolation position, for observation.
		mu_ = delay - static_cast<StateScalar>(
		          static_cast<long long>(static_cast<double>(delay)));

		// Lock metric: smoothed mean error over smoothed signal power.
		err_mean_  = err_mean_  + lock_alpha_ * (e - err_mean_);
		const StateScalar py = detail::real_of_a_conj_b<StateScalar>(y, y);
		sig_power_ = sig_power_ + lock_alpha_ * (py - sig_power_);
		if (sig_power_ < StateScalar{}) sig_power_ = StateScalar{};      // I3
		locked_ = symbols_ > 4 * static_cast<std::size_t>(
		              static_cast<double>(cfg_.lock_average_symbols)) &&
		          lock_metric() < cfg_.lock_threshold;

		prev_symbol_ = y;
		have_prev_   = true;
		++symbols_;
		return {true, y};
	}

	// Invariants I1-I4, for tests to assert. A predicate rather than an
	// assert(), because CI runs Release where NDEBUG strips assertions.
	bool invariants_hold() const {
		using std::abs;
		const StateScalar nom = cfg_.samples_per_symbol;
		const StateScalar lim = nom * cfg_.max_deviation;
		const StateScalar hi = nom + lim;
		const StateScalar slack = static_cast<StateScalar>(1e-9);
		if (omega_dev_ < -lim - slack || omega_dev_ > lim + slack) return false;  // I1
		if (sig_power_ < StateScalar{}) return false;                   // I3
		if (!(k_p_ > StateScalar{}) || !(k_i_ > StateScalar{})) return false;  // I4
		// I2: the oldest tap any detector reaches for must be in the ring.
		const double reach = 2.0 + static_cast<double>(hi) / 2.0 + 2.0;
		if (reach >= static_cast<double>(ring_size_)) return false;
		return true;
	}

private:
	static StateScalar tiny_power() { return static_cast<StateScalar>(1e-30); }

	static void validate(const config_type& c) {
		if (!(c.samples_per_symbol >= StateScalar{1}))
			throw std::invalid_argument("TimingRecovery: samples_per_symbol must be >= 1");
		if (!(c.loop_bandwidth > StateScalar{}))
			throw std::invalid_argument("TimingRecovery: loop_bandwidth must be > 0");
		if (!(c.damping > StateScalar{}))
			throw std::invalid_argument("TimingRecovery: damping must be > 0");
		if (!(c.detector_gain > StateScalar{}))
			throw std::invalid_argument("TimingRecovery: detector_gain must be > 0");
		if (!(c.max_deviation > StateScalar{}) || c.max_deviation >= StateScalar{1})
			throw std::invalid_argument("TimingRecovery: max_deviation must be in (0, 1)");
		if (c.detector == TimingDetector::gardner &&
		    !(c.samples_per_symbol >= StateScalar(2)))
			throw std::invalid_argument(
				"TimingRecovery: the Gardner detector needs the mid-symbol "
				"sample, so samples_per_symbol must be >= 2 (use "
				"Mueller-Muller at 1 sample/symbol)");
	}

	void compute_gains() {
		const StateScalar z  = cfg_.damping;
		const StateScalar bt = cfg_.loop_bandwidth;
		const StateScalar theta = bt / (z + StateScalar{1} / (StateScalar(4) * z));
		const StateScalar den = StateScalar{1} + StateScalar(2) * z * theta + theta * theta;
		k_p_ = (StateScalar(4) * z * theta / den) / cfg_.detector_gain;
		k_i_ = (StateScalar(4) * theta * theta / den) / cfg_.detector_gain;
	}

	std::size_t warmup() const {
		// Enough history for the mid-symbol tap plus the cubic's neighbours.
		return static_cast<std::size_t>(
			2.0 * static_cast<double>(cfg_.samples_per_symbol)) + 4;
	}

	void push(SampleScalar x) {
		ring_[newest_] = x;
		newest_ = (newest_ + 1) % ring_size_;
		++total_;
	}

	// Sample `k` steps back from the newest (k = 0 is the newest).
	const SampleScalar& at(std::size_t k) const {
		const std::size_t idx = (newest_ + ring_size_ - 1 - (k % ring_size_)) % ring_size_;
		return ring_[idx];
	}

	// Signal value at `delay` samples before the newest sample.
	//
	// Larger delay means older, so the four taps in TIME order run from the
	// larger index to the smaller: s[0] = at(i+2) is oldest. The point sits
	// between at(i+1) and at(i), a fraction (1 - f) of the way from the
	// older to the newer — which is why the Farrow argument is 1 - f and
	// not f. Getting that inversion wrong yields a loop that locks to the
	// wrong instant, so it is spelled out.
	SampleScalar interpolate(StateScalar delay) const {
		const double d = static_cast<double>(delay);
		std::size_t i = static_cast<std::size_t>(d);
		const StateScalar f = delay - static_cast<StateScalar>(i);
		if (i < 1) i = 1;
		const SampleScalar s[4] = { at(i + 2), at(i + 1), at(i), at(i - 1) };
		return detail::farrow_cubic<StateScalar>(s, StateScalar{1} - f);
	}

	StateScalar detect(const SampleScalar& y, StateScalar delay) {
		if (!have_prev_) return StateScalar{};

		if (cfg_.detector == TimingDetector::gardner) {
			// Midpoint sits half a symbol further back than this symbol.
			const SampleScalar mid = interpolate(delay + omega() / StateScalar(2));
			const SampleScalar diff = difference(y, prev_symbol_);
			return detail::real_of_a_conj_b<StateScalar>(diff, mid);
		}
		// Mueller-Muller: Re{ conj(a[k-1])*y[k] - conj(a[k])*y[k-1] }
		//
		// NEGATED, so that both detectors hand the loop the same convention:
		// positive means sampling late. Measured open-loop the two S-curves
		// have opposite slopes at the zero crossing — Gardner about +0.7 per
		// symbol, Mueller-Muller about -1.8 — so sharing one loop without
		// this negation makes whichever detector was not tuned for drive
		// itself away from lock. Normalizing here keeps the sign convention
		// stated in one place rather than duplicated into the loop.
		const SampleScalar a_now  = detail::unit_slice(y);
		const SampleScalar a_prev = detail::unit_slice(prev_symbol_);
		return -(detail::real_of_a_conj_b<StateScalar>(y, a_prev)
		       - detail::real_of_a_conj_b<StateScalar>(prev_symbol_, a_now));
	}

	static SampleScalar difference(const SampleScalar& a, const SampleScalar& b) {
		if constexpr (detail::ComplexSample<SampleScalar>) {
			return SampleScalar(a.real() - b.real(), a.imag() - b.imag());
		} else {
			return static_cast<SampleScalar>(a - b);
		}
	}

	void clamp_omega() {
		const StateScalar lim = cfg_.samples_per_symbol * cfg_.max_deviation;
		if (omega_dev_ < -lim) omega_dev_ = -lim;
		if (omega_dev_ >  lim) omega_dev_ =  lim;
	}

	config_type cfg_;
	StateScalar k_p_{}, k_i_{};
	StateScalar lock_alpha_{};

	std::vector<SampleScalar> ring_;
	std::size_t ring_size_ = 0;
	std::size_t newest_    = 0;
	std::size_t total_     = 0;

	// The integrator holds the DEVIATION from nominal, not the absolute
	// samples-per-symbol.
	//
	// This is a precision decision, not a stylistic one. The integral step is
	// k_i * error, which for Bn*T = 0.01 against typical detector self-noise
	// runs about 1e-4. Held as an absolute ~2.0, that increment sits below
	// posit16's ULP at that magnitude (~1e-3), so the integrator cannot take
	// a small step at all: it either ignores the correction or jumps a whole
	// ULP, and under noise the result is a random walk that rails at the
	// deviation clamp with the eye shut. Held as a deviation the state sits
	// near zero, where the same type's ULP is orders of magnitude finer and
	// the increment lands normally. Measured, this is the difference between
	// posit16 failing at every loop bandwidth and tracking correctly.
	StateScalar omega_dev_{};
	StateScalar mu_{};
	// Position of the next symbol relative to the newest input sample:
	// positive means still in the future, negative already past.
	//
	// RELATIVE, not absolute. An absolute time accumulator grows without
	// bound — 8000 symbols at 2 samples each reaches 16000, where posit16's
	// ULP is about 8. A step of ~2 then cannot advance it at all and the
	// loop freezes with the eye shut, which is precisely how posit16 failed
	// before this was changed. Kept relative the value stays inside
	// [-1, ~2], where every scalar type this library supports has ample
	// resolution, and nothing degrades over an arbitrarily long run.
	StateScalar ahead_{};
	StateScalar last_error_{};
	StateScalar err_mean_{};
	StateScalar sig_power_{};
	bool        locked_  = false;
	std::size_t symbols_ = 0;

	SampleScalar prev_symbol_{};
	bool         have_prev_ = false;
};

} // namespace sw::dsp::sdr
