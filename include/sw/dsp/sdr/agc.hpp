#pragma once
// agc.hpp: automatic gain control.
//
// Normalizes signal amplitude ahead of demodulation so a demapper sees the
// constellation at the scale it expects, whatever the path loss. This is the
// first genuinely STATEFUL piece of the SDR module: a feedback loop with
// memory, whose precision shows up as settling time and residual ripple
// rather than as a static error.
//
// THE LOOP IS CLOSED IN THE LOG DOMAIN, which is what buys the wide dynamic
// range the requirement asks for. Two reasons:
//
//   * A multiplicative correction becomes additive, so the loop's
//     convergence rate is the same whether it is climbing out of -60 dB or
//     trimming 3 dB. A linear-gain loop is sluggish when the gain is small
//     and twitchy when it is large.
//   * The gain never has to be represented as a huge or tiny number. 60 dB
//     of range is 1000x linear but only 6.9 nepers, which any scalar type
//     holds comfortably.
//
// The state is stored in nepers (natural log) because that is what the loop
// arithmetic wants; the public interface is in dB because that is what
// engineers want.
//
// ATTACK AND DECAY. "Attack" is the fast direction, taken when the signal is
// TOO LOUD and the gain must come down before something clips. "Decay" (also
// called release) is the slow direction, taken when the signal is too quiet.
// The asymmetry is deliberate: a fast release would pump the noise floor up
// between bursts. Both are specified as time constants in seconds, converted
// to per-sample one-pole coefficients against the configured sample rate.
//
// SampleScalar may be real or complex — an SDR chain carries I/Q — while
// StateScalar is always real, because a gain and a level are real quantities
// no matter what they multiply. That is why this takes two scalar parameters
// where a filter in this library takes three: there are no coefficients to
// carry a precision of their own.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>

namespace sw::dsp::sdr {

// How the loop measures the level it is regulating.
enum class LevelDetector {
	// Instantaneous |y|. Reacts immediately, so the loop sees the modulation
	// as well as the envelope — fine for constant-envelope signals, noisy
	// for QAM.
	magnitude,
	// sqrt of a one-pole average of |y|^2. Smooths across the constellation
	// so the loop regulates average power rather than chasing each symbol.
	// The default, and what a QAM receiver wants.
	rms
};

// Loop configuration. Times are in seconds against `sample_rate_hz`; leaving
// the rate at 1 makes them counts of samples instead, which is convenient in
// tests and for rate-agnostic code.
template <DspField T = double>
struct AgcConfig {
	T reference_level   = T{1};        // target output amplitude
	T attack_time_s     = T(0.001);    // fast: signal too loud, gain coming down
	T decay_time_s      = T(0.100);    // slow: signal too quiet, gain coming up
	T averaging_time_s  = T(0.001);    // rms detector window (rms mode only)
	T sample_rate_hz    = T{1};
	T min_gain_db       = T(-60);
	T max_gain_db       = T(60);
	T initial_gain_db   = T{0};
	LevelDetector detector = LevelDetector::rms;
};

namespace detail {

// A sample type that exposes real()/imag() is treated as complex.
template <typename S>
concept ComplexSample = requires(S s) { s.real(); s.imag(); };

// |x| as a StateScalar, for real or complex samples.
template <DspField State, typename S>
State sample_magnitude(const S& x) {
	if constexpr (ComplexSample<S>) {
		const State re = static_cast<State>(x.real());
		const State im = static_cast<State>(x.imag());
		using std::sqrt;
		return sqrt(re * re + im * im);
	} else {
		using std::abs;
		return static_cast<State>(abs(x));
	}
}

// x * g, for real or complex samples with a real gain. Written
// component-wise rather than relying on a mixed-type operator*, which is not
// guaranteed across every complex/scalar pairing this library supports.
template <DspField State, typename S>
S scale_sample(const S& x, State g) {
	if constexpr (ComplexSample<S>) {
		using Component = std::remove_cvref_t<decltype(x.real())>;
		const Component c = static_cast<Component>(g);
		return S(x.real() * c, x.imag() * c);
	} else {
		return static_cast<S>(x * static_cast<S>(g));
	}
}

} // namespace detail

// ============================================================================
// AutomaticGainControl
// ============================================================================

template <DspField StateScalar = double, typename SampleScalar = StateScalar>
class AutomaticGainControl {
public:
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	using config_type   = AgcConfig<StateScalar>;

	// Establishes every invariant. Throws std::invalid_argument if the
	// configuration cannot support them (I5), which is the only way they can
	// fail — thereafter the loop maintains them itself.
	explicit AutomaticGainControl(const config_type& cfg = config_type{})
		: cfg_(cfg) {
		validate(cfg_);
		attack_coeff_ = one_pole_coeff(cfg_.attack_time_s);
		decay_coeff_  = one_pole_coeff(cfg_.decay_time_s);
		avg_coeff_    = one_pole_coeff(cfg_.averaging_time_s);
		// The error is formed against a floored level, so a dead-silent
		// input produces a large-but-finite correction rather than the
		// -infinity that log(0) would give. The gain clamp then bounds the
		// outcome; without the floor a single silent sample would poison the
		// state with a NaN that never washes out.
		using std::log;
		level_floor_ = cfg_.reference_level * static_cast<StateScalar>(1e-9);
		reset();
	}

	// --- configuration -------------------------------------------------

	const config_type& config() const { return cfg_; }

	// --- state ---------------------------------------------------------

	// Current linear gain.
	StateScalar gain() const { return gain_linear_; }

	// Current gain in dB. Always within [min_gain_db, max_gain_db] (I1).
	StateScalar gain_db() const { return nepers_to_db(log_gain_); }

	// Most recent level estimate, in the same units as reference_level.
	StateScalar level() const { return level_; }

	// Restore the initial gain and clear the detector.
	void reset() {
		set_gain_db(cfg_.initial_gain_db);
		level_ = StateScalar{};
		power_ = StateScalar{};
		primed_ = false;
	}

	// Force the gain, clamped into range so I1 cannot be broken from
	// outside. Useful for seeding the loop from a known link budget.
	void set_gain_db(StateScalar db) {
		if (db < cfg_.min_gain_db) db = cfg_.min_gain_db;
		if (db > cfg_.max_gain_db) db = cfg_.max_gain_db;
		log_gain_ = db_to_nepers(db);
		refresh_linear_gain();
	}

	// --- processing ----------------------------------------------------

	// Apply the current gain, then update it from what that produced.
	//
	// Pre:  none.
	// Post: returns gain_before * in; the gain has moved toward the value
	//       that would have put the measured level at reference_level, by
	//       one attack or decay step; all invariants hold.
	//
	// The gain is applied BEFORE the update, so the loop is strictly causal
	// and a caller can reason about latency: sample n is scaled by the gain
	// derived from samples 0..n-1.
	SampleScalar process(SampleScalar in) {
		const SampleScalar out = detail::scale_sample<StateScalar>(in, gain_linear_);
		update(detail::sample_magnitude<StateScalar>(out));
		return out;
	}

	// Block form. Equivalent to calling process() in order.
	mtl::vec::dense_vector<SampleScalar>
	process_block(std::span<const SampleScalar> input) {
		mtl::vec::dense_vector<SampleScalar> out(input.size());
		for (std::size_t i = 0; i < input.size(); ++i) out[i] = process(input[i]);
		return out;
	}

	// --- invariants ----------------------------------------------------

	// I1-I4, checkable at runtime. Exposed as a predicate rather than
	// enforced with assert(): this project's tests run in Release where
	// NDEBUG strips assertions, so an invariant worth stating is worth
	// asserting from a test.
	bool invariants_hold() const {
		using std::abs;
		using std::exp;
		const StateScalar db = gain_db();
		// I1, with a rounding allowance: the clamp is applied in nepers and
		// read back in dB.
		const StateScalar slack = static_cast<StateScalar>(1e-6);
		if (db < cfg_.min_gain_db - slack) return false;
		if (db > cfg_.max_gain_db + slack) return false;
		// I2: the cached linear gain still matches the log state.
		const StateScalar want = exp(log_gain_);
		const StateScalar denom = (want > StateScalar{}) ? want : StateScalar{1};
		if (abs(gain_linear_ - want) > static_cast<StateScalar>(1e-6) * denom)
			return false;
		// I3
		if (level_ < StateScalar{}) return false;
		// I4
		for (StateScalar c : {attack_coeff_, decay_coeff_, avg_coeff_})
			if (!(c > StateScalar{}) || c > StateScalar{1}) return false;
		return true;
	}

private:
	static void validate(const config_type& c) {
		auto positive = [](StateScalar v) { return v > StateScalar{}; };
		if (!positive(c.reference_level))
			throw std::invalid_argument("AGC: reference_level must be > 0");
		if (!positive(c.attack_time_s))
			throw std::invalid_argument("AGC: attack_time_s must be > 0");
		if (!positive(c.decay_time_s))
			throw std::invalid_argument("AGC: decay_time_s must be > 0");
		if (!positive(c.averaging_time_s))
			throw std::invalid_argument("AGC: averaging_time_s must be > 0");
		if (!positive(c.sample_rate_hz))
			throw std::invalid_argument("AGC: sample_rate_hz must be > 0");
		if (c.min_gain_db > c.max_gain_db)
			throw std::invalid_argument("AGC: min_gain_db must be <= max_gain_db");
	}

	// One-pole coefficient for a time constant, clamped into (0, 1] so I4
	// holds even for a time constant far shorter than one sample.
	StateScalar one_pole_coeff(StateScalar tau_s) const {
		using std::exp;
		const StateScalar n = tau_s * cfg_.sample_rate_hz;   // samples
		StateScalar a = (n > StateScalar{})
			? StateScalar{1} - exp(-StateScalar{1} / n)
			: StateScalar{1};
		if (!(a > StateScalar{})) a = std::numeric_limits<StateScalar>::min();
		if (a > StateScalar{1})   a = StateScalar{1};
		return a;
	}

	static StateScalar db_to_nepers(StateScalar db) {
		// dB is 20*log10(x); nepers is ln(x). ln(10)/20 converts between.
		return db * static_cast<StateScalar>(2.302585092994046 / 20.0);
	}
	static StateScalar nepers_to_db(StateScalar np) {
		return np * static_cast<StateScalar>(20.0 / 2.302585092994046);
	}

	void refresh_linear_gain() {
		using std::exp;
		gain_linear_ = exp(log_gain_);   // re-establishes I2
	}

	// Fold one output magnitude into the detector, then take one loop step.
	void update(StateScalar mag) {
		// --- detector ---
		if (cfg_.detector == LevelDetector::magnitude) {
			level_ = mag;
		} else {
			const StateScalar p = mag * mag;
			// Seed on the first sample so the loop does not spend its first
			// time constant climbing out of a zero that was never measured.
			power_ = primed_ ? power_ + avg_coeff_ * (p - power_) : p;
			using std::sqrt;
			level_ = sqrt(power_);
		}
		primed_ = true;

		// --- error, in the log domain ---
		using std::log;
		const StateScalar floored = (level_ > level_floor_) ? level_ : level_floor_;
		const StateScalar err = log(cfg_.reference_level) - log(floored);

		// --- loop step: fast when reducing gain, slow when raising it ---
		const StateScalar rate = (err < StateScalar{}) ? attack_coeff_ : decay_coeff_;
		log_gain_ = log_gain_ + rate * err;

		// --- clamp (I1) ---
		const StateScalar lo = db_to_nepers(cfg_.min_gain_db);
		const StateScalar hi = db_to_nepers(cfg_.max_gain_db);
		if (log_gain_ < lo) log_gain_ = lo;
		if (log_gain_ > hi) log_gain_ = hi;

		refresh_linear_gain();
	}

	config_type cfg_;
	StateScalar attack_coeff_{};
	StateScalar decay_coeff_{};
	StateScalar avg_coeff_{};
	StateScalar level_floor_{};

	StateScalar log_gain_{};      // nepers
	StateScalar gain_linear_{1};  // exp(log_gain_)
	StateScalar level_{};
	StateScalar power_{};
	bool        primed_ = false;
};

} // namespace sw::dsp::sdr
