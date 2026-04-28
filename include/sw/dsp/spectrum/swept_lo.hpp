#pragma once
// swept_lo.hpp: spectrum-analyzer swept local-oscillator (chirp generator).
//
// In a swept-tuned analyzer, the LO walks across the input band so the
// downstream mixer + RBW filter sees one frequency at a time. SweptLO
// is a phase-coherent chirp generator that produces (cos, sin) at a
// frequency that varies linearly or logarithmically from f_start to
// f_stop over a configurable duration, then restarts.
//
// Built on the same phase-accumulator pattern as NCO
// (acquisition/nco.hpp) — the only difference is that the per-sample
// phase increment is itself a function of time:
//
//   Linear:       phase_inc(n) = phase_inc_start + n * delta_inc
//                 where delta_inc = (phase_inc_stop - phase_inc_start) / N
//
//   Logarithmic:  phase_inc(n) = phase_inc_start * ratio_inc^n
//                 where ratio_inc = (phase_inc_stop / phase_inc_start)^(1/(N-1))
//
// The phase ACCUMULATOR is continuous across the sweep boundary: at
// the end of one sweep the phase increment snaps back to phase_inc_
// start but the phase itself keeps walking forward. No discontinuity
// glitch in the cos/sin output.
//
// Mixed-precision contract:
//   - CoeffScalar holds the design-time sweep parameters (delta_inc
//     for linear, ratio_inc for log). For log sweeps over many
//     samples, ratio_inc is very close to 1 — narrow CoeffScalar
//     types may lose precision in this small-difference regime.
//     Coefficients are computed in double and cast to CoeffScalar
//     at construction (same convention as VBW/RBW filters).
//   - StateScalar holds the phase accumulator and the running
//     phase_inc. Phase precision sets SFDR at fixed frequency,
//     same constraint as NCO.
//   - SampleScalar is the cos/sin output type.
//   - DenormalPrevention<SampleScalar> AC injection on each output —
//     same convention as NCO.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/math/denormal.hpp>

namespace sw::dsp::spectrum {

template <DspField CoeffScalar  = double,
          DspField StateScalar  = CoeffScalar,
          DspField SampleScalar = StateScalar>
class SweptLO {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;

	enum class Sweep { Linear, Logarithmic };

	// f_start_hz:        starting frequency. > 0 finite.
	// f_stop_hz:         ending frequency. > 0 finite. May be < f_start
	//                    for a downward sweep.
	// sweep_duration_s:  > 0 finite. Mapped to N samples via
	//                    floor(sweep_duration_s * sample_rate_hz);
	//                    must yield N >= 2.
	// sample_rate_hz:    > 0 finite.
	// mode:              Linear or Logarithmic. Log requires
	//                    f_start * f_stop > 0 (same sign).
	SweptLO(double f_start_hz, double f_stop_hz,
	        double sweep_duration_s, double sample_rate_hz,
	        Sweep mode = Sweep::Linear)
		: f_start_hz_(f_start_hz),
		  f_stop_hz_(f_stop_hz),
		  sweep_duration_s_(sweep_duration_s),
		  sample_rate_hz_(sample_rate_hz),
		  mode_(mode) {
		validate_inputs();
		// Number of samples per sweep. floor() ensures we don't
		// over-shoot the requested duration.
		const double N_d = std::floor(sweep_duration_s * sample_rate_hz);
		if (!(N_d >= 2.0))
			throw std::invalid_argument(
				"SweptLO: sweep_duration_s * sample_rate_hz must yield >= 2 "
				"samples (got "
				+ std::to_string(N_d) + ")");
		num_sweep_samples_ = static_cast<std::size_t>(N_d);

		// Normalized phase increments (cycles per sample) at start/stop.
		const double inc_start_d = f_start_hz / sample_rate_hz;
		const double inc_stop_d  = f_stop_hz  / sample_rate_hz;

		// Design math in double, project to CoeffScalar at the end —
		// same rationale as VBW/RBW: intermediate constants like the
		// sweep delta or log ratio may sit very close to 0 or 1 and
		// benefit from double's range while we compute them.
		if (mode == Sweep::Linear) {
			delta_inc_ = static_cast<CoeffScalar>(
				(inc_stop_d - inc_start_d) /
				static_cast<double>(num_sweep_samples_ - 1));
			ratio_inc_ = CoeffScalar{1};   // unused for linear
		} else {
			// Log sweep needs same-sign frequencies; today
			// validate_inputs already rejects non-positive values for
			// both, so the same-sign property is automatic. If
			// negative-frequency support is added later, reintroduce
			// the explicit f_start * f_stop > 0 check here.
			const double ratio_d = std::pow(
				inc_stop_d / inc_start_d,
				1.0 / static_cast<double>(num_sweep_samples_ - 1));
			ratio_inc_ = static_cast<CoeffScalar>(ratio_d);
			delta_inc_ = CoeffScalar{};   // unused for log
		}

		phase_inc_start_ = static_cast<StateScalar>(inc_start_d);
		phase_inc_       = phase_inc_start_;
		two_pi_state_    = static_cast<StateScalar>(two_pi);
	}

	// Generate one (cos, sin) sample at the current sweep phase, then
	// advance both phase and the per-sample phase increment.
	std::pair<SampleScalar, SampleScalar> process() {
		using std::cos; using std::sin;
		const StateScalar angle = phase_ * two_pi_state_;
		SampleScalar c = static_cast<SampleScalar>(cos(angle))
		               + denormal_.ac();
		SampleScalar s = static_cast<SampleScalar>(sin(angle))
		               + denormal_.ac();

		// Advance phase.
		phase_ = phase_ + phase_inc_;
		wrap_phase();

		// Advance phase increment for the NEXT sample.
		++sample_count_;
		if (sample_count_ >= num_sweep_samples_) {
			// Sweep boundary: snap phase_inc back to start, mark
			// completion, but DO NOT touch phase_ (continuous restart).
			phase_inc_      = phase_inc_start_;
			sample_count_   = 0;
			++total_sweeps_;
			just_wrapped_  = true;
		} else {
			just_wrapped_ = false;
			if (mode_ == Sweep::Linear) {
				phase_inc_ = phase_inc_ + static_cast<StateScalar>(delta_inc_);
			} else {
				phase_inc_ = phase_inc_ * static_cast<StateScalar>(ratio_inc_);
			}
		}
		return {c, s};
	}

	// Restart the sweep at f_start with phase_ = 0. Coefficients
	// (delta_inc / ratio_inc) preserved.
	void reset() {
		phase_         = StateScalar{};
		phase_inc_     = phase_inc_start_;
		sample_count_  = 0;
		total_sweeps_  = 0;
		just_wrapped_  = false;
		denormal_      = DenormalPrevention<SampleScalar>{};
	}

	// Current instantaneous frequency in Hz, derived from phase_inc.
	[[nodiscard]] double current_frequency_hz() const {
		return static_cast<double>(phase_inc_) * sample_rate_hz_;
	}

	// True iff the most recent process() call wrapped a sweep boundary.
	// Self-resets on the next process() call (one-shot per sweep).
	[[nodiscard]] bool sweep_complete() const { return just_wrapped_; }

	// Total number of sweep boundaries crossed since construction or
	// the last reset(). Useful for monotone "wait for sweep N" logic.
	[[nodiscard]] std::size_t total_sweeps() const { return total_sweeps_; }

	[[nodiscard]] double f_start_hz()        const { return f_start_hz_; }
	[[nodiscard]] double f_stop_hz()         const { return f_stop_hz_; }
	[[nodiscard]] double sweep_duration_s()  const { return sweep_duration_s_; }
	[[nodiscard]] double sample_rate_hz()    const { return sample_rate_hz_; }
	[[nodiscard]] Sweep  mode()              const { return mode_; }
	[[nodiscard]] std::size_t num_sweep_samples() const { return num_sweep_samples_; }

private:
	void validate_inputs() const {
		auto check_pos_finite = [](double v, const char* name) {
			if (!std::isfinite(v) || !(v > 0.0))
				throw std::invalid_argument(
					std::string("SweptLO: ") + name
					+ " must be positive and finite (got "
					+ std::to_string(v) + ")");
		};
		check_pos_finite(f_start_hz_,       "f_start_hz");
		check_pos_finite(f_stop_hz_,        "f_stop_hz");
		check_pos_finite(sweep_duration_s_, "sweep_duration_s");
		check_pos_finite(sample_rate_hz_,   "sample_rate_hz");
	}

	void wrap_phase() {
		// Normalized phase in [0, 1). For typical sweeps phase_inc < 1
		// so a single subtraction suffices; the loop handles the rare
		// case of a per-sample phase_inc near or above 1.
		while (phase_ >= StateScalar{1}) phase_ = phase_ - StateScalar{1};
		while (phase_ <  StateScalar{}) phase_ = phase_ + StateScalar{1};
	}

	double      f_start_hz_;
	double      f_stop_hz_;
	double      sweep_duration_s_;
	double      sample_rate_hz_;
	Sweep       mode_;
	std::size_t num_sweep_samples_ = 0;
	std::size_t sample_count_      = 0;
	std::size_t total_sweeps_      = 0;
	bool        just_wrapped_      = false;

	StateScalar phase_           = StateScalar{};
	StateScalar phase_inc_       = StateScalar{};
	StateScalar phase_inc_start_ = StateScalar{};
	StateScalar two_pi_state_    = StateScalar{};

	CoeffScalar delta_inc_ = CoeffScalar{};
	CoeffScalar ratio_inc_ = CoeffScalar{1};

	DenormalPrevention<SampleScalar> denormal_;
};

} // namespace sw::dsp::spectrum
