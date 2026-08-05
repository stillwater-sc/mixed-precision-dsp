#pragma once
// nco.hpp: Numerically Controlled Oscillator for digital mixing
//
// An NCO generates complex sinusoids (I/Q) for digital down-conversion
// (DDC) and up-conversion (DUC) chains. The phase accumulator width
// directly determines spurious-free dynamic range (SFDR):
//   SFDR ~ 6.02 * W dB for a W-bit phase accumulator with truncation.
//
// Posit's tapered precision near +/-1 can provide better SFDR than
// fixed-point at the same bit width — a key mixed-precision finding.
//
// Two-scalar parameterization:
//   StateScalar  — phase accumulator (determines SFDR)
//   SampleScalar — output I/Q samples (streaming precision)
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
#include <sw/dsp/math/denormal.hpp>

namespace sw::dsp {

// Numerically Controlled Oscillator
//
// Generates complex sinusoids by accumulating phase and computing
// sin/cos at each sample. The phase accumulator operates in normalized
// units [0, 1) representing one full cycle, avoiding precision loss
// from large radian values.
//
// StateScalar controls the phase accumulator resolution and thus SFDR.
// SampleScalar controls the output I/Q precision.
template <DspField StateScalar = double,
          DspField SampleScalar = StateScalar>
class NCO {
public:
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	using complex_t     = complex_for_t<SampleScalar>;

	// Construct an NCO with the given output frequency and sample rate.
	// Frequency can be positive (counter-clockwise) or negative (clockwise).
	//
	// FREQUENCY AND SAMPLE RATE ARE `double`, NOT `StateScalar`. They are
	// configuration inputs rather than datapath state, and holding them at
	// the datapath's precision gains nothing while costing a great deal: a
	// realistic RF pair like 1.2 GHz on a 5 GSPS front end overflows every
	// narrow state type on the way in, even though the ratio it encodes is
	// always in [0, 0.5) and every one of those types represents it
	// comfortably. Converting the arguments before the division that brings
	// them back into range produced a NaN phase accumulator and NaN output
	// samples thereafter, with nothing to indicate why. (issue #207)
	// Accepts any pair convertible to double — including StateScalar, so
	// existing callers keep working — and converts BEFORE the division
	// rather than after, which is the whole point.
	template <typename F, typename R>
	NCO(const F& frequency, const R& sample_rate)
		: phase_{},
		  phase_offset_{},
		  two_pi_state_(static_cast<StateScalar>(two_pi)) {
		set_frequency(static_cast<double>(frequency), static_cast<double>(sample_rate));
	}

	// Set output frequency. The phase increment is frequency / sample_rate,
	// normalized so 1.0 = one full cycle.
	//
	// The ratio is formed in double and only its result is converted, so
	// absolute Hz work for every StateScalar rather than only the wide ones.
	template <typename F, typename R>
	void set_frequency(const F& frequency, const R& sample_rate) {
		set_frequency(static_cast<double>(frequency), static_cast<double>(sample_rate));
	}

	void set_frequency(double frequency, double sample_rate) {
		if (!(sample_rate > 0.0))
			throw std::invalid_argument("NCO: sample_rate must be positive");
		const double inc = frequency / sample_rate;
		if (!std::isfinite(inc))
			throw std::invalid_argument(
				"NCO: frequency / sample_rate is not finite");
		phase_inc_ = static_cast<StateScalar>(inc);
		// Post-condition: a phase increment that is not finite in the state
		// type would poison the accumulator and every sample after it, with
		// no other signal that anything went wrong. (issue #207)
		if (!is_finite_state(phase_inc_))
			throw std::invalid_argument(
				"NCO: phase increment " + std::to_string(inc) +
				" is not representable in the configured StateScalar");
	}

	// Finiteness test that works for the native types and for Universal's,
	// which do not all specialize std::isfinite.
	static bool is_finite_state(StateScalar v) {
		const double d = static_cast<double>(v);
		return std::isfinite(d);
	}

	// Set a fixed phase offset (in normalized units, 1.0 = full cycle)
	void set_phase_offset(StateScalar offset) {
		using std::floor;
		phase_offset_ = offset - floor(offset);
	}

	// Get the current phase accumulator value [0, 1)
	StateScalar phase() const { return phase_; }

	// Get the phase increment per sample
	StateScalar phase_increment() const { return phase_inc_; }

	// Generate a single complex I/Q sample and advance the phase.
	complex_t generate_sample() {
		using std::cos; using std::sin;
		StateScalar angle = (phase_ + phase_offset_) * two_pi_state_;

		SampleScalar i_out = static_cast<SampleScalar>(cos(angle))
		                   + denormal_.ac();
		SampleScalar q_out = static_cast<SampleScalar>(sin(angle))
		                   + denormal_.ac();

		phase_ = phase_ + phase_inc_;
		wrap_phase();

		return complex_t(i_out, q_out);
	}

	// Generate a single real (cosine) sample and advance the phase.
	SampleScalar generate_real() {
		using std::cos;
		StateScalar angle = (phase_ + phase_offset_) * two_pi_state_;

		SampleScalar out = static_cast<SampleScalar>(cos(angle))
		                 + denormal_.ac();

		phase_ = phase_ + phase_inc_;
		wrap_phase();

		return out;
	}

	// Block generation: fill a span with complex I/Q samples
	void generate_block(std::span<complex_t> output) {
		for (std::size_t i = 0; i < output.size(); ++i) {
			output[i] = generate_sample();
		}
	}

	// Block generation: return dense_vector of complex I/Q samples
	mtl::vec::dense_vector<complex_t> generate_block(std::size_t length) {
		mtl::vec::dense_vector<complex_t> output(length);
		for (std::size_t i = 0; i < length; ++i) {
			output[i] = generate_sample();
		}
		return output;
	}

	// Block generation: fill a span with real (cosine) samples
	void generate_block_real(std::span<SampleScalar> output) {
		for (std::size_t i = 0; i < output.size(); ++i) {
			output[i] = generate_real();
		}
	}

	// Block generation: return dense_vector of real (cosine) samples
	mtl::vec::dense_vector<SampleScalar> generate_block_real(std::size_t length) {
		mtl::vec::dense_vector<SampleScalar> output(length);
		for (std::size_t i = 0; i < length; ++i) {
			output[i] = generate_real();
		}
		return output;
	}

	// Mix (multiply) a real input signal with the NCO conjugate for down-conversion.
	// Returns: input[n] * conj(nco[n]) for each sample.
	mtl::vec::dense_vector<complex_t> mix_down(
			const mtl::vec::dense_vector<SampleScalar>& input) {
		mtl::vec::dense_vector<complex_t> output(input.size());
		for (std::size_t i = 0; i < input.size(); ++i) {
			complex_t lo = generate_sample();
			output[i] = complex_t(
				input[i] * lo.real(),
				-(input[i] * lo.imag()));
		}
		return output;
	}

	void reset() {
		phase_ = StateScalar{};
	}

private:
	StateScalar phase_;
	StateScalar phase_inc_;
	StateScalar phase_offset_;
	StateScalar two_pi_state_;
	DenormalPrevention<SampleScalar> denormal_;

	void wrap_phase() {
		using std::floor;
		StateScalar one{1};
		StateScalar zero{};
		if (phase_ >= one || phase_ < zero) {
			phase_ = phase_ - floor(phase_);
		}
	}
};

} // namespace sw::dsp
