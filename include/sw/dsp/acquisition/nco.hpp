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
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>

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
	NCO(SampleScalar frequency, SampleScalar sample_rate)
		: phase_{},
		  phase_offset_{} {
		if (!(sample_rate > SampleScalar{}))
			throw std::invalid_argument("NCO: sample_rate must be positive");
		set_frequency(frequency, sample_rate);
	}

	// Set output frequency. The phase increment is frequency / sample_rate,
	// normalized so 1.0 = one full cycle.
	void set_frequency(SampleScalar frequency, SampleScalar sample_rate) {
		if (!(sample_rate > SampleScalar{}))
			throw std::invalid_argument("NCO: sample_rate must be positive");
		phase_inc_ = static_cast<StateScalar>(frequency)
		           / static_cast<StateScalar>(sample_rate);
	}

	// Set a fixed phase offset (in normalized units, 1.0 = full cycle)
	void set_phase_offset(StateScalar offset) {
		phase_offset_ = offset;
	}

	// Get the current phase accumulator value [0, 1)
	StateScalar phase() const { return phase_; }

	// Get the phase increment per sample
	StateScalar phase_increment() const { return phase_inc_; }

	// Generate a single complex I/Q sample and advance the phase.
	complex_t generate_sample() {
		StateScalar total_phase = phase_ + phase_offset_;
		double angle = static_cast<double>(total_phase) * two_pi;

		SampleScalar i_out = static_cast<SampleScalar>(std::cos(angle));
		SampleScalar q_out = static_cast<SampleScalar>(std::sin(angle));

		phase_ = phase_ + phase_inc_;
		wrap_phase();

		return complex_t(i_out, q_out);
	}

	// Generate a single real (cosine) sample and advance the phase.
	SampleScalar generate_real() {
		StateScalar total_phase = phase_ + phase_offset_;
		double angle = static_cast<double>(total_phase) * two_pi;

		SampleScalar out = static_cast<SampleScalar>(std::cos(angle));

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
				SampleScalar{} - input[i] * lo.imag());
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

	void wrap_phase() {
		double p = static_cast<double>(phase_);
		if (p >= 1.0 || p < 0.0) {
			p = p - std::floor(p);
			phase_ = static_cast<StateScalar>(p);
		}
	}
};

} // namespace sw::dsp
