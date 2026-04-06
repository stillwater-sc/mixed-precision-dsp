#pragma once
// fir_filter.hpp: FIR filter with circular buffer delay line
//
// Three-scalar parameterization:
//   CoeffScalar  — tap coefficients (designed in high precision)
//   StateScalar  — delay line (accumulation precision)
//   SampleScalar — input/output samples
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <span>
#include <stdexcept>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

template <DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class FIRFilter {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;

	explicit FIRFilter(const mtl::vec::dense_vector<CoeffScalar>& taps)
		: taps_(taps), delay_(taps.size(), StateScalar{}), write_pos_(0) {
		if (taps.size() == 0)
			throw std::invalid_argument("FIRFilter: taps must not be empty");
	}

	// Set new taps (resets state)
	void set_taps(const mtl::vec::dense_vector<CoeffScalar>& taps) {
		if (taps.size() == 0)
			throw std::invalid_argument("FIRFilter: taps must not be empty");
		taps_ = taps;
		delay_ = mtl::vec::dense_vector<StateScalar>(taps.size(), StateScalar{});
		write_pos_ = 0;
	}

	// Process a single sample through the FIR filter
	SampleScalar process(SampleScalar in) {
		std::size_t N = taps_.size();

		// Write input to circular buffer
		delay_[write_pos_] = static_cast<StateScalar>(in);

		// Convolve: y = sum(taps[k] * delay[write_pos - k])
		StateScalar acc{};
		std::size_t idx = write_pos_;
		for (std::size_t k = 0; k < N; ++k) {
			acc = acc + static_cast<StateScalar>(taps_[k]) * delay_[idx];
			idx = (idx == 0) ? N - 1 : idx - 1;
		}

		// Advance write position
		write_pos_ = (write_pos_ + 1) % N;

		return static_cast<SampleScalar>(acc);
	}

	// Process a block of samples in-place
	void process_block(std::span<SampleScalar> samples) {
		for (auto& s : samples) {
			s = process(s);
		}
	}

	// Process a block, writing to output
	void process_block(std::span<const SampleScalar> input,
	                   std::span<SampleScalar> output) {
		std::size_t n = std::min(input.size(), output.size());
		for (std::size_t i = 0; i < n; ++i) {
			output[i] = process(input[i]);
		}
	}

	// Reset delay line to zero
	void reset() {
		for (std::size_t i = 0; i < delay_.size(); ++i) {
			delay_[i] = StateScalar{};
		}
		write_pos_ = 0;
	}

	// Access
	std::size_t order() const { return taps_.size() > 0 ? taps_.size() - 1 : 0; }
	std::size_t num_taps() const { return taps_.size(); }
	const mtl::vec::dense_vector<CoeffScalar>& taps() const { return taps_; }

private:
	mtl::vec::dense_vector<CoeffScalar> taps_;
	mtl::vec::dense_vector<StateScalar> delay_;
	std::size_t write_pos_{0};
};

} // namespace sw::dsp
