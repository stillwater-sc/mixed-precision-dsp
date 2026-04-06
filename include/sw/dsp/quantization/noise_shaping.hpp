#pragma once
// noise_shaping.hpp: error-feedback noise shaping
//
// Feeds the quantization error back through a filter to reshape
// the noise spectrum, pushing noise energy out of the band of
// interest. First-order noise shaping is the simplest form.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

// First-order error-feedback noise shaper.
//
// For each sample:
//   shaped = input + error_feedback
//   quantized = quantize(shaped)    // conversion to OutputT
//   error = shaped - quantized      // quantization error
//   error_feedback = -error         // feed back for next sample
//
// This pushes noise energy to higher frequencies (first-order
// high-pass shaping of the noise floor).
template <DspField HighPrecT, DspField LowPrecT>
class FirstOrderNoiseShaper {
public:
	LowPrecT process(HighPrecT input) {
		HighPrecT shaped = input + error_feedback_;
		LowPrecT quantized = static_cast<LowPrecT>(shaped);
		HighPrecT reconstructed = static_cast<HighPrecT>(quantized);
		HighPrecT error = shaped - reconstructed;
		error_feedback_ = HighPrecT{} - error;  // negative feedback
		return quantized;
	}

	mtl::vec::dense_vector<LowPrecT> process(const mtl::vec::dense_vector<HighPrecT>& input) {
		mtl::vec::dense_vector<LowPrecT> output(input.size());
		for (std::size_t i = 0; i < input.size(); ++i) {
			output[i] = process(input[i]);
		}
		return output;
	}

	void reset() { error_feedback_ = HighPrecT{}; }

private:
	HighPrecT error_feedback_{};
};

} // namespace sw::dsp
