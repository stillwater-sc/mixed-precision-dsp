#pragma once
// adc.hpp: analog-to-digital conversion model
//
// Models quantization from a high-precision input type to a
// lower-precision output type. The ADC maps a continuous-valued
// input in [-1, 1] to discrete output levels.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

// ADC: convert from InputT (high precision) to OutputT (lower precision).
// Simply performs static_cast, which triggers the target type's quantization.
// For Universal types, this naturally rounds/truncates per the type's rules.
template <DspField InputT, DspField OutputT>
class ADC {
public:
	// Convert a single sample
	OutputT convert(InputT sample) const {
		return static_cast<OutputT>(sample);
	}

	// Convert a vector of samples
	mtl::vec::dense_vector<OutputT> convert(const mtl::vec::dense_vector<InputT>& input) const {
		mtl::vec::dense_vector<OutputT> output(input.size());
		for (std::size_t i = 0; i < input.size(); ++i) {
			output[i] = static_cast<OutputT>(input[i]);
		}
		return output;
	}
};

} // namespace sw::dsp
