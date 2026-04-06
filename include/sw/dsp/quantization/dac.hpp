#pragma once
// dac.hpp: digital-to-analog reconstruction model
//
// Models reconstruction from a quantized type back to a
// high-precision type. Inverse of ADC.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

// DAC: convert from InputT (quantized) to OutputT (high precision).
template <DspField InputT, DspField OutputT>
class DAC {
public:
	OutputT convert(InputT sample) const {
		return static_cast<OutputT>(sample);
	}

	mtl::vec::dense_vector<OutputT> convert(const mtl::vec::dense_vector<InputT>& input) const {
		mtl::vec::dense_vector<OutputT> output(input.size());
		for (std::size_t i = 0; i < input.size(); ++i) {
			output[i] = static_cast<OutputT>(input[i]);
		}
		return output;
	}
};

} // namespace sw::dsp
