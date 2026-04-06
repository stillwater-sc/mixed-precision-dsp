#pragma once
// sampling.hpp: basic sample rate conversion utilities
//
// Simple integer-factor upsampling (zero-insert) and downsampling (decimate).
// For quality resampling with anti-aliasing, use the polyphase resampler
// in conditioning/ (Phase 8).
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <stdexcept>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

// Upsample by integer factor: insert (factor-1) zeros between each sample.
// Output length = input.size() * factor.
// Note: this is zero-insertion only. Apply a lowpass filter after to
// interpolate (see conditioning/interpolator.hpp).
template <DspField T>
mtl::vec::dense_vector<T> upsample(const mtl::vec::dense_vector<T>& input, std::size_t factor) {
	if (factor == 0) throw std::invalid_argument("upsample: factor must be > 0");
	std::size_t out_len = input.size() * factor;
	mtl::vec::dense_vector<T> output(out_len, T{});
	for (std::size_t i = 0; i < input.size(); ++i) {
		output[i * factor] = input[i];
	}
	return output;
}

// Downsample by integer factor: keep every factor-th sample.
// Output length = input.size() / factor.
// Note: apply a lowpass anti-aliasing filter before downsampling
// to avoid aliasing (see conditioning/decimator.hpp).
template <DspField T>
mtl::vec::dense_vector<T> downsample(const mtl::vec::dense_vector<T>& input, std::size_t factor) {
	if (factor == 0) throw std::invalid_argument("downsample: factor must be > 0");
	std::size_t out_len = input.size() / factor;
	mtl::vec::dense_vector<T> output(out_len);
	for (std::size_t i = 0; i < out_len; ++i) {
		output[i] = input[i * factor];
	}
	return output;
}

} // namespace sw::dsp
