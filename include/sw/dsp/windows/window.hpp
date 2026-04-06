#pragma once
// window.hpp: window application utility
//
// Apply a window to a signal by element-wise multiplication.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

// Apply a window to a signal in-place
template <DspField T>
void apply_window(mtl::vec::dense_vector<T>& signal,
                  const mtl::vec::dense_vector<T>& window) {
	std::size_t n = std::min(signal.size(), window.size());
	for (std::size_t i = 0; i < n; ++i) {
		signal[i] = signal[i] * window[i];
	}
}

// Apply a window, returning a new vector
template <DspField T>
mtl::vec::dense_vector<T> windowed(const mtl::vec::dense_vector<T>& signal,
                                    const mtl::vec::dense_vector<T>& window) {
	std::size_t n = std::min(signal.size(), window.size());
	mtl::vec::dense_vector<T> result(n);
	for (std::size_t i = 0; i < n; ++i) {
		result[i] = signal[i] * window[i];
	}
	return result;
}

} // namespace sw::dsp
