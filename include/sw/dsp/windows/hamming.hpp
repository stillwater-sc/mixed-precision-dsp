#pragma once
// hamming.hpp: Hamming window
//
// w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>

namespace sw::dsp {

template <DspField T>
mtl::vec::dense_vector<T> hamming_window(std::size_t length) {
	mtl::vec::dense_vector<T> w(length);
	if (length <= 1) { if (length == 1) w[0] = T{1}; return w; }
	double N = static_cast<double>(length - 1);
	for (std::size_t n = 0; n < length; ++n) {
		w[n] = static_cast<T>(0.54 - 0.46 * std::cos(two_pi * static_cast<double>(n) / N));
	}
	return w;
}

} // namespace sw::dsp
