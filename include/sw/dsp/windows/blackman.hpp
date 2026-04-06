#pragma once
// blackman.hpp: Blackman window
//
// w[n] = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))
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
mtl::vec::dense_vector<T> blackman_window(std::size_t length) {
	mtl::vec::dense_vector<T> w(length);
	if (length <= 1) { if (length == 1) w[0] = T{1}; return w; }
	double N = static_cast<double>(length - 1);
	for (std::size_t n = 0; n < length; ++n) {
		double x = static_cast<double>(n) / N;
		w[n] = static_cast<T>(0.42 - 0.5 * std::cos(two_pi * x) + 0.08 * std::cos(2.0 * two_pi * x));
	}
	return w;
}

} // namespace sw::dsp
