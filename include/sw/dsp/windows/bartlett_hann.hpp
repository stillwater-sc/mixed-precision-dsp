#pragma once
// bartlett_hann.hpp: Bartlett-Hann window
//
// w[n] = 0.62 - 0.48 * |n/(N-1) - 0.5| + 0.38 * cos(2*pi * (n/(N-1) - 0.5))
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
mtl::vec::dense_vector<T> bartlett_hann_window(std::size_t length) {
	mtl::vec::dense_vector<T> w(length);
	if (length <= 1) { if (length == 1) w[0] = T{1}; return w; }
	double N = static_cast<double>(length - 1);
	for (std::size_t n = 0; n < length; ++n) {
		double x = static_cast<double>(n) / N - 0.5;
		w[n] = static_cast<T>(0.62 - 0.48 * std::abs(x) + 0.38 * std::cos(two_pi * x));
	}
	return w;
}

} // namespace sw::dsp
