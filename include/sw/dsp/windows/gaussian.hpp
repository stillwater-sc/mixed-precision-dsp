#pragma once
// gaussian.hpp: Gaussian window
//
// w[n] = exp(-0.5 * ((n - (N-1)/2) / (sigma * (N-1)/2))^2)
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

template <DspField T>
mtl::vec::dense_vector<T> gaussian_window(std::size_t length, double sigma = 0.4) {
	mtl::vec::dense_vector<T> w(length);
	if (length <= 1) { if (length == 1) w[0] = T{1}; return w; }
	double half_N = static_cast<double>(length - 1) * 0.5;
	double denom = sigma * half_N;
	for (std::size_t n = 0; n < length; ++n) {
		double x = (static_cast<double>(n) - half_N) / denom;
		w[n] = static_cast<T>(std::exp(-0.5 * x * x));
	}
	return w;
}

} // namespace sw::dsp
