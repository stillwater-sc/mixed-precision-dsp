#pragma once
// tukey.hpp: Tukey (cosine-tapered) window
//
// w[n] = { 0.5 * (1 - cos(2*pi*n / (alpha*(N-1))))                  for 0 <= n < alpha*(N-1)/2
//        { 1                                                          for alpha*(N-1)/2 <= n <= (N-1)*(1-alpha/2)
//        { 0.5 * (1 - cos(2*pi*(N-1-n) / (alpha*(N-1))))             for (N-1)*(1-alpha/2) < n <= N-1
//
// alpha=0 → rectangular, alpha=1 → Hanning
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
mtl::vec::dense_vector<T> tukey_window(std::size_t length, double alpha = 0.5) {
	mtl::vec::dense_vector<T> w(length);
	if (length <= 1) { if (length == 1) w[0] = T{1}; return w; }
	if (alpha <= 0.0) {
		for (std::size_t n = 0; n < length; ++n) w[n] = T{1};
		return w;
	}
	if (alpha >= 1.0) {
		double N = static_cast<double>(length - 1);
		for (std::size_t n = 0; n < length; ++n) {
			w[n] = static_cast<T>(0.5 * (1.0 - std::cos(two_pi * static_cast<double>(n) / N)));
		}
		return w;
	}
	double N = static_cast<double>(length - 1);
	double half_taper = alpha * N * 0.5;
	for (std::size_t n = 0; n < length; ++n) {
		double nd = static_cast<double>(n);
		if (nd < half_taper) {
			w[n] = static_cast<T>(0.5 * (1.0 - std::cos(two_pi * nd / (alpha * N))));
		} else if (nd > N - half_taper) {
			w[n] = static_cast<T>(0.5 * (1.0 - std::cos(two_pi * (N - nd) / (alpha * N))));
		} else {
			w[n] = T{1};
		}
	}
	return w;
}

} // namespace sw::dsp
