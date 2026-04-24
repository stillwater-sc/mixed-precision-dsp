#pragma once
// tukey.hpp: Tukey (cosine-tapered) window
//
// w[n] = { 0.5 * (1 - cos(2*pi*n / (alpha*(N-1))))                  for 0 <= n < alpha*(N-1)/2
//        { 1                                                          for alpha*(N-1)/2 <= n <= (N-1)*(1-alpha/2)
//        { 0.5 * (1 - cos(2*pi*(N-1-n) / (alpha*(N-1))))             for (N-1)*(1-alpha/2) < n <= N-1
//
// alpha=0 -> rectangular, alpha=1 -> Hanning.
//
// Intermediate math runs in T; ADL trig for Universal types. `alpha`
// stays as a double parameter since it's a design-time knob, but the
// per-sample computation is performed entirely in T.
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
	using std::cos;
	mtl::vec::dense_vector<T> w(length);
	if (length <= 1) { if (length == 1) w[0] = T(1); return w; }
	if (alpha <= 0.0) {
		for (std::size_t n = 0; n < length; ++n) w[n] = T(1);
		return w;
	}

	constexpr T two_pi_T = T(two_pi);
	constexpr T half = T(0.5);
	constexpr T one  = T(1);

	if (alpha >= 1.0) {
		const T N = T(length - 1);
		for (std::size_t n = 0; n < length; ++n) {
			w[n] = half * (one - cos(two_pi_T * T(n) / N));
		}
		return w;
	}

	const T N = T(length - 1);
	const T alpha_T = T(alpha);
	const T alpha_N = alpha_T * N;
	const T half_taper = alpha_N * half;
	for (std::size_t n = 0; n < length; ++n) {
		const T nd = T(n);
		if (nd < half_taper) {
			w[n] = half * (one - cos(two_pi_T * nd / alpha_N));
		} else if (nd > N - half_taper) {
			w[n] = half * (one - cos(two_pi_T * (N - nd) / alpha_N));
		} else {
			w[n] = one;
		}
	}
	return w;
}

} // namespace sw::dsp
