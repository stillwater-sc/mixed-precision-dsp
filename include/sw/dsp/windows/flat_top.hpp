#pragma once
// flat_top.hpp: flat-top window
//
// Optimized for amplitude accuracy in spectral analysis.
// w[n] = a0 - a1*cos(2*pi*n/(N-1)) + a2*cos(4*pi*n/(N-1))
//        - a3*cos(6*pi*n/(N-1)) + a4*cos(8*pi*n/(N-1))
//
// Coefficients from ISO 18431-2 (HFT90D):
// a0=1, a1=1.942604, a2=1.340318, a3=0.440811, a4=0.043097
//
// Intermediate math runs in T; ADL trig for Universal types.
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
mtl::vec::dense_vector<T> flat_top_window(std::size_t length) {
	using std::cos;
	mtl::vec::dense_vector<T> w(length);
	if (length <= 1) { if (length == 1) w[0] = T(1); return w; }

	constexpr T two_pi_T = T(two_pi);
	constexpr T a0 = T(1.0);
	constexpr T a1 = T(1.942604);
	constexpr T a2 = T(1.340318);
	constexpr T a3 = T(0.440811);
	constexpr T a4 = T(0.043097);
	constexpr T two   = T(2);
	constexpr T three = T(3);
	constexpr T four  = T(4);
	const T N = T(length - 1);
	for (std::size_t n = 0; n < length; ++n) {
		const T x = two_pi_T * T(n) / N;
		w[n] = a0 - a1 * cos(x)
		          + a2 * cos(two   * x)
		          - a3 * cos(three * x)
		          + a4 * cos(four  * x);
	}
	return w;
}

} // namespace sw::dsp
