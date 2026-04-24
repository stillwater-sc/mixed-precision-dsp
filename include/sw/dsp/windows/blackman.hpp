#pragma once
// blackman.hpp: Blackman window
//
// w[n] = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))
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
mtl::vec::dense_vector<T> blackman_window(std::size_t length) {
	using std::cos;
	mtl::vec::dense_vector<T> w(length);
	if (length <= 1) { if (length == 1) w[0] = T(1); return w; }

	constexpr T two_pi_T = T(two_pi);
	constexpr T a0 = T(0.42);
	constexpr T a1 = T(0.5);
	constexpr T a2 = T(0.08);
	constexpr T two = T(2);
	const T N = T(length - 1);
	for (std::size_t n = 0; n < length; ++n) {
		const T x = two_pi_T * T(n) / N;
		w[n] = a0 - a1 * cos(x) + a2 * cos(two * x);
	}
	return w;
}

} // namespace sw::dsp
