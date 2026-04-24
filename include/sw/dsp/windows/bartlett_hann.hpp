#pragma once
// bartlett_hann.hpp: Bartlett-Hann window
//
// w[n] = 0.62 - 0.48 * |n/(N-1) - 0.5| + 0.38 * cos(2*pi * (n/(N-1) - 0.5))
//
// Intermediate math runs in T; ADL trig and abs for Universal types.
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
	using std::cos; using std::abs;
	mtl::vec::dense_vector<T> w(length);
	if (length <= 1) { if (length == 1) w[0] = T(1); return w; }

	constexpr T two_pi_T = T(two_pi);
	constexpr T half = T(0.5);
	constexpr T a0   = T(0.62);
	constexpr T a1   = T(0.48);
	constexpr T a2   = T(0.38);
	const T N = T(length - 1);
	for (std::size_t n = 0; n < length; ++n) {
		const T x = T(n) / N - half;
		w[n] = a0 - a1 * abs(x) + a2 * cos(two_pi_T * x);
	}
	return w;
}

} // namespace sw::dsp
