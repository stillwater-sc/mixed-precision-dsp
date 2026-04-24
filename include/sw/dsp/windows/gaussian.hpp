#pragma once
// gaussian.hpp: Gaussian window
//
// w[n] = exp(-0.5 * ((n - (N-1)/2) / (sigma * (N-1)/2))^2)
//
// Intermediate math runs in T; ADL std::exp picks up sw::universal::exp
// for Universal types.
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
	using std::exp;
	mtl::vec::dense_vector<T> w(length);
	if (length <= 1) { if (length == 1) w[0] = T(1); return w; }
	if (sigma <= 0.0) sigma = 0.4;

	constexpr T neg_half = T(-0.5);
	constexpr T half = T(0.5);
	const T N_minus_1 = T(length - 1);
	const T half_N = N_minus_1 * half;
	const T denom = T(sigma) * half_N;
	for (std::size_t n = 0; n < length; ++n) {
		const T x = (T(n) - half_N) / denom;
		w[n] = exp(neg_half * x * x);
	}
	return w;
}

} // namespace sw::dsp
