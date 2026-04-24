#pragma once
// kaiser.hpp: Kaiser window
//
// w[n] = I0(beta * sqrt(1 - ((2n/(N-1)) - 1)^2)) / I0(beta)
//
// where I0 is the zeroth-order modified Bessel function of the first kind.
// beta controls the trade-off between main lobe width and side lobe level.
//
// Intermediate math runs in T. detail::bessel_I0 is templated so the
// Bessel series expansion runs at the caller's declared precision;
// sqrt and max dispatch via ADL.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

namespace detail {

// Modified Bessel function of the first kind, order 0.
// Series expansion: I0(x) = sum_{k=0}^{inf} ((x/2)^k / k!)^2
// Evaluated entirely in T so non-native CoeffScalar callers don't
// silently fall back to double precision.
template <DspField T>
T bessel_I0(const T& x) {
	constexpr T half = T(0.5);
	constexpr T tol  = T(1e-15);
	T sum  = T(1);
	T term = T(1);
	const T x_half = x * half;
	for (int k = 1; k < 30; ++k) {
		const T ratio = x_half / T(k);
		term = term * ratio * ratio;
		sum = sum + term;
		if (term < sum * tol) break;
	}
	return sum;
}

} // namespace detail

template <DspField T>
mtl::vec::dense_vector<T> kaiser_window(std::size_t length, double beta = 8.6) {
	using std::sqrt;
	mtl::vec::dense_vector<T> w(length);
	if (length <= 1) { if (length == 1) w[0] = T(1); return w; }

	constexpr T zero = T(0);
	constexpr T one  = T(1);
	constexpr T two  = T(2);
	const T N = T(length - 1);
	const T beta_T = T(beta);
	const T I0_beta = detail::bessel_I0(beta_T);
	for (std::size_t n = 0; n < length; ++n) {
		const T x = two * T(n) / N - one;
		const T radicand = one - x * x;
		const T clamped  = (radicand < zero) ? zero : radicand;
		const T arg = beta_T * sqrt(clamped);
		w[n] = detail::bessel_I0(arg) / I0_beta;
	}
	return w;
}

} // namespace sw::dsp
