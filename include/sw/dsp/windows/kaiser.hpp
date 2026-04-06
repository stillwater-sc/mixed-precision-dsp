#pragma once
// kaiser.hpp: Kaiser window
//
// w[n] = I0(beta * sqrt(1 - ((2n/(N-1)) - 1)^2)) / I0(beta)
//
// where I0 is the zeroth-order modified Bessel function of the first kind.
// beta controls the trade-off between main lobe width and side lobe level.
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
inline double bessel_I0(double x) {
	double sum = 1.0;
	double term = 1.0;
	double x_half = x * 0.5;
	for (int k = 1; k < 30; ++k) {
		term *= (x_half / k) * (x_half / k);
		sum += term;
		if (term < sum * 1e-15) break;
	}
	return sum;
}

} // namespace detail

template <DspField T>
mtl::vec::dense_vector<T> kaiser_window(std::size_t length, double beta = 8.6) {
	mtl::vec::dense_vector<T> w(length);
	if (length <= 1) { if (length == 1) w[0] = T{1}; return w; }
	double N = static_cast<double>(length - 1);
	double I0_beta = detail::bessel_I0(beta);
	for (std::size_t n = 0; n < length; ++n) {
		double x = 2.0 * static_cast<double>(n) / N - 1.0;
		double arg = beta * std::sqrt(std::max(0.0, 1.0 - x * x));
		w[n] = static_cast<T>(detail::bessel_I0(arg) / I0_beta);
	}
	return w;
}

} // namespace sw::dsp
