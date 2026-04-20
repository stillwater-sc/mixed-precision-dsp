#pragma once
// dolph_chebyshev.hpp: Dolph-Chebyshev window
//
// Equiripple sidelobes at a specified attenuation level. Optimal in the
// sense of narrowest main lobe for a given sidelobe level.
//
// Uses the DFT-based construction:
//   W[k] = T_{N-1}(x0 * cos(pi*k/N)) / T_{N-1}(x0)
// where T_n is the Chebyshev polynomial, x0 = cosh(acosh(10^(atten/20))/(N-1))
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>

namespace sw::dsp {

namespace detail {

inline double chebyshev_poly(int n, double x) {
	if (std::abs(x) <= 1.0)
		return std::cos(static_cast<double>(n) * std::acos(x));
	return std::cosh(static_cast<double>(n) * std::acosh(std::abs(x)));
}

} // namespace detail

template <DspField T>
mtl::vec::dense_vector<T> dolph_chebyshev_window(std::size_t length,
                                                  double attenuation_db = 100.0) {
	mtl::vec::dense_vector<T> w(length);
	if (length <= 1) { if (length == 1) w[0] = T{1}; return w; }

	int N = static_cast<int>(length);
	int order = N - 1;
	double atten_linear = std::pow(10.0, attenuation_db / 20.0);
	double x0 = std::cosh(std::acosh(atten_linear) / static_cast<double>(order));

	// Compute frequency-domain window W[k] via Chebyshev polynomial
	// (-1)^k factor ensures the IDFT produces a real, symmetric result
	std::vector<double> W(N);
	for (int k = 0; k < N; ++k) {
		double sign = (k % 2 == 0) ? 1.0 : -1.0;
		double arg = x0 * std::cos(pi * static_cast<double>(k) / static_cast<double>(N));
		W[k] = sign * detail::chebyshev_poly(order, arg) / atten_linear;
	}

	// Inverse DFT to get time-domain window (real-valued, symmetric)
	std::vector<double> wn(N);
	double max_val = 0.0;
	for (int n = 0; n < N; ++n) {
		double sum = 0.0;
		for (int k = 0; k < N; ++k) {
			sum += W[k] * std::cos(two_pi * static_cast<double>(k) *
			       static_cast<double>(n) / static_cast<double>(N));
		}
		wn[n] = sum;
		if (std::abs(sum) > max_val) max_val = std::abs(sum);
	}

	// Enforce symmetry then normalize to unity peak
	for (int n = 0; n < N / 2; ++n) {
		double avg = 0.5 * (wn[n] + wn[N - 1 - n]);
		wn[n] = avg;
		wn[N - 1 - n] = avg;
	}
	max_val = 0.0;
	for (int n = 0; n < N; ++n) {
		if (std::abs(wn[n]) > max_val) max_val = std::abs(wn[n]);
	}
	for (int n = 0; n < N; ++n) {
		w[n] = static_cast<T>(wn[n] / max_val);
	}
	return w;
}

} // namespace sw::dsp
