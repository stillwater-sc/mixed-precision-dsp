#pragma once
// elliptic_integrals.hpp: Jacobi elliptic functions for Elliptic filter design
//
// Complete elliptic integral of the first kind K(k) using the
// arithmetic-geometric mean (AGM). Also provides the Jacobi sn function
// needed for zero placement in Elliptic (Cauer) filters.
//
// All arithmetic parameterized on T for mixed-precision support.
// Uses ADL-friendly math calls.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <limits>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>

namespace sw::dsp {

// Complete elliptic integral of the first kind K(k).
// Uses arithmetic-geometric mean (AGM) iteration.
// Fast convergence, peak error less than 2e-16.
template <DspField T>
T elliptic_K(T k) {
	using std::sqrt;  // ADL
	T m = k * k;
	T a = T{1};
	T b = sqrt(T{1} - m);
	T c = a - b;
	T co;
	do {
		co = c;
		c = (a - b) / T{2};
		T ao = (a + b) / T{2};
		b = sqrt(a * b);
		a = ao;
	} while (c < co);

	return pi_v<T> / (a + a);
}

// Jacobi sn function evaluated via q-series expansion.
//
// Given u (real argument), K = K(k) (complete elliptic integral of the
// first kind), and Kprime = K(k') where k' = sqrt(1-k^2), returns the
// q-series sum:
//
//   S(u, K, Kprime) = sum_{j=0}^{inf} q^(j+1/2) *
//                     sin((2j+1) * pi u / (2K)) / (1 - q^(2j+1))
//
// where q = exp(-pi * Kprime / K) is the nome.
//
// The actual Jacobi sn(u, k) equals S * (2*pi) / (k * K). Callers using
// the modified form typical in Elliptic-filter zero placement (where
// only the ratio 1/sn is used) can apply the scale factor themselves;
// see filter/iir/elliptic.hpp for that convention.
//
// Convergence: |q| < 1 (guaranteed when 0 < k < 1) and the series
// terms fall off geometrically. Terminates when the leading q^(j+1/2)
// factor drops below ~1000 * epsilon.
template <DspField T>
T elliptic_sn_series(T u, T K, T Kprime) {
	using std::exp;
	using std::pow;
	using std::sin;

	T q = exp(T{-1} * pi_v<T> * Kprime / K);
	T v = half_pi_v<T> * u / K;
	T tol = std::numeric_limits<T>::epsilon() * T{1000};
	T sn{};
	for (int j = 0; ; ++j) {
		T w = pow(q, static_cast<T>(j) + T{0.5});
		sn = sn + w * sin((T{2} * static_cast<T>(j) + T{1}) * v)
		          / (T{1} - w * w);
		if (w < tol) break;
	}
	return sn;
}

} // namespace sw::dsp
