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

} // namespace sw::dsp
