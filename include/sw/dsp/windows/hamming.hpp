#pragma once
// hamming.hpp: Hamming window
//
// w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
//
// Intermediate math runs in T so posit/cfloat/etc. callers get the
// window computed at their declared precision (required for embedded
// mixed-precision deployments). ADL trig (using std::cos) selects
// sw::universal::cos for Universal types and std::cos for native.
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
mtl::vec::dense_vector<T> hamming_window(std::size_t length) {
	using std::cos;
	mtl::vec::dense_vector<T> w(length);
	if (length <= 1) { if (length == 1) w[0] = T(1); return w; }

	// Constants are built from double literals via the constexpr ctor
	// (operator/ is not yet constexpr for posit).
	constexpr T two_pi_T = T(two_pi);
	constexpr T a0 = T(0.54);
	constexpr T a1 = T(0.46);
	const T N = T(length - 1);
	for (std::size_t n = 0; n < length; ++n) {
		w[n] = a0 - a1 * cos(two_pi_T * T(n) / N);
	}
	return w;
}

} // namespace sw::dsp
