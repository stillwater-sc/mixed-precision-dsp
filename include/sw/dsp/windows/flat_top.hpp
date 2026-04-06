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
	mtl::vec::dense_vector<T> w(length);
	if (length <= 1) { if (length == 1) w[0] = T{1}; return w; }
	double N = static_cast<double>(length - 1);
	constexpr double a0 = 1.0, a1 = 1.942604, a2 = 1.340318, a3 = 0.440811, a4 = 0.043097;
	for (std::size_t n = 0; n < length; ++n) {
		double x = two_pi * static_cast<double>(n) / N;
		w[n] = static_cast<T>(a0 - a1*std::cos(x) + a2*std::cos(2*x) - a3*std::cos(3*x) + a4*std::cos(4*x));
	}
	return w;
}

} // namespace sw::dsp
