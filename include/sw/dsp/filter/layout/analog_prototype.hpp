#pragma once
// analog_prototype.hpp: analog lowpass prototype for pole-based filter design
//
// Analog prototypes place poles and zeros in the s-plane to define
// the filter's frequency response shape. Each filter family (Butterworth,
// Chebyshev, etc.) has its own prototype that is then transformed to
// the desired band and digitized via the bilinear transform.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <complex>
#include <limits>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/filter/layout/layout.hpp>

namespace sw::dsp {

// Helper: the s-plane representation of a zero at infinity.
// Used for all-pole prototypes (Butterworth, Chebyshev, Bessel)
// where every zero is at s = infinity.
template <DspField T>
inline std::complex<T> s_infinity() {
	return std::complex<T>(std::numeric_limits<T>::infinity(), T{});
}

} // namespace sw::dsp
