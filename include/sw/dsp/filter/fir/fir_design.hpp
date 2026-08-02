#pragma once
// fir_design.hpp: FIR filter design via the window method
//
// Designs FIR lowpass, highpass, and bandpass filters by windowing
// an ideal (sinc) impulse response. The window controls the
// trade-off between transition bandwidth and stopband attenuation.
//
// Intermediate math runs in the template scalar T so that callers using
// posit, cfloat, fixpnt, etc. get design-time computation at their
// declared precision — a requirement for embedded mixed-precision
// deployments where filter design may execute on-target. ADL-friendly
// trig (using std::sin) picks up sw::universal::sin for Universal number
// types and std::sin for native floats.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>

namespace sw::dsp {

// Design a lowpass FIR filter using the window method.
//
// num_taps: filter length (odd recommended for linear phase)
// cutoff:   normalized cutoff frequency in (0, 0.5) where 0.5 = Nyquist
// window:   window vector of length num_taps (e.g., from hamming_window)
//
// Returns tap coefficients as dense_vector<T>.
template <DspField T>
mtl::vec::dense_vector<T> design_fir_lowpass(std::size_t num_taps, T cutoff,
                                              const mtl::vec::dense_vector<T>& window) {
	using std::abs; using std::sin;
	if (window.size() != num_taps)
		throw std::invalid_argument("design_fir_lowpass: window size must equal num_taps");
	mtl::vec::dense_vector<T> taps(num_taps);

	// Constants built from pure constructor calls (constexpr since
	// Universal v4.6.10 for posit's IEEE-754 ctors).
	constexpr T two    = T(2);
	constexpr T pi_T   = T(pi);
	constexpr T two_pi_T = T(two_pi);

	// Detect the center tap via integer INDEX equality instead of a
	// float-space threshold. Only odd num_taps produces x = 0 exactly (at
	// n = M/2, an integer); for even num_taps M/2 is a half-integer and the
	// sinc argument never lands on zero.
	//
	// Previous code used a threshold `T(1) / T(1e12)` for float-space
	// comparison. For narrow cfloat CoeffScalar types (cfloat<16,5>,
	// cfloat<24,5>) T(1e12) overflows the 5-bit exponent range (max ~65k),
	// saturating to +inf and collapsing tiny to 0. That made the guard
	// `abs(x) < tiny` never fire, so the sinc(0) = 0/0 case fell through
	// to `sin(0) / (pi*0) = NaN` at the center tap — see issue #201 for
	// the reproducer and root-cause analysis. The integer-index check
	// below is representation-independent by construction.
	const std::size_t M = num_taps - 1;
	const T half_M = T(M) / two;
	const bool is_odd_length = (num_taps % 2) == 1;
	const std::size_t center = M / 2;  // exact center only when is_odd_length

	for (std::size_t n = 0; n < num_taps; ++n) {
		T h;
		if (is_odd_length && n == center) {
			// L'Hopital limit of sin(2π·fc·x) / (π·x) at x=0 is 2·fc.
			h = two * cutoff;
		} else {
			T x = T(n) - half_M;
			h = sin(two_pi_T * cutoff * x) / (pi_T * x);
		}
		taps[n] = h * window[n];
	}
	return taps;
}

// Design a highpass FIR filter using spectral inversion.
// Designs a lowpass and subtracts from a delayed impulse.
template <DspField T>
mtl::vec::dense_vector<T> design_fir_highpass(std::size_t num_taps, T cutoff,
                                               const mtl::vec::dense_vector<T>& window) {
	auto lp = design_fir_lowpass<T>(num_taps, cutoff, window);

	constexpr T one = T(1);

	// Spectral inversion: negate all taps, add 1 to center tap
	for (std::size_t n = 0; n < num_taps; ++n) {
		lp[n] = -lp[n];
	}
	std::size_t center = (num_taps - 1) / 2;
	lp[center] = lp[center] + one;

	return lp;
}

// Design a bandpass FIR filter.
// Designs lowpass at f_high, subtracts lowpass at f_low.
template <DspField T>
mtl::vec::dense_vector<T> design_fir_bandpass(std::size_t num_taps,
                                               T f_low, T f_high,
                                               const mtl::vec::dense_vector<T>& window) {
	auto lp_high = design_fir_lowpass<T>(num_taps, f_high, window);
	auto lp_low  = design_fir_lowpass<T>(num_taps, f_low,  window);

	mtl::vec::dense_vector<T> bp(num_taps);
	for (std::size_t n = 0; n < num_taps; ++n) {
		bp[n] = lp_high[n] - lp_low[n];
	}
	return bp;
}

} // namespace sw::dsp
