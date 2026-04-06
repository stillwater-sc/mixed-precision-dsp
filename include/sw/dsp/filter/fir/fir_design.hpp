#pragma once
// fir_design.hpp: FIR filter design via the window method
//
// Designs FIR lowpass, highpass, and bandpass filters by windowing
// an ideal (sinc) impulse response. The window controls the
// trade-off between transition bandwidth and stopband attenuation.
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
	if (window.size() != num_taps)
		throw std::invalid_argument("design_fir_lowpass: window size must equal num_taps");
	mtl::vec::dense_vector<T> taps(num_taps);
	int M = static_cast<int>(num_taps - 1);
	double fc = static_cast<double>(cutoff);

	for (std::size_t n = 0; n < num_taps; ++n) {
		double x = static_cast<double>(n) - M * 0.5;
		double h;
		if (std::abs(x) < 1e-10) {
			h = 2.0 * fc;  // sinc(0) = 1, scaled by 2*fc
		} else {
			h = std::sin(2.0 * pi * fc * x) / (pi * x);
		}
		taps[n] = static_cast<T>(h) * window[n];
	}
	return taps;
}

// Design a highpass FIR filter using spectral inversion.
// Designs a lowpass and subtracts from a delayed impulse.
template <DspField T>
mtl::vec::dense_vector<T> design_fir_highpass(std::size_t num_taps, T cutoff,
                                               const mtl::vec::dense_vector<T>& window) {
	auto lp = design_fir_lowpass(num_taps, cutoff, window);

	// Spectral inversion: negate all taps, add 1 to center tap
	for (std::size_t n = 0; n < num_taps; ++n) {
		lp[n] = T{} - lp[n];
	}
	std::size_t center = (num_taps - 1) / 2;
	lp[center] = lp[center] + T{1};

	return lp;
}

// Design a bandpass FIR filter.
// Designs lowpass at f_high, subtracts lowpass at f_low.
template <DspField T>
mtl::vec::dense_vector<T> design_fir_bandpass(std::size_t num_taps,
                                               T f_low, T f_high,
                                               const mtl::vec::dense_vector<T>& window) {
	auto lp_high = design_fir_lowpass(num_taps, f_high, window);
	auto lp_low = design_fir_lowpass(num_taps, f_low, window);

	mtl::vec::dense_vector<T> bp(num_taps);
	for (std::size_t n = 0; n < num_taps; ++n) {
		bp[n] = lp_high[n] - lp_low[n];
	}
	return bp;
}

} // namespace sw::dsp
