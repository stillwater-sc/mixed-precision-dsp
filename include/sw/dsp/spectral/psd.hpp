#pragma once
// psd.hpp: power spectral density estimation
//
// Periodogram and Welch's method for PSD estimation.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/spectral/fft.hpp>
#include <sw/dsp/windows/windows.hpp>

namespace sw::dsp::spectral {

// Periodogram: |FFT(x)|^2 / N, returns one-sided PSD (N/2+1 bins).
template <DspField T>
mtl::vec::dense_vector<double> periodogram(const mtl::vec::dense_vector<T>& x) {
	auto X = fft(x);
	std::size_t N = X.size();
	std::size_t half = N / 2 + 1;
	double inv_N = 1.0 / static_cast<double>(N);

	mtl::vec::dense_vector<double> psd(half);
	for (std::size_t k = 0; k < half; ++k) {
		double mag = static_cast<double>(std::abs(X[k]));
		psd[k] = mag * mag * inv_N;
		// Double non-DC, non-Nyquist bins for one-sided
		if (k > 0 && k < N / 2) psd[k] *= 2.0;
	}
	return psd;
}

// Welch's method: averaged periodograms with overlapping windowed segments.
//
// segment_size: length of each segment (should be power of 2 for FFT)
// overlap:      number of overlapping samples between segments
// window:       window to apply to each segment
//
// Returns one-sided PSD (segment_size/2 + 1 bins).
template <DspField T>
mtl::vec::dense_vector<double> welch(
		const mtl::vec::dense_vector<T>& x,
		std::size_t segment_size,
		std::size_t overlap,
		const mtl::vec::dense_vector<T>& window) {
	using std::abs;

	std::size_t half = segment_size / 2 + 1;
	mtl::vec::dense_vector<double> psd(half, 0.0);
	double inv_seg = 1.0 / static_cast<double>(segment_size);

	// Compute window power for normalization
	double win_power = 0.0;
	for (std::size_t i = 0; i < window.size(); ++i) {
		double w = static_cast<double>(window[i]);
		win_power += w * w;
	}
	win_power /= static_cast<double>(window.size());

	std::size_t step = segment_size - overlap;
	int num_segments = 0;

	for (std::size_t start = 0; start + segment_size <= x.size(); start += step) {
		// Extract and window segment
		mtl::vec::dense_vector<T> segment(segment_size);
		for (std::size_t i = 0; i < segment_size; ++i) {
			segment[i] = x[start + i] * window[i];
		}

		auto X = fft(segment);

		for (std::size_t k = 0; k < half; ++k) {
			double mag = static_cast<double>(abs(X[k]));
			double p = mag * mag * inv_seg;
			if (k > 0 && k < segment_size / 2) p *= 2.0;
			psd[k] = psd[k] + p;
		}
		++num_segments;
	}

	// Average and normalize by window power
	if (num_segments > 0 && win_power > 0) {
		double scale = 1.0 / (static_cast<double>(num_segments) * win_power);
		for (std::size_t k = 0; k < half; ++k) {
			psd[k] = psd[k] * scale;
		}
	}

	return psd;
}

// Convenience: PSD in dB
template <DspField T>
mtl::vec::dense_vector<double> psd_db(const mtl::vec::dense_vector<T>& x,
                                       double min_db = -120.0) {
	auto psd = periodogram(x);
	for (std::size_t k = 0; k < psd.size(); ++k) {
		psd[k] = (psd[k] > 0) ? 10.0 * std::log10(psd[k]) : min_db;
	}
	return psd;
}

} // namespace sw::dsp::spectral
