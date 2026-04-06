#pragma once
// spectrogram.hpp: short-time Fourier transform (STFT) spectrogram
//
// Computes a time-frequency representation by applying FFT to
// overlapping windowed segments of the input signal.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/spectral/fft.hpp>

namespace sw::dsp::spectral {

// STFT result: vector of frequency-domain frames.
// Each frame is a complex spectrum of size fft_size.
template <DspField T>
struct SpectrogramResult {
	std::vector<mtl::vec::dense_vector<complex_for_t<T>>> frames;
	std::size_t fft_size;
	std::size_t hop_size;
	std::size_t window_size;

	// Number of time frames
	std::size_t num_frames() const { return frames.size(); }
	// Number of frequency bins (one-sided)
	std::size_t num_bins() const { return fft_size / 2 + 1; }
};

// Compute STFT spectrogram.
//
// x:           input signal
// window:      analysis window (determines window_size)
// hop_size:    step between successive frames
// fft_size:    FFT size (>= window_size, zero-padded if larger)
//
// Returns SpectrogramResult with complex spectra.
template <DspField T>
SpectrogramResult<T> spectrogram(
		const mtl::vec::dense_vector<T>& x,
		const mtl::vec::dense_vector<T>& window,
		std::size_t hop_size,
		std::size_t fft_size = 0) {
	using complex_t = complex_for_t<T>;

	std::size_t win_size = window.size();
	if (win_size == 0)
		throw std::invalid_argument("spectrogram: window must be non-empty");
	if (hop_size == 0)
		throw std::invalid_argument("spectrogram: hop_size must be > 0");
	if (fft_size == 0) {
		// Default: next power of 2 >= window_size
		fft_size = 1;
		while (fft_size < win_size) fft_size <<= 1;
	}
	if (fft_size < win_size)
		throw std::invalid_argument("spectrogram: fft_size must be >= window.size()");

	SpectrogramResult<T> result;
	result.fft_size = fft_size;
	result.hop_size = hop_size;
	result.window_size = win_size;

	for (std::size_t start = 0; start + win_size <= x.size(); start += hop_size) {
		// Extract windowed segment, zero-pad to fft_size
		mtl::vec::dense_vector<complex_t> frame(fft_size, complex_t{});
		for (std::size_t i = 0; i < win_size; ++i) {
			frame[i] = complex_t(x[start + i] * window[i]);
		}

		fft_forward<T>(frame);
		result.frames.push_back(std::move(frame));
	}

	return result;
}

// Extract magnitude spectrogram in dB from STFT result.
// Returns vector of vectors: [frame_index][frequency_bin] in dB.
template <DspField T>
std::vector<mtl::vec::dense_vector<double>> spectrogram_magnitude_db(
		const SpectrogramResult<T>& stft, double min_db = -120.0) {
	using std::abs;
	std::size_t half = stft.num_bins();
	std::vector<mtl::vec::dense_vector<double>> mag_db;

	for (const auto& frame : stft.frames) {
		mtl::vec::dense_vector<double> db(half);
		for (std::size_t k = 0; k < half; ++k) {
			double m = static_cast<double>(abs(frame[k]));
			db[k] = (m > 0) ? 20.0 * std::log10(m) : min_db;
		}
		mag_db.push_back(std::move(db));
	}

	return mag_db;
}

} // namespace sw::dsp::spectral
