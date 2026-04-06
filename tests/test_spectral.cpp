// test_spectral.cpp: test Z-transform, Laplace, PSD, and spectrogram
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/spectral/spectral.hpp>
#include <sw/dsp/types/transfer_function.hpp>
#include <sw/dsp/signals/generators.hpp>
#include <sw/dsp/windows/windows.hpp>
#include <sw/dsp/math/constants.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace sw::dsp;
using namespace sw::dsp::spectral;

bool near(double a, double b, double eps = 1e-4) {
	return std::abs(a - b) < eps;
}

// ========== Z-transform Tests ==========

void test_freqz() {
	// Unity transfer function: H(z) = 1
	TransferFunction<double> unity;
	unity.numerator = mtl::vec::dense_vector<double>({1.0});

	auto H = freqz(unity, 64);
	if (!(H.size() == 64)) throw std::runtime_error("test failed: freqz size");

	// All magnitudes should be 1
	for (std::size_t k = 0; k < H.size(); ++k) {
		if (!(near(std::abs(H[k]), 1.0, 1e-10)))
			throw std::runtime_error("test failed: freqz unity magnitude");
	}

	std::cout << "  freqz: passed\n";
}

void test_group_delay() {
	// Pure delay: H(z) = z^-1 -> constant group delay of 1 sample
	TransferFunction<double> delay;
	delay.numerator = mtl::vec::dense_vector<double>({0.0, 1.0});

	auto gd = group_delay(delay, 64);
	if (!(gd.size() == 64)) throw std::runtime_error("test failed: group delay size");

	// Group delay should be approximately 1 sample everywhere
	for (std::size_t k = 1; k < 60; ++k) {  // skip near DC and Nyquist
		if (!(near(gd[k], 1.0, 0.1)))
			throw std::runtime_error("test failed: group delay of unit delay at bin " + std::to_string(k));
	}

	std::cout << "  group_delay: passed\n";
}

// ========== Laplace Tests ==========

void test_laplace_integrator() {
	// H(s) = 1/s (integrator): magnitude decreases with frequency
	ContinuousTransferFunction<double> integrator;
	integrator.numerator = mtl::vec::dense_vector<double>({1.0});
	integrator.denominator = mtl::vec::dense_vector<double>({0.0, 1.0});  // s

	auto H = freqs(integrator, 100.0, 64);
	if (!(H.size() == 64)) throw std::runtime_error("test failed: freqs size");

	// Magnitude at w=1 should be 1.0 (|1/j| = 1)
	auto r = integrator.frequency_response(1.0);
	if (!(near(std::abs(r), 1.0, 0.01)))
		throw std::runtime_error("test failed: integrator |H(j)| = 1");

	// Magnitude at w=10 should be 0.1
	auto r10 = integrator.frequency_response(10.0);
	if (!(near(std::abs(r10), 0.1, 0.01)))
		throw std::runtime_error("test failed: integrator |H(j10)| = 0.1");

	std::cout << "  laplace_integrator: passed\n";
}

// ========== PSD Tests ==========

void test_periodogram_sine() {
	// Periodogram of a pure sine should peak at the signal frequency
	constexpr std::size_t N = 256;
	auto sig = sine<double>(N, 32.0, static_cast<double>(N));
	mtl::vec::dense_vector<double> x(N);
	for (std::size_t i = 0; i < N; ++i) x[i] = sig[i];

	auto psd = periodogram(x);
	if (!(psd.size() == N / 2 + 1))
		throw std::runtime_error("test failed: periodogram size");

	// Find peak
	double max_p = 0;
	std::size_t peak_bin = 0;
	for (std::size_t k = 1; k < psd.size(); ++k) {
		if (psd[k] > max_p) { max_p = psd[k]; peak_bin = k; }
	}
	// 32 Hz at 256 Hz sample rate -> bin 32
	if (!(peak_bin == 32))
		throw std::runtime_error("test failed: periodogram peak at " + std::to_string(peak_bin));

	std::cout << "  periodogram_sine: passed (peak bin=" << peak_bin << ")\n";
}

void test_welch() {
	// Welch PSD should also peak at signal frequency with less variance
	constexpr std::size_t N = 1024;
	auto sig = sine<double>(N, 50.0, static_cast<double>(N));
	mtl::vec::dense_vector<double> x(N);
	for (std::size_t i = 0; i < N; ++i) x[i] = sig[i];

	auto win = hamming_window<double>(256);
	auto psd = welch(x, 256, 128, win);

	if (!(psd.size() == 129))  // 256/2 + 1
		throw std::runtime_error("test failed: welch size");

	// Find peak
	double max_p = 0;
	std::size_t peak_bin = 0;
	for (std::size_t k = 1; k < psd.size(); ++k) {
		if (psd[k] > max_p) { max_p = psd[k]; peak_bin = k; }
	}
	// 50 Hz at 1024 Hz sample rate, 256-point segments
	// Bin = 50 * 256 / 1024 = 12.5 -> expect near bin 12 or 13
	if (!(peak_bin >= 11 && peak_bin <= 14))
		throw std::runtime_error("test failed: welch peak at " + std::to_string(peak_bin));

	std::cout << "  welch: passed (peak bin=" << peak_bin << ")\n";
}

void test_psd_db() {
	mtl::vec::dense_vector<double> x({1.0, 0.0, 0.0, 0.0});
	auto db = psd_db(x);
	if (!(db.size() == 3))  // 4/2+1
		throw std::runtime_error("test failed: psd_db size");
	// All values should be finite
	for (std::size_t k = 0; k < db.size(); ++k) {
		if (!(std::isfinite(db[k])))
			throw std::runtime_error("test failed: psd_db not finite");
	}

	std::cout << "  psd_db: passed\n";
}

// ========== Spectrogram Tests ==========

void test_spectrogram_basic() {
	constexpr std::size_t N = 512;
	auto sig = sine<double>(N, 50.0, static_cast<double>(N));
	mtl::vec::dense_vector<double> x(N);
	for (std::size_t i = 0; i < N; ++i) x[i] = sig[i];

	auto win = hamming_window<double>(128);
	auto stft = spectrogram(x, win, 64);

	if (!(stft.fft_size == 128))
		throw std::runtime_error("test failed: spectrogram fft_size");
	if (!(stft.hop_size == 64))
		throw std::runtime_error("test failed: spectrogram hop_size");
	if (!(stft.num_frames() > 0))
		throw std::runtime_error("test failed: spectrogram empty");

	// Expected frames: (512 - 128) / 64 + 1 = 7
	if (!(stft.num_frames() == 7))
		throw std::runtime_error("test failed: spectrogram frame count " + std::to_string(stft.num_frames()));

	std::cout << "  spectrogram_basic: passed (" << stft.num_frames() << " frames)\n";
}

void test_spectrogram_magnitude_db() {
	constexpr std::size_t N = 256;
	auto sig = sine<double>(N, 32.0, static_cast<double>(N));
	mtl::vec::dense_vector<double> x(N);
	for (std::size_t i = 0; i < N; ++i) x[i] = sig[i];

	auto win = hamming_window<double>(64);
	auto stft = spectrogram(x, win, 32);
	auto mag = spectrogram_magnitude_db(stft);

	if (!(mag.size() == stft.num_frames()))
		throw std::runtime_error("test failed: spectrogram mag size");
	if (!(mag[0].size() == stft.num_bins()))
		throw std::runtime_error("test failed: spectrogram mag bins");

	// All values should be finite
	for (const auto& frame : mag) {
		for (std::size_t k = 0; k < frame.size(); ++k) {
			if (!(std::isfinite(frame[k])))
				throw std::runtime_error("test failed: spectrogram mag not finite");
		}
	}

	std::cout << "  spectrogram_magnitude_db: passed\n";
}

int main() {
	try {
		std::cout << "Spectral Analysis Tests\n";

		test_freqz();
		test_group_delay();
		test_laplace_integrator();
		test_periodogram_sine();
		test_welch();
		test_psd_db();
		test_spectrogram_basic();
		test_spectrogram_magnitude_db();

		std::cout << "All spectral analysis tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << '\n';
		return 1;
	}
}
