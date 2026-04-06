// test_fft.cpp: test DFT, FFT, and TransferFunction
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/spectral/spectral.hpp>
#include <sw/dsp/types/transfer_function.hpp>
#include <sw/dsp/signals/generators.hpp>
#include <sw/dsp/math/constants.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace sw::dsp;
using namespace sw::dsp::spectral;

bool near(double a, double b, double eps = 1e-6) {
	return std::abs(a - b) < eps;
}

// ========== DFT Tests ==========

void test_dft_impulse() {
	// DFT of impulse [1, 0, 0, 0] = [1, 1, 1, 1] (flat spectrum)
	mtl::vec::dense_vector<double> x({1.0, 0.0, 0.0, 0.0});
	auto X = dft(x);

	if (!(X.size() == 4)) throw std::runtime_error("test failed: DFT impulse size");
	for (std::size_t k = 0; k < 4; ++k) {
		if (!(near(X[k].real(), 1.0, 1e-10)))
			throw std::runtime_error("test failed: DFT impulse magnitude");
		if (!(near(X[k].imag(), 0.0, 1e-10)))
			throw std::runtime_error("test failed: DFT impulse phase");
	}

	std::cout << "  dft_impulse: passed\n";
}

void test_dft_dc() {
	// DFT of [1, 1, 1, 1] = [4, 0, 0, 0]
	mtl::vec::dense_vector<double> x({1.0, 1.0, 1.0, 1.0});
	auto X = dft(x);

	if (!(near(X[0].real(), 4.0, 1e-10)))
		throw std::runtime_error("test failed: DFT DC bin");
	for (std::size_t k = 1; k < 4; ++k) {
		if (!(near(std::abs(X[k]), 0.0, 1e-10)))
			throw std::runtime_error("test failed: DFT DC non-zero bin");
	}

	std::cout << "  dft_dc: passed\n";
}

void test_dft_idft_roundtrip() {
	// DFT then IDFT should recover original signal
	mtl::vec::dense_vector<double> x({0.5, -0.3, 0.8, -0.1, 0.6, 0.2, -0.7, 0.4});
	auto X = dft(x);
	auto y = idft<double>(X);

	if (!(y.size() == x.size()))
		throw std::runtime_error("test failed: IDFT size");
	for (std::size_t i = 0; i < x.size(); ++i) {
		if (!(near(y[i], x[i], 1e-10)))
			throw std::runtime_error("test failed: IDFT roundtrip at " + std::to_string(i));
	}

	std::cout << "  dft_idft_roundtrip: passed\n";
}

void test_dft_parseval() {
	// Parseval's theorem: sum|x[n]|^2 = (1/N) sum|X[k]|^2
	mtl::vec::dense_vector<double> x({0.5, -0.3, 0.8, -0.1});
	auto X = dft(x);

	double time_energy = 0, freq_energy = 0;
	for (std::size_t i = 0; i < x.size(); ++i) {
		time_energy += x[i] * x[i];
	}
	for (std::size_t k = 0; k < X.size(); ++k) {
		double m = std::abs(X[k]);
		freq_energy += m * m;
	}
	freq_energy /= static_cast<double>(x.size());

	if (!(near(time_energy, freq_energy, 1e-10)))
		throw std::runtime_error("test failed: Parseval's theorem");

	std::cout << "  dft_parseval: passed\n";
}

// ========== FFT Tests ==========

void test_fft_matches_dft() {
	// FFT should produce same result as DFT for power-of-2 sizes
	mtl::vec::dense_vector<double> x({0.5, -0.3, 0.8, -0.1, 0.6, 0.2, -0.7, 0.4});
	auto X_dft = dft(x);
	auto X_fft = fft(x);

	if (!(X_fft.size() == X_dft.size()))
		throw std::runtime_error("test failed: FFT vs DFT size");
	for (std::size_t k = 0; k < X_dft.size(); ++k) {
		if (!(near(X_fft[k].real(), X_dft[k].real(), 1e-10)))
			throw std::runtime_error("test failed: FFT vs DFT real at " + std::to_string(k));
		if (!(near(X_fft[k].imag(), X_dft[k].imag(), 1e-10)))
			throw std::runtime_error("test failed: FFT vs DFT imag at " + std::to_string(k));
	}

	std::cout << "  fft_matches_dft: passed\n";
}

void test_fft_inverse_roundtrip() {
	mtl::vec::dense_vector<double> x({0.5, -0.3, 0.8, -0.1, 0.6, 0.2, -0.7, 0.4});
	auto X = fft(x);
	auto y = ifft_real<double>(X);

	for (std::size_t i = 0; i < x.size(); ++i) {
		if (!(near(y[i], x[i], 1e-10)))
			throw std::runtime_error("test failed: FFT inverse roundtrip at " + std::to_string(i));
	}

	std::cout << "  fft_inverse_roundtrip: passed\n";
}

void test_fft_parseval() {
	auto sig = sine<double>(256, 10.0, 256.0);
	mtl::vec::dense_vector<double> x(sig.size());
	for (std::size_t i = 0; i < sig.size(); ++i) x[i] = sig[i];

	auto X = fft(x);

	double time_energy = 0, freq_energy = 0;
	for (std::size_t i = 0; i < x.size(); ++i) time_energy += x[i] * x[i];
	for (std::size_t k = 0; k < X.size(); ++k) {
		double m = std::abs(X[k]);
		freq_energy += m * m;
	}
	freq_energy /= static_cast<double>(X.size());

	if (!(near(time_energy, freq_energy, 1e-6)))
		throw std::runtime_error("test failed: FFT Parseval");

	std::cout << "  fft_parseval: passed\n";
}

void test_fft_sine_peak() {
	// FFT of a pure sine should have peaks at the signal frequency bin
	constexpr std::size_t N = 256;
	auto sig = sine<double>(N, 10.0, static_cast<double>(N));
	mtl::vec::dense_vector<double> x(N);
	for (std::size_t i = 0; i < N; ++i) x[i] = sig[i];

	auto X = fft(x);

	// Find peak bin (excluding DC)
	double max_mag = 0;
	std::size_t peak_bin = 0;
	for (std::size_t k = 1; k < N / 2; ++k) {
		double m = std::abs(X[k]);
		if (m > max_mag) { max_mag = m; peak_bin = k; }
	}

	// 10 Hz signal at 256 Hz sample rate -> bin 10
	if (!(peak_bin == 10))
		throw std::runtime_error("test failed: FFT sine peak at bin " + std::to_string(peak_bin));

	std::cout << "  fft_sine_peak: passed (bin=" << peak_bin << ")\n";
}

void test_fft_magnitude_db() {
	mtl::vec::dense_vector<double> x({1.0, 0.0, 0.0, 0.0});
	auto X = fft(x);
	auto mag = magnitude_spectrum_db<double>(X);

	// Impulse: flat magnitude at 0 dB
	for (std::size_t k = 0; k < mag.size(); ++k) {
		if (!(near(mag[k], 0.0, 0.01)))
			throw std::runtime_error("test failed: magnitude dB of impulse");
	}

	std::cout << "  fft_magnitude_db: passed\n";
}

void test_fft_zero_padding() {
	// fft() should zero-pad to next power of 2
	mtl::vec::dense_vector<double> x({1.0, 0.5, 0.25});  // length 3 -> padded to 4
	auto X = fft(x);
	if (!(X.size() == 4))
		throw std::runtime_error("test failed: FFT zero-padding size");

	std::cout << "  fft_zero_padding: passed\n";
}

// ========== TransferFunction Tests ==========

void test_tf_evaluate() {
	// H(z) = 1 (numerator = [1], denominator = [])
	TransferFunction<double> unity;
	unity.numerator = mtl::vec::dense_vector<double>({1.0});

	auto r = unity.frequency_response(0.25);
	if (!(near(std::abs(r), 1.0, 1e-10)))
		throw std::runtime_error("test failed: unity TF response");

	std::cout << "  tf_evaluate: passed\n";
}

void test_tf_first_order_lp() {
	// H(z) = 0.5 * (1 + z^-1) / 1 = simple average
	// numerator = [0.5, 0.5], denominator = []
	TransferFunction<double> avg;
	avg.numerator = mtl::vec::dense_vector<double>({0.5, 0.5});

	// At DC (z=1): H(1) = 0.5 + 0.5 = 1.0
	auto r_dc = avg.frequency_response(0.0);
	if (!(near(std::abs(r_dc), 1.0, 1e-10)))
		throw std::runtime_error("test failed: average TF DC");

	// At Nyquist (z=-1): H(-1) = 0.5 - 0.5 = 0.0
	auto r_nyq = avg.frequency_response(0.5);
	if (!(near(std::abs(r_nyq), 0.0, 1e-10)))
		throw std::runtime_error("test failed: average TF Nyquist");

	std::cout << "  tf_first_order_lp: passed\n";
}

void test_tf_stability() {
	// Stable: denominator with root inside unit circle
	TransferFunction<double> stable;
	stable.numerator = mtl::vec::dense_vector<double>({1.0});
	stable.denominator = mtl::vec::dense_vector<double>({-0.5});  // pole at z=0.5
	if (!(stable.is_stable()))
		throw std::runtime_error("test failed: stable TF should be stable");

	std::cout << "  tf_stability: passed\n";
}

void test_tf_cascade() {
	// H1(z) = 1 + z^-1, H2(z) = 1 - z^-1
	// H1*H2 = 1 - z^-2 (numerator convolution)
	TransferFunction<double> h1, h2;
	h1.numerator = mtl::vec::dense_vector<double>({1.0, 1.0});
	h2.numerator = mtl::vec::dense_vector<double>({1.0, -1.0});

	auto h = h1 * h2;
	if (!(h.numerator.size() == 3))
		throw std::runtime_error("test failed: cascade num size");
	if (!(near(h.numerator[0], 1.0, 1e-10)))
		throw std::runtime_error("test failed: cascade b0");
	if (!(near(h.numerator[1], 0.0, 1e-10)))
		throw std::runtime_error("test failed: cascade b1");
	if (!(near(h.numerator[2], -1.0, 1e-10)))
		throw std::runtime_error("test failed: cascade b2");

	std::cout << "  tf_cascade: passed\n";
}

int main() {
	try {
		std::cout << "Spectral & TransferFunction Tests\n";

		test_dft_impulse();
		test_dft_dc();
		test_dft_idft_roundtrip();
		test_dft_parseval();

		test_fft_matches_dft();
		test_fft_inverse_roundtrip();
		test_fft_parseval();
		test_fft_sine_peak();
		test_fft_magnitude_db();
		test_fft_zero_padding();

		test_tf_evaluate();
		test_tf_first_order_lp();
		test_tf_stability();
		test_tf_cascade();

		std::cout << "All spectral & TF tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << '\n';
		return 1;
	}
}
