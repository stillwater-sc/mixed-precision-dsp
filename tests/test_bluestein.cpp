// test_bluestein.cpp: Bluestein chirp-z arbitrary-length DFT tests
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/spectral/bluestein.hpp>
#include <sw/dsp/spectral/dft.hpp>
#include <sw/dsp/math/constants.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace sw::dsp;
using namespace sw::dsp::spectral;

constexpr double tolerance = 1e-8;

bool near(double a, double b, double eps = tolerance) {
	return std::abs(a - b) < eps;
}

void check(bool condition, const std::string& msg) {
	if (!condition) throw std::runtime_error("test failed: " + msg);
}

// Test 1: Bluestein matches naive DFT for prime length 7
void test_prime_7() {
	constexpr std::size_t N = 7;
	using complex_t = std::complex<double>;

	mtl::vec::dense_vector<complex_t> x(N);
	for (std::size_t i = 0; i < N; ++i) {
		double t = static_cast<double>(i) / static_cast<double>(N);
		x[i] = complex_t(std::cos(two_pi * 2.0 * t), std::sin(two_pi * 3.0 * t));
	}

	auto X_blue = bluestein_forward<double>(x);

	mtl::vec::dense_vector<double> xr(N);
	for (std::size_t i = 0; i < N; ++i) xr[i] = x[i].real();
	auto X_ref = dft(xr);

	// Compare against complex DFT computed manually
	for (std::size_t k = 0; k < N; ++k) {
		complex_t ref{};
		for (std::size_t n = 0; n < N; ++n) {
			double angle = -two_pi * static_cast<double>(k) * static_cast<double>(n)
			               / static_cast<double>(N);
			complex_t tw(std::cos(angle), std::sin(angle));
			ref += x[n] * tw;
		}
		check(near(X_blue[k].real(), ref.real(), 1e-6),
		      "prime_7 real[" + std::to_string(k) + "]=" +
		      std::to_string(X_blue[k].real()) + " vs " + std::to_string(ref.real()));
		check(near(X_blue[k].imag(), ref.imag(), 1e-6),
		      "prime_7 imag[" + std::to_string(k) + "]");
	}

	std::cout << "  prime_7: passed\n";
}

// Test 2: Bluestein matches naive DFT for prime length 13
void test_prime_13() {
	constexpr std::size_t N = 13;
	using complex_t = std::complex<double>;

	mtl::vec::dense_vector<double> xr(N);
	for (std::size_t i = 0; i < N; ++i)
		xr[i] = std::sin(two_pi * 3.0 * static_cast<double>(i) / static_cast<double>(N));

	auto X_ref = dft(xr);

	mtl::vec::dense_vector<complex_t> xc(N);
	for (std::size_t i = 0; i < N; ++i) xc[i] = complex_t(xr[i], 0.0);
	auto X_blue = bluestein_forward<double>(xc);

	for (std::size_t k = 0; k < N; ++k) {
		check(near(X_blue[k].real(), X_ref[k].real(), 1e-6),
		      "prime_13 real[" + std::to_string(k) + "]");
		check(near(X_blue[k].imag(), X_ref[k].imag(), 1e-6),
		      "prime_13 imag[" + std::to_string(k) + "]");
	}

	std::cout << "  prime_13: passed\n";
}

// Test 3: Large prime 127
void test_prime_127() {
	constexpr std::size_t N = 127;
	using complex_t = std::complex<double>;

	mtl::vec::dense_vector<double> xr(N);
	for (std::size_t i = 0; i < N; ++i)
		xr[i] = std::cos(two_pi * 5.0 * static_cast<double>(i) / static_cast<double>(N));

	auto X_ref = dft(xr);

	mtl::vec::dense_vector<complex_t> xc(N);
	for (std::size_t i = 0; i < N; ++i) xc[i] = complex_t(xr[i], 0.0);
	auto X_blue = bluestein_forward<double>(xc);

	for (std::size_t k = 0; k < N; ++k) {
		check(near(X_blue[k].real(), X_ref[k].real(), 1e-5),
		      "prime_127 real[" + std::to_string(k) + "]");
		check(near(X_blue[k].imag(), X_ref[k].imag(), 1e-5),
		      "prime_127 imag[" + std::to_string(k) + "]");
	}

	std::cout << "  prime_127: passed\n";
}

// Test 4: Large prime 251
void test_prime_251() {
	constexpr std::size_t N = 251;
	using complex_t = std::complex<double>;

	mtl::vec::dense_vector<double> xr(N);
	for (std::size_t i = 0; i < N; ++i)
		xr[i] = std::sin(two_pi * 7.0 * static_cast<double>(i) / static_cast<double>(N))
		       + 0.5 * std::cos(two_pi * 23.0 * static_cast<double>(i) / static_cast<double>(N));

	auto X_ref = dft(xr);

	mtl::vec::dense_vector<complex_t> xc(N);
	for (std::size_t i = 0; i < N; ++i) xc[i] = complex_t(xr[i], 0.0);
	auto X_blue = bluestein_forward<double>(xc);

	for (std::size_t k = 0; k < N; ++k) {
		check(near(X_blue[k].real(), X_ref[k].real(), 1e-4),
		      "prime_251 real[" + std::to_string(k) + "]");
		check(near(X_blue[k].imag(), X_ref[k].imag(), 1e-4),
		      "prime_251 imag[" + std::to_string(k) + "]");
	}

	std::cout << "  prime_251: passed\n";
}

// Test 5: Parseval's theorem — energy conservation
void test_parseval() {
	constexpr std::size_t N = 17;
	using complex_t = std::complex<double>;

	mtl::vec::dense_vector<complex_t> x(N);
	for (std::size_t i = 0; i < N; ++i) {
		double t = static_cast<double>(i) / static_cast<double>(N);
		x[i] = complex_t(std::sin(two_pi * 3.0 * t), 0.0);
	}

	auto X = bluestein_forward<double>(x);

	double time_energy = 0.0, freq_energy = 0.0;
	for (std::size_t i = 0; i < N; ++i) {
		time_energy += x[i].real() * x[i].real() + x[i].imag() * x[i].imag();
	}
	for (std::size_t k = 0; k < N; ++k) {
		freq_energy += X[k].real() * X[k].real() + X[k].imag() * X[k].imag();
	}
	freq_energy /= static_cast<double>(N);

	check(near(time_energy, freq_energy, 1e-6),
	      "Parseval: time=" + std::to_string(time_energy) +
	      " freq=" + std::to_string(freq_energy));

	std::cout << "  parseval: passed\n";
}

// Test 6: Forward-inverse roundtrip
void test_roundtrip() {
	constexpr std::size_t N = 23;
	using complex_t = std::complex<double>;

	mtl::vec::dense_vector<complex_t> x(N);
	for (std::size_t i = 0; i < N; ++i) {
		double t = static_cast<double>(i) / static_cast<double>(N);
		x[i] = complex_t(std::cos(two_pi * 5.0 * t), std::sin(two_pi * 2.0 * t));
	}

	auto X = bluestein_forward<double>(x);
	auto y = bluestein_inverse<double>(X);

	for (std::size_t i = 0; i < N; ++i) {
		check(near(y[i].real(), x[i].real(), 1e-8),
		      "roundtrip real[" + std::to_string(i) + "]");
		check(near(y[i].imag(), x[i].imag(), 1e-8),
		      "roundtrip imag[" + std::to_string(i) + "]");
	}

	std::cout << "  roundtrip: passed (N=23)\n";
}

// Test 7: Power-of-2 should match radix-2 FFT
void test_power_of_2_match() {
	constexpr std::size_t N = 16;
	using complex_t = std::complex<double>;

	mtl::vec::dense_vector<complex_t> x(N);
	for (std::size_t i = 0; i < N; ++i) {
		x[i] = complex_t(static_cast<double>(i) * 0.1, 0.0);
	}

	auto X_blue = bluestein_forward<double>(x);

	auto x_copy = x;
	fft_forward<double>(x_copy);

	for (std::size_t k = 0; k < N; ++k) {
		check(near(X_blue[k].real(), x_copy[k].real(), 1e-8),
		      "pow2_match real[" + std::to_string(k) + "]");
		check(near(X_blue[k].imag(), x_copy[k].imag(), 1e-8),
		      "pow2_match imag[" + std::to_string(k) + "]");
	}

	std::cout << "  power_of_2_match: passed\n";
}

// Test 8: Auto-dispatching dft_forward/dft_inverse
void test_auto_dispatch() {
	using complex_t = std::complex<double>;

	// Power-of-2: should use radix-2 path
	{
		constexpr std::size_t N = 8;
		mtl::vec::dense_vector<complex_t> x(N);
		for (std::size_t i = 0; i < N; ++i)
			x[i] = complex_t(static_cast<double>(i), 0.0);

		auto X = dft_forward<double>(x);
		auto y = dft_inverse<double>(X);

		for (std::size_t i = 0; i < N; ++i) {
			check(near(y[i].real(), x[i].real(), 1e-10),
			      "dispatch_pow2 real[" + std::to_string(i) + "]");
			check(near(y[i].imag(), x[i].imag(), 1e-10),
			      "dispatch_pow2 imag[" + std::to_string(i) + "]");
		}
	}

	// Non-power-of-2: should use Bluestein path
	{
		constexpr std::size_t N = 11;
		mtl::vec::dense_vector<complex_t> x(N);
		for (std::size_t i = 0; i < N; ++i)
			x[i] = complex_t(std::sin(two_pi * static_cast<double>(i) / static_cast<double>(N)), 0.0);

		auto X = dft_forward<double>(x);
		auto y = dft_inverse<double>(X);

		for (std::size_t i = 0; i < N; ++i) {
			check(near(y[i].real(), x[i].real(), 1e-8),
			      "dispatch_prime real[" + std::to_string(i) + "]");
			check(near(y[i].imag(), x[i].imag(), 1e-8),
			      "dispatch_prime imag[" + std::to_string(i) + "]");
		}
	}

	std::cout << "  auto_dispatch: passed\n";
}

// Test 9: Edge cases — N=1, N=2, N=3
void test_edge_cases() {
	using complex_t = std::complex<double>;

	// N=0
	{
		mtl::vec::dense_vector<complex_t> x;
		auto X = bluestein_forward<double>(x);
		check(X.size() == 0, "edge N=0 size");
	}

	// N=1
	{
		mtl::vec::dense_vector<complex_t> x({complex_t(3.14, 0.0)});
		auto X = bluestein_forward<double>(x);
		check(X.size() == 1, "edge N=1 size");
		check(near(X[0].real(), 3.14, 1e-10), "edge N=1 value");
	}

	// N=2
	{
		mtl::vec::dense_vector<complex_t> x({complex_t(1.0, 0.0), complex_t(2.0, 0.0)});
		auto X = bluestein_forward<double>(x);
		check(X.size() == 2, "edge N=2 size");
		check(near(X[0].real(), 3.0, 1e-8), "edge N=2 DC");
		check(near(X[1].real(), -1.0, 1e-8), "edge N=2 Nyquist");
	}

	// N=3
	{
		mtl::vec::dense_vector<complex_t> x({complex_t(1.0, 0.0), complex_t(1.0, 0.0), complex_t(1.0, 0.0)});
		auto X = bluestein_forward<double>(x);
		check(X.size() == 3, "edge N=3 size");
		check(near(X[0].real(), 3.0, 1e-8), "edge N=3 DC");
		check(near(std::abs(X[1]), 0.0, 1e-8), "edge N=3 bin1");
		check(near(std::abs(X[2]), 0.0, 1e-8), "edge N=3 bin2");
	}

	std::cout << "  edge_cases: passed\n";
}

// Test 10: Impulse response — DFT of [1, 0, ..., 0] = flat spectrum
void test_impulse() {
	constexpr std::size_t N = 19;
	using complex_t = std::complex<double>;

	mtl::vec::dense_vector<complex_t> x(N, complex_t{});
	x[0] = complex_t(1.0, 0.0);

	auto X = bluestein_forward<double>(x);

	for (std::size_t k = 0; k < N; ++k) {
		check(near(X[k].real(), 1.0, 1e-8),
		      "impulse real[" + std::to_string(k) + "]=" + std::to_string(X[k].real()));
		check(near(X[k].imag(), 0.0, 1e-8),
		      "impulse imag[" + std::to_string(k) + "]");
	}

	std::cout << "  impulse: passed (N=19)\n";
}

int main() {
	try {
		std::cout << "Bluestein chirp-z arbitrary-length DFT tests\n";

		test_prime_7();
		test_prime_13();
		test_prime_127();
		test_prime_251();
		test_parseval();
		test_roundtrip();
		test_power_of_2_match();
		test_auto_dispatch();
		test_edge_cases();
		test_impulse();

		std::cout << "All Bluestein tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << '\n';
		return 1;
	}
}
