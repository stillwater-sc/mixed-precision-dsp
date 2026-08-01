// regression_fft_roundtrip.cpp: large-N FFT round-trip stability check.
//
// Runs forward FFT + inverse FFT at increasing power-of-two sizes and
// verifies that the maximum absolute reconstruction error stays under
// a size-scaled tolerance. Catches numerical drift and twiddle-factor
// accumulation bugs that a small-N unit test would miss.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/spectral/fft.hpp>

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

using namespace sw::dsp;
using namespace sw::dsp::spectral;

static void run_roundtrip(std::size_t N, double tolerance) {
	std::mt19937 rng(0xC0FFEE);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);

	mtl::vec::dense_vector<double> x(N);
	for (std::size_t i = 0; i < N; ++i) x[i] = dist(rng);

	auto X = fft<double>(x);
	auto y = ifft_real<double>(X);

	if (y.size() != N)
		throw std::runtime_error("regression_fft_roundtrip: size mismatch at N=" + std::to_string(N));

	double max_err = 0.0;
	for (std::size_t i = 0; i < N; ++i) {
		double err = std::abs(y[i] - x[i]);
		if (err > max_err) max_err = err;
	}

	std::cout << "  N=" << N << "  max_err=" << max_err << "  tol=" << tolerance << "\n";

	if (!(max_err < tolerance))
		throw std::runtime_error("regression_fft_roundtrip: error " + std::to_string(max_err) +
		                         " exceeded tolerance " + std::to_string(tolerance) +
		                         " at N=" + std::to_string(N));
}

int main() try {
	// Tolerance scales with sqrt(N) for radix-2 FFT round-off accumulation.
	// Coefficients picked empirically with generous headroom for float64.
	run_roundtrip(1u << 14, 1e-10);   // 16k
	run_roundtrip(1u << 16, 1e-10);   // 64k
	run_roundtrip(1u << 18, 1e-9);    // 256k

	std::cout << "regression_fft_roundtrip: passed\n";
	return 0;
} catch (const std::exception& e) {
	std::cerr << "regression_fft_roundtrip: FAILED — " << e.what() << "\n";
	return 1;
}
