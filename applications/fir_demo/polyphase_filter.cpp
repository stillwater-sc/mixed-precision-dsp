// polyphase_filter.cpp: demonstrate polyphase interpolation and decimation
//
// 1. Design a lowpass filter at the high (interpolated) rate.
// 2. 4× interpolate a test signal via polyphase vs. naive upsample+FIR.
// 3. 4× decimate the interpolated signal back to the original rate.
// 4. Print the first few samples, error, and basic timing comparison.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/filter/fir/fir.hpp>
#include <sw/dsp/signals/sampling.hpp>
#include <sw/dsp/signals/generators.hpp>
#include <sw/dsp/math/constants.hpp>

#include <chrono>
#include <cmath>
#include <iostream>

using namespace sw::dsp;

int main() {
	constexpr std::size_t L       = 4;       // interpolation / decimation factor
	constexpr std::size_t N       = 1000;    // input signal length
	constexpr std::size_t M_TAPS  = 63;      // lowpass filter length (full rate)
	constexpr double      FS      = 44100.0; // original sample rate
	constexpr double      F_TONE  = 1000.0;  // test tone frequency

	// ----- Generate test signal: a pure sine -----
	mtl::vec::dense_vector<double> x(N);
	for (std::size_t i = 0; i < N; ++i) {
		double t = static_cast<double>(i) / FS;
		x[i] = std::sin(two_pi * F_TONE * t);
	}

	// ----- Design a simple Gaussian-windowed lowpass -----
	mtl::vec::dense_vector<double> taps(M_TAPS);
	{
		double center = static_cast<double>(M_TAPS - 1) / 2.0;
		double sum = 0.0;
		for (std::size_t i = 0; i < M_TAPS; ++i) {
			double d = (static_cast<double>(i) - center) / 4.0;
			taps[i] = std::exp(-d * d * 0.5);
			sum += taps[i];
		}
		for (std::size_t i = 0; i < M_TAPS; ++i) taps[i] /= sum;
	}

	// ----- Polyphase interpolation -----
	PolyphaseInterpolator<double> interp(taps, L);
	auto t0 = std::chrono::high_resolution_clock::now();
	auto y_poly = interp.process_block(std::span<const double>(x.data(), x.size()));
	auto t1 = std::chrono::high_resolution_clock::now();
	double us_poly = std::chrono::duration<double, std::micro>(t1 - t0).count();

	// ----- Naive: upsample then FIR -----
	auto t2 = std::chrono::high_resolution_clock::now();
	auto x_up = upsample(x, L);
	FIRFilter<double> fir(taps);
	mtl::vec::dense_vector<double> y_naive(x_up.size());
	for (std::size_t i = 0; i < x_up.size(); ++i) {
		y_naive[i] = fir.process(x_up[i]);
	}
	auto t3 = std::chrono::high_resolution_clock::now();
	double us_naive = std::chrono::duration<double, std::micro>(t3 - t2).count();

	// ----- Compare -----
	double max_err = 0.0;
	std::size_t cmp_len = std::min(y_poly.size(), y_naive.size());
	for (std::size_t i = 0; i < cmp_len; ++i) {
		double d = std::abs(y_poly[i] - y_naive[i]);
		if (d > max_err) max_err = d;
	}

	std::cout << "=== Polyphase Interpolation Demo ===\n";
	std::cout << "Input:        " << N << " samples @ " << FS << " Hz\n";
	std::cout << "Tone:         " << F_TONE << " Hz\n";
	std::cout << "Filter:       " << M_TAPS << " taps\n";
	std::cout << "Factor:       " << L << "x (" << FS << " -> " << FS * L << " Hz)\n";
	std::cout << "Output:       " << y_poly.size() << " samples\n";
	std::cout << "\nFirst 8 interpolated samples (polyphase vs naive):\n";
	for (std::size_t i = 0; i < 8 && i < cmp_len; ++i) {
		std::cout << "  y[" << i << "] = " << y_poly[i]
		          << "  (naive " << y_naive[i] << ")\n";
	}
	std::cout << "\nMax |polyphase - naive|: " << max_err << "\n";
	std::cout << "Polyphase time:  " << us_poly << " us\n";
	std::cout << "Naive time:      " << us_naive << " us\n";
	std::cout << "Speedup:         " << us_naive / us_poly << "x\n";

	// ----- Polyphase decimation: go back to original rate -----
	PolyphaseDecimator<double> dec(taps, L);
	auto y_dec = dec.process_block(std::span<const double>(y_poly.data(), y_poly.size()));

	std::cout << "\n=== Polyphase Decimation ===\n";
	std::cout << "Decimated output: " << y_dec.size() << " samples\n";
	std::cout << "First 8 samples:\n";
	for (std::size_t i = 0; i < 8 && i < y_dec.size(); ++i) {
		std::cout << "  y_dec[" << i << "] = " << y_dec[i] << "\n";
	}

	// ----- Overlap-add one-shot convolution -----
	auto y_oa = overlap_add_convolve(x, taps);
	std::cout << "\n=== Overlap-Add Convolution ===\n";
	std::cout << "Output length: " << y_oa.size()
	          << " (expected " << x.size() + taps.size() - 1 << ")\n";

	return 0;
}
