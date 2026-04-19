// test_remez.cpp: Parks-McClellan (Remez exchange) equiripple FIR design tests
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/filter/fir/remez.hpp>
#include <sw/dsp/filter/fir/fir_filter.hpp>
#include <sw/dsp/math/constants.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sw::dsp;

constexpr double tolerance = 1e-4;

bool near(double a, double b, double eps = tolerance) {
	return std::abs(a - b) < eps;
}

void check(bool condition, const std::string& msg) {
	if (!condition) throw std::runtime_error("test failed: " + msg);
}

// Test 1: Basic equiripple lowpass design — verify filter produces taps
// and has the right number of them
void test_basic_lowpass() {
	std::size_t N = 31;
	std::vector<double> bands    = {0.0, 0.2, 0.3, 0.5};
	std::vector<double> desired  = {1.0, 1.0, 0.0, 0.0};
	std::vector<double> weights  = {1.0, 1.0};

	auto taps = remez<double>(N, bands, desired, weights);

	check(taps.size() == N, "tap count is " + std::to_string(taps.size()) + ", expected " + std::to_string(N));

	// All taps should be finite
	for (std::size_t i = 0; i < N; ++i) {
		check(std::isfinite(taps[i]),
		      "tap[" + std::to_string(i) + "] = " + std::to_string(taps[i]) + " is not finite");
	}

	std::cout << "  basic_lowpass: passed (N=" << N << ")\n";
}

// Test 2: Symmetric impulse response (linear phase)
// Type I (odd taps) should have h[n] = h[N-1-n]
void test_linear_phase_symmetry() {
	std::size_t N = 31;
	std::vector<double> bands    = {0.0, 0.2, 0.3, 0.5};
	std::vector<double> desired  = {1.0, 1.0, 0.0, 0.0};
	std::vector<double> weights  = {1.0, 1.0};

	auto taps = remez<double>(N, bands, desired, weights);

	for (std::size_t i = 0; i < N / 2; ++i) {
		check(near(taps[i], taps[N - 1 - i], 1e-10),
		      "symmetry: h[" + std::to_string(i) + "]=" + std::to_string(taps[i]) +
		      " != h[" + std::to_string(N-1-i) + "]=" + std::to_string(taps[N-1-i]));
	}

	std::cout << "  linear_phase_symmetry: passed\n";
}

// Test 3: DC gain should be near 1.0 for a lowpass
void test_dc_gain() {
	std::size_t N = 31;
	std::vector<double> bands    = {0.0, 0.2, 0.3, 0.5};
	std::vector<double> desired  = {1.0, 1.0, 0.0, 0.0};
	std::vector<double> weights  = {1.0, 1.0};

	auto taps = remez<double>(N, bands, desired, weights);

	double dc_sum = 0.0;
	for (std::size_t i = 0; i < N; ++i)
		dc_sum += taps[i];

	double dc_db = 20.0 * std::log10(std::abs(dc_sum));

	check(near(dc_db, 0.0, 1.0),
	      "DC gain = " + std::to_string(dc_db) + " dB, expected near 0 dB");

	std::cout << "  dc_gain: passed (" << dc_db << " dB)\n";
}

// Test 4: Stopband rejection — response should be small above stopband edge
void test_stopband_rejection() {
	std::size_t N = 31;
	std::vector<double> bands    = {0.0, 0.2, 0.3, 0.5};
	std::vector<double> desired  = {1.0, 1.0, 0.0, 0.0};
	std::vector<double> weights  = {1.0, 1.0};

	auto taps = remez<double>(N, bands, desired, weights);

	// Evaluate frequency response at several stopband frequencies
	for (double f = 0.35; f <= 0.49; f += 0.05) {
		double re = 0.0, im = 0.0;
		for (std::size_t n = 0; n < N; ++n) {
			double w = two_pi * f * static_cast<double>(n);
			re += taps[n] * std::cos(w);
			im -= taps[n] * std::sin(w);
		}
		double mag = std::sqrt(re * re + im * im);
		double db = 20.0 * std::log10(mag + 1e-30);

		check(db < -10.0,
		      "stopband at f=" + std::to_string(f) + ": " + std::to_string(db) +
		      " dB (expected < -10 dB)");
	}

	std::cout << "  stopband_rejection: passed\n";
}

// Test 5: Passband flatness — response should be near 1.0 in passband
void test_passband_flatness() {
	std::size_t N = 31;
	std::vector<double> bands    = {0.0, 0.2, 0.3, 0.5};
	std::vector<double> desired  = {1.0, 1.0, 0.0, 0.0};
	std::vector<double> weights  = {1.0, 1.0};

	auto taps = remez<double>(N, bands, desired, weights);

	for (double f = 0.01; f <= 0.19; f += 0.03) {
		double re = 0.0, im = 0.0;
		for (std::size_t n = 0; n < N; ++n) {
			double w = two_pi * f * static_cast<double>(n);
			re += taps[n] * std::cos(w);
			im -= taps[n] * std::sin(w);
		}
		double mag = std::sqrt(re * re + im * im);
		double db = 20.0 * std::log10(mag);

		check(std::abs(db) < 3.0,
		      "passband at f=" + std::to_string(f) + ": " + std::to_string(db) +
		      " dB (expected within 3 dB of 0)");
	}

	std::cout << "  passband_flatness: passed\n";
}

// Test 6: Even tap count (Type II)
void test_even_taps() {
	std::size_t N = 32;
	std::vector<double> bands    = {0.0, 0.2, 0.3, 0.5};
	std::vector<double> desired  = {1.0, 1.0, 0.0, 0.0};
	std::vector<double> weights  = {1.0, 1.0};

	auto taps = remez<double>(N, bands, desired, weights);
	check(taps.size() == N, "even tap count");

	// Should still have symmetry
	for (std::size_t i = 0; i < N / 2; ++i) {
		check(near(taps[i], taps[N - 1 - i], 1e-10),
		      "even symmetry at " + std::to_string(i));
	}

	// DC gain should be near 1.0
	double dc = 0.0;
	for (std::size_t i = 0; i < N; ++i) dc += taps[i];
	check(std::abs(dc - 1.0) < 0.5, "even DC gain = " + std::to_string(dc));

	std::cout << "  even_taps: passed (N=" << N << ")\n";
}

// Test 7: Convenience wrapper — equiripple lowpass
void test_convenience_lowpass() {
	auto taps = design_fir_equiripple_lowpass<double>(31, 0.2, 0.3);
	check(taps.size() == 31, "convenience lowpass tap count");

	double dc = 0.0;
	for (std::size_t i = 0; i < taps.size(); ++i) dc += taps[i];
	check(std::abs(dc - 1.0) < 0.5, "convenience lowpass DC gain = " + std::to_string(dc));

	std::cout << "  convenience_lowpass: passed\n";
}

// Test 8: Convenience wrapper — equiripple bandpass
void test_convenience_bandpass() {
	auto taps = design_fir_equiripple_bandpass<double>(51, 0.1, 0.2, 0.3, 0.4);
	check(taps.size() == 51, "convenience bandpass tap count");

	// DC gain should be near 0 (bandpass)
	double dc = 0.0;
	for (std::size_t i = 0; i < taps.size(); ++i) dc += taps[i];
	check(std::abs(dc) < 0.3, "convenience bandpass DC gain = " + std::to_string(dc));

	std::cout << "  convenience_bandpass: passed\n";
}

// Test 9: Taps are usable with FIRFilter
void test_fir_integration() {
	auto taps = design_fir_equiripple_lowpass<double>(31, 0.2, 0.3);

	FIRFilter<double> f(taps);
	check(f.num_taps() == 31, "FIRFilter tap count");

	// Process impulse
	double y0 = f.process(1.0);
	check(std::isfinite(y0), "FIR impulse response finite");

	for (int i = 0; i < 50; ++i) {
		double y = f.process(0.0);
		check(std::isfinite(y), "FIR zero-input response finite");
	}

	std::cout << "  fir_integration: passed\n";
}

// Test 10: Input validation
void test_validation() {
	bool caught = false;

	// Too few taps
	try {
		remez<double>(2, {0.0, 0.2, 0.3, 0.5}, {1.0, 1.0, 0.0, 0.0}, {1.0, 1.0});
	} catch (const std::invalid_argument&) {
		caught = true;
	}
	check(caught, "num_taps < 3 should throw");

	// Odd number of band edges
	caught = false;
	try {
		remez<double>(31, {0.0, 0.2, 0.3}, {1.0, 1.0, 0.0}, {1.0});
	} catch (const std::invalid_argument&) {
		caught = true;
	}
	check(caught, "odd band edges should throw");

	// Mismatched desired/bands
	caught = false;
	try {
		remez<double>(31, {0.0, 0.2, 0.3, 0.5}, {1.0, 1.0}, {1.0, 1.0});
	} catch (const std::invalid_argument&) {
		caught = true;
	}
	check(caught, "mismatched desired size should throw");

	std::cout << "  validation: passed\n";
}

int main() {
	try {
		std::cout << "Parks-McClellan (Remez) equiripple FIR design tests\n";

		test_basic_lowpass();
		test_linear_phase_symmetry();
		test_dc_gain();
		test_stopband_rejection();
		test_passband_flatness();
		test_even_taps();
		test_convenience_lowpass();
		test_convenience_bandpass();
		test_fir_integration();
		test_validation();

		std::cout << "All Remez tests passed.\n";
		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << "\n";
		return 1;
	}
}
