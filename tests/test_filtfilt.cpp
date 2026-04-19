// test_filtfilt.cpp: zero-phase forward-backward filtering tests
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/filter/iir/butterworth.hpp>
#include <sw/dsp/filter/filtfilt.hpp>
#include <sw/dsp/math/constants.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sw::dsp;

constexpr double tolerance = 1e-6;

bool near(double a, double b, double eps = tolerance) {
	return std::abs(a - b) < eps;
}

void check(bool condition, const std::string& msg) {
	if (!condition) throw std::runtime_error("test failed: " + msg);
}

// Test 1: filtfilt output should have zero phase shift.
// A symmetric signal filtered with filtfilt should remain symmetric.
void test_zero_phase() {
	iir::ButterworthLowPass<4> filter;
	filter.setup(4, 1000.0, 100.0);
	const auto& cascade = filter.cascade();

	// Symmetric signal: triangle pulse centered at N/2
	constexpr int N = 200;
	std::vector<double> input(N, 0.0);
	int center = N / 2;
	int half_width = 20;
	for (int i = -half_width; i <= half_width; ++i) {
		input[center + i] = 1.0 - std::abs(static_cast<double>(i)) / half_width;
	}

	auto output = filtfilt(cascade, input);
	check(output.size() == static_cast<std::size_t>(N), "output size matches input");

	// Output should be symmetric around center
	for (int i = 1; i < N / 2; ++i) {
		double left  = output[static_cast<std::size_t>(center - i)];
		double right = output[static_cast<std::size_t>(center + i)];
		check(near(left, right, 1e-10),
		      "symmetry at offset " + std::to_string(i) +
		      " left=" + std::to_string(left) + " right=" + std::to_string(right));
	}

	std::cout << "  zero_phase: passed\n";
}

// Test 2: filtfilt magnitude should be the square of single-pass magnitude.
// At the cutoff frequency, single-pass is -3 dB, filtfilt should be -6 dB.
void test_squared_magnitude() {
	iir::ButterworthLowPass<4> filter;
	filter.setup(4, 44100.0, 1000.0);
	const auto& cascade = filter.cascade();

	// Single-pass magnitude at cutoff
	double fc = 1000.0 / 44100.0;
	auto r_single = cascade.response(fc);
	double mag_single = std::abs(r_single);
	double db_single = 20.0 * std::log10(mag_single);

	// filtfilt effective magnitude is |H(f)|^2, so dB doubles
	double db_filtfilt_expected = 2.0 * db_single;

	// Generate a sinusoid at the cutoff frequency
	constexpr int N = 8820;  // ~200ms at 44100 Hz
	std::vector<double> input(N);
	for (int n = 0; n < N; ++n) {
		input[n] = std::sin(2.0 * pi * fc * n);
	}

	auto output = filtfilt(cascade, input);

	// Measure output amplitude in the steady-state region (middle)
	double max_out = 0.0;
	for (int n = N / 3; n < 2 * N / 3; ++n) {
		max_out = std::max(max_out, std::abs(output[n]));
	}
	double db_measured = 20.0 * std::log10(max_out);

	check(near(db_measured, db_filtfilt_expected, 1.0),
	      "squared magnitude: expected " + std::to_string(db_filtfilt_expected) +
	      " dB, got " + std::to_string(db_measured) + " dB");

	std::cout << "  squared_magnitude: passed (single=" << db_single
	          << " dB, filtfilt=" << db_measured << " dB)\n";
}

// Test 3: all three state forms should produce the same result
void test_state_forms_agree() {
	iir::ButterworthLowPass<4> filter;
	filter.setup(4, 1000.0, 100.0);
	const auto& cascade = filter.cascade();

	constexpr int N = 200;
	std::vector<double> input(N);
	for (int n = 0; n < N; ++n) {
		input[n] = std::sin(2.0 * pi * 50.0 / 1000.0 * n)
		         + 0.5 * std::sin(2.0 * pi * 300.0 / 1000.0 * n);
	}

	using DFI  = DirectFormI<double>;
	using DFII = DirectFormII<double>;
	using TDFII = TransposedDirectFormII<double>;
	auto out_df1  = sw::dsp::filtfilt<DFI>(cascade, input);
	auto out_df2  = sw::dsp::filtfilt<DFII>(cascade, input);
	auto out_tdf2 = sw::dsp::filtfilt<TDFII>(cascade, input);

	for (int n = 0; n < N; ++n) {
		check(near(out_df1[n], out_df2[n], 1e-10),
		      "DFI vs DFII at sample " + std::to_string(n));
		check(near(out_df2[n], out_tdf2[n], 1e-10),
		      "DFII vs TDFII at sample " + std::to_string(n));
	}

	std::cout << "  state_forms_agree: passed\n";
}

// Test 4: filtfilt of a DC signal should return the same DC level
void test_dc_passthrough() {
	iir::ButterworthLowPass<4> filter;
	filter.setup(4, 1000.0, 100.0);
	const auto& cascade = filter.cascade();

	constexpr int N = 500;
	double dc_level = 0.42;
	std::vector<double> input(N, dc_level);

	auto output = filtfilt(cascade, input);

	// Interior samples should be near-perfect; edges have small transient artifacts
	for (int n = N / 10; n < 9 * N / 10; ++n) {
		check(near(output[n], dc_level, 1e-6),
		      "DC passthrough at sample " + std::to_string(n) +
		      " got " + std::to_string(output[n]));
	}
	// Edge samples should still be within 2%
	for (int n = 0; n < N; ++n) {
		check(near(output[n], dc_level, 0.02),
		      "DC edge tolerance at sample " + std::to_string(n) +
		      " got " + std::to_string(output[n]));
	}

	std::cout << "  dc_passthrough: passed\n";
}

// Test 5: filtfilt should reject frequencies above the cutoff
void test_stopband_rejection() {
	iir::ButterworthLowPass<4> filter;
	filter.setup(4, 1000.0, 50.0);  // 50 Hz cutoff at 1000 Hz sample rate
	const auto& cascade = filter.cascade();

	// Pure high-frequency signal at 400 Hz (well into stopband)
	constexpr int N = 1000;
	std::vector<double> input(N);
	for (int n = 0; n < N; ++n) {
		input[n] = std::sin(2.0 * pi * 400.0 / 1000.0 * n);
	}

	auto output = filtfilt(cascade, input);

	// Output energy should be tiny relative to input
	double input_energy = 0.0, output_energy = 0.0;
	for (int n = N / 4; n < 3 * N / 4; ++n) {
		input_energy  += input[n] * input[n];
		output_energy += output[n] * output[n];
	}

	double attenuation_db = 10.0 * std::log10(output_energy / input_energy);
	check(attenuation_db < -80.0,
	      "stopband rejection: " + std::to_string(attenuation_db) + " dB (expected < -80 dB)");

	std::cout << "  stopband_rejection: passed (" << attenuation_db << " dB)\n";
}

// Test 6: empty and short input edge cases
void test_edge_cases() {
	iir::ButterworthLowPass<4> filter;
	filter.setup(4, 1000.0, 100.0);
	const auto& cascade = filter.cascade();

	// Empty input
	std::vector<double> empty;
	auto out_empty = filtfilt(cascade, empty);
	check(out_empty.empty(), "empty input returns empty output");

	// Single sample
	std::vector<double> single = {1.0};
	auto out_single = filtfilt(cascade, single);
	check(out_single.size() == 1, "single sample returns single sample");
	check(std::isfinite(out_single[0]), "single sample output is finite");

	// Two samples
	std::vector<double> two = {1.0, 0.5};
	auto out_two = filtfilt(cascade, two);
	check(out_two.size() == 2, "two samples returns two samples");
	check(std::isfinite(out_two[0]) && std::isfinite(out_two[1]),
	      "two sample outputs are finite");

	std::cout << "  edge_cases: passed\n";
}

// Test 7: convenience overload (no explicit StateForm) works
void test_default_state_form() {
	iir::ButterworthLowPass<2> filter;
	filter.setup(2, 1000.0, 100.0);
	const auto& cascade = filter.cascade();

	constexpr int N = 100;
	std::vector<double> input(N);
	for (int n = 0; n < N; ++n) {
		input[n] = std::sin(2.0 * pi * 50.0 / 1000.0 * n);
	}

	// This calls the convenience overload (no StateForm template argument)
	auto output = filtfilt(cascade, input);
	check(output.size() == static_cast<std::size_t>(N), "default overload returns correct size");

	// Compare with explicit DFII
	using DFII = DirectFormII<double>;
	auto output_explicit = sw::dsp::filtfilt<DFII>(cascade, input);
	for (int n = 0; n < N; ++n) {
		check(near(output[n], output_explicit[n], 1e-15),
		      "default matches explicit DFII at sample " + std::to_string(n));
	}

	std::cout << "  default_state_form: passed\n";
}

int main() {
	try {
		std::cout << "filtfilt (zero-phase filtering) tests\n";

		test_zero_phase();
		test_squared_magnitude();
		test_state_forms_agree();
		test_dc_passthrough();
		test_stopband_rejection();
		test_edge_cases();
		test_default_state_form();

		std::cout << "All filtfilt tests passed.\n";
		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << "\n";
		return 1;
	}
}
