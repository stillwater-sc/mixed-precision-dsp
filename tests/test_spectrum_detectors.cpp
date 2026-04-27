// test_spectrum_detectors.cpp: tests for the spectrum-analyzer detector
// modes (peak / sample / average / RMS / negative-peak).
//
// Coverage:
//   - Known-answer reductions on a square wave (peak, neg-peak, RMS)
//   - Known-answer reductions on a sine (RMS = amp / sqrt(2))
//   - Constant-bin sanity (all five modes return the constant or |constant|)
//   - Edge cases: empty span throws, single sample
//   - Runtime dispatcher matches the named entry points
//   - Mixed-precision sanity: float input gives reasonable output for all
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)`.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <iostream>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <sw/dsp/spectrum/detectors.hpp>

using namespace sw::dsp::spectrum;

#define REQUIRE(cond) \
	do { if (!(cond)) throw std::runtime_error( \
		std::string("test failed: ") + #cond + \
		" at " __FILE__ ":" + std::to_string(__LINE__)); } while (0)

static bool approx(double a, double b, double tol) {
	return std::abs(a - b) <= tol;
}

// ============================================================================
// Known-answer reductions on a +-A square wave
// ============================================================================

void test_square_wave_detectors() {
	// 100-sample square wave: 50 samples at +2.0, then 50 at -1.0.
	std::vector<double> sq(100);
	for (std::size_t n = 0; n < 50; ++n)  sq[n] = 2.0;
	for (std::size_t n = 50; n < 100; ++n) sq[n] = -1.0;
	std::span<const double> bin{sq};

	REQUIRE(approx(detect_peak(bin),         2.0,                 1e-12));
	REQUIRE(approx(detect_negative_peak(bin), -1.0,               1e-12));
	REQUIRE(approx(detect_average(bin),       0.5,                1e-12));
	REQUIRE(approx(detect_rms(bin),           std::sqrt(2.5),     1e-12));
	REQUIRE(approx(detect_sample(bin),        2.0,                1e-12));   // first sample
	std::cout << "  square_wave_detectors: passed (peak=2, neg=-1, rms="
	          << detect_rms(bin) << ")\n";
}

// ============================================================================
// Known-answer reductions on a unit-amplitude sine
// ============================================================================

void test_sine_detectors() {
	// 1024 samples of one full cycle of unit-amplitude sine.
	const std::size_t N = 1024;
	const double pi = std::numbers::pi_v<double>;
	std::vector<double> s(N);
	for (std::size_t n = 0; n < N; ++n)
		s[n] = std::sin(2.0 * pi * static_cast<double>(n) / static_cast<double>(N));
	std::span<const double> bin{s};

	REQUIRE(approx(detect_peak(bin),          1.0,                 1e-3));   // close to +1
	REQUIRE(approx(detect_negative_peak(bin), -1.0,                1e-3));
	REQUIRE(approx(detect_average(bin),        0.0,                1e-12));  // full cycle
	REQUIRE(approx(detect_rms(bin),            1.0 / std::sqrt(2.0), 1e-3));
	std::cout << "  sine_detectors: passed (peak="
	          << detect_peak(bin) << ", rms="
	          << detect_rms(bin) << ")\n";
}

// ============================================================================
// Constant bin: all five modes return the constant (or |constant| for RMS)
// ============================================================================

void test_constant_bin() {
	std::vector<double> flat(64, 1.5);
	std::span<const double> bin{flat};
	REQUIRE(approx(detect_peak(bin),          1.5, 1e-12));
	REQUIRE(approx(detect_negative_peak(bin), 1.5, 1e-12));
	REQUIRE(approx(detect_sample(bin),        1.5, 1e-12));
	REQUIRE(approx(detect_average(bin),       1.5, 1e-12));
	REQUIRE(approx(detect_rms(bin),           1.5, 1e-12));   // sqrt(1.5^2) = 1.5
	std::cout << "  constant_bin: passed\n";
}

void test_constant_negative_bin() {
	// Make sure RMS returns |value|, not value, for a constant negative.
	std::vector<double> flat(32, -2.0);
	std::span<const double> bin{flat};
	REQUIRE(approx(detect_peak(bin),         -2.0, 1e-12));
	REQUIRE(approx(detect_negative_peak(bin), -2.0, 1e-12));
	REQUIRE(approx(detect_average(bin),       -2.0, 1e-12));
	REQUIRE(approx(detect_rms(bin),            2.0, 1e-12));   // |-2| = 2
	std::cout << "  constant_negative_bin: passed (RMS = " << detect_rms(bin) << ")\n";
}

// ============================================================================
// Edge cases
// ============================================================================

void test_empty_span_throws() {
	std::span<const double> empty{};
	bool t1 = false, t2 = false, t3 = false, t4 = false, t5 = false;
	try { (void)detect_peak(empty); }           catch (const std::invalid_argument&) { t1 = true; }
	try { (void)detect_negative_peak(empty); }  catch (const std::invalid_argument&) { t2 = true; }
	try { (void)detect_sample(empty); }         catch (const std::invalid_argument&) { t3 = true; }
	try { (void)detect_average(empty); }        catch (const std::invalid_argument&) { t4 = true; }
	try { (void)detect_rms(empty); }            catch (const std::invalid_argument&) { t5 = true; }
	REQUIRE(t1);
	REQUIRE(t2);
	REQUIRE(t3);
	REQUIRE(t4);
	REQUIRE(t5);

	// Dispatcher path also throws.
	bool t_disp = false;
	try { (void)detect(empty, DetectorMode::Peak); }
	catch (const std::invalid_argument&) { t_disp = true; }
	REQUIRE(t_disp);
	std::cout << "  empty_span_throws: passed\n";
}

void test_single_sample() {
	std::vector<double> one = {3.5};
	std::span<const double> bin{one};
	REQUIRE(approx(detect_peak(bin),          3.5, 1e-12));
	REQUIRE(approx(detect_negative_peak(bin), 3.5, 1e-12));
	REQUIRE(approx(detect_sample(bin),        3.5, 1e-12));
	REQUIRE(approx(detect_average(bin),       3.5, 1e-12));
	REQUIRE(approx(detect_rms(bin),           3.5, 1e-12));
	std::cout << "  single_sample: passed\n";
}

// ============================================================================
// Runtime dispatcher
// ============================================================================

void test_dispatcher_matches_named() {
	// For an arbitrary bin, the dispatcher must agree bit-exactly with
	// the named entry points (no rounding difference allowed: same code path).
	std::vector<double> bin = {0.3, -0.7, 1.2, 0.0, -0.4, 0.9};
	std::span<const double> b{bin};
	REQUIRE(detect(b, DetectorMode::Peak)         == detect_peak(b));
	REQUIRE(detect(b, DetectorMode::NegativePeak) == detect_negative_peak(b));
	REQUIRE(detect(b, DetectorMode::Sample)       == detect_sample(b));
	REQUIRE(detect(b, DetectorMode::Average)      == detect_average(b));
	REQUIRE(detect(b, DetectorMode::RMS)          == detect_rms(b));
	std::cout << "  dispatcher_matches_named: passed\n";
}

// ============================================================================
// Mixed-precision sanity: float input
// ============================================================================

void test_float_input() {
	// Same square wave as the double test but with float samples.
	// Detectors run in double internally so the result should match
	// the double answer to within float epsilon scaled by N.
	std::vector<float> sq(100);
	for (std::size_t n = 0; n < 50; ++n) sq[n] = 2.0f;
	for (std::size_t n = 50; n < 100; ++n) sq[n] = -1.0f;
	std::span<const float> bin{sq};

	REQUIRE(approx(detect_peak(bin),          2.0,            1e-6));
	REQUIRE(approx(detect_negative_peak(bin), -1.0,           1e-6));
	REQUIRE(approx(detect_average(bin),        0.5,           1e-6));
	REQUIRE(approx(detect_rms(bin),            std::sqrt(2.5), 1e-6));
	std::cout << "  float_input: passed\n";
}

// ============================================================================
// main
// ============================================================================

int main() {
	try {
		std::cout << "test_spectrum_detectors\n";

		test_square_wave_detectors();
		test_sine_detectors();
		test_constant_bin();
		test_constant_negative_bin();

		test_empty_span_throws();
		test_single_sample();
		test_dispatcher_matches_named();

		test_float_input();

		std::cout << "all tests passed\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "FAILED: " << ex.what() << "\n";
		return 1;
	}
}
