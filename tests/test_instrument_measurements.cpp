// test_instrument_measurements.cpp: tests for scope-style waveform
// measurements (peak-to-peak, mean, RMS, rise/fall time, period,
// frequency).
//
// Coverage:
//   - Aggregations: square wave, sine wave (known analytical answers)
//   - Linear ramp: rise_time matches the analytical answer
//   - Period/frequency: noise-free sine, with linear interpolation
//     making the result accurate well below 1 sample
//   - Edge cases: empty (throws), single sample, all-equal samples,
//     monotonic (no transition for rise/fall)
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

#include <sw/dsp/instrument/measurements.hpp>

using namespace sw::dsp::instrument;

#define REQUIRE(cond) \
	do { if (!(cond)) throw std::runtime_error( \
		std::string("test failed: ") + #cond + \
		" at " __FILE__ ":" + std::to_string(__LINE__)); } while (0)

static bool approx(double a, double b, double tol) {
	return std::abs(a - b) <= tol;
}

// ============================================================================
// Aggregations on a known square wave
// ============================================================================

void test_square_wave_aggregations() {
	// 100-sample square wave: 50 samples at +2.0, then 50 at -1.0.
	// DC offset = +0.5, peak-to-peak = 3.0, RMS = sqrt((50*4 + 50*1)/100)
	//                                            = sqrt(2.5) ~= 1.5811.
	std::vector<double> sq(100);
	for (std::size_t n = 0; n < 50; ++n) sq[n] = 2.0;
	for (std::size_t n = 50; n < 100; ++n) sq[n] = -1.0;
	std::span<const double> seg{sq};

	REQUIRE(approx(peak_to_peak(seg), 3.0, 1e-12));
	REQUIRE(approx(mean(seg), 0.5, 1e-12));
	REQUIRE(approx(rms(seg), std::sqrt(2.5), 1e-12));
	std::cout << "  square_wave_aggregations: passed (p2p=3, mean=0.5, rms="
	          << rms(seg) << ")\n";
}

// ============================================================================
// Aggregations on a unit-amplitude sine
// ============================================================================

void test_sine_aggregations() {
	// 1024 samples of one full cycle of unit-amplitude sine, no DC.
	const std::size_t N = 1024;
	const double pi = std::numbers::pi_v<double>;
	std::vector<double> s(N);
	for (std::size_t n = 0; n < N; ++n)
		s[n] = std::sin(2.0 * pi * static_cast<double>(n) / static_cast<double>(N));
	std::span<const double> seg{s};

	REQUIRE(approx(peak_to_peak(seg), 2.0, 1e-3));     // exact 2 - tiny grid err
	REQUIRE(approx(mean(seg), 0.0, 1e-12));            // one full cycle
	REQUIRE(approx(rms(seg), 1.0 / std::sqrt(2.0), 1e-3));
	std::cout << "  sine_aggregations: passed (p2p=" << peak_to_peak(seg)
	          << ", mean=" << mean(seg) << ", rms=" << rms(seg) << ")\n";
}

// ============================================================================
// Linear ramp: rise time has a closed-form answer
// ============================================================================

void test_linear_ramp_rise_time() {
	// Ramp from 0 to 1 over 100 samples (samples 0..100).
	// Peak-to-peak = 1.0. Default thresholds: low=0.1, high=0.9.
	// 10% threshold crossed at sample 10, 90% threshold at sample 90.
	// Expected rise time = 80 samples.
	std::vector<double> ramp(101);
	for (std::size_t n = 0; n <= 100; ++n)
		ramp[n] = static_cast<double>(n) / 100.0;

	const double rt = rise_time_samples(std::span<const double>{ramp});
	REQUIRE(approx(rt, 80.0, 1e-9));
	std::cout << "  linear_ramp_rise_time: passed (rise_time=" << rt
	          << " samples, expected 80)\n";
}

void test_steep_step_rise_time() {
	// Regression test for a one-sample step that crosses BOTH the 10%
	// and 90% thresholds within a single sample interval. Previous
	// `else if` logic skipped the t_hi check on the iteration that set
	// t_lo, returning NaN. After the fix both crossings are recovered
	// in the same iteration via linear interpolation.
	std::vector<double> step = {0.0, 0.0, 1.0, 1.0, 1.0};
	const double rt = rise_time_samples(std::span<const double>{step});
	// p2p = 1.0; thr_lo = 0.1, thr_hi = 0.9.
	// Crossing inside sample [1, 2] (a=0, b=1):
	//   t_lo at fraction 0.1 -> absolute 1.1
	//   t_hi at fraction 0.9 -> absolute 1.9
	// Expected rise time = 0.8 samples.
	REQUIRE(!std::isnan(rt));
	REQUIRE(approx(rt, 0.8, 1e-12));
	std::cout << "  steep_step_rise_time: passed (rise_time=" << rt
	          << " samples within one sample step, expected 0.8)\n";
}

void test_steep_step_fall_time() {
	// Mirror of the rise-time steep-step test.
	std::vector<double> step = {1.0, 1.0, 0.0, 0.0, 0.0};
	const double ft = fall_time_samples(std::span<const double>{step});
	// p2p = 1.0; thr_hi = 0.9, thr_lo = 0.1.
	// Crossing inside sample [1, 2] (a=1, b=0):
	//   t_hi at fraction 0.1 (signal hits 0.9 going down) -> absolute 1.1
	//   t_lo at fraction 0.9 (signal hits 0.1 going down) -> absolute 1.9
	// Expected fall time = 0.8 samples.
	REQUIRE(!std::isnan(ft));
	REQUIRE(approx(ft, 0.8, 1e-12));
	std::cout << "  steep_step_fall_time: passed (fall_time=" << ft
	          << " samples within one sample step, expected 0.8)\n";
}

void test_linear_ramp_fall_time() {
	// Ramp from 1 down to 0 over 100 samples.
	// 90% threshold crossed at sample 10, 10% threshold at sample 90.
	// Expected fall time = 80 samples.
	std::vector<double> ramp(101);
	for (std::size_t n = 0; n <= 100; ++n)
		ramp[n] = 1.0 - static_cast<double>(n) / 100.0;

	const double ft = fall_time_samples(std::span<const double>{ramp});
	REQUIRE(approx(ft, 80.0, 1e-9));
	std::cout << "  linear_ramp_fall_time: passed (fall_time=" << ft
	          << " samples, expected 80)\n";
}

// ============================================================================
// Period & frequency: noise-free sine
// ============================================================================

void test_period_and_frequency_sine() {
	// Sample rate 1 MHz, signal 50 kHz. Period = 20 samples.
	const double pi = std::numbers::pi_v<double>;
	const double sample_rate = 1.0e6;
	const double f           = 50.0e3;
	const std::size_t N      = 4000;   // 200 cycles
	std::vector<double> s(N);
	for (std::size_t n = 0; n < N; ++n)
		s[n] = std::sin(2.0 * pi * f * static_cast<double>(n) / sample_rate);
	std::span<const double> seg{s};

	const double T_samp = period_samples(seg);
	const double freq   = frequency_hz(seg, sample_rate);
	REQUIRE(approx(T_samp, 20.0, 1e-6));
	REQUIRE(approx(freq, f, 1.0));   // within 1 Hz of 50 kHz
	std::cout << "  period_and_frequency_sine: passed (T=" << T_samp
	          << " samples, f=" << freq << " Hz)\n";
}

// ============================================================================
// Edge cases
// ============================================================================

void test_empty_segment_throws() {
	std::span<const double> empty{};
	bool threw_p2p = false, threw_mean = false, threw_rms = false;
	try { (void)peak_to_peak(empty); }
	catch (const std::invalid_argument&) { threw_p2p = true; }
	try { (void)mean(empty); }
	catch (const std::invalid_argument&) { threw_mean = true; }
	try { (void)rms(empty); }
	catch (const std::invalid_argument&) { threw_rms = true; }
	REQUIRE(threw_p2p);
	REQUIRE(threw_mean);
	REQUIRE(threw_rms);
	std::cout << "  empty_segment_throws: passed\n";
}

void test_single_sample() {
	std::vector<double> one = {3.5};
	std::span<const double> seg{one};
	REQUIRE(approx(peak_to_peak(seg), 0.0, 1e-12));
	REQUIRE(approx(mean(seg), 3.5, 1e-12));
	REQUIRE(approx(rms(seg), 3.5, 1e-12));   // sqrt(3.5^2) = 3.5
	REQUIRE(std::isnan(rise_time_samples(seg)));
	REQUIRE(std::isnan(fall_time_samples(seg)));
	REQUIRE(std::isnan(period_samples(seg)));
	REQUIRE(std::isnan(frequency_hz(seg, 1.0e6)));
	std::cout << "  single_sample: passed\n";
}

void test_all_equal_samples() {
	std::vector<double> flat(64, 1.5);
	std::span<const double> seg{flat};
	REQUIRE(approx(peak_to_peak(seg), 0.0, 1e-12));
	REQUIRE(approx(mean(seg), 1.5, 1e-12));
	REQUIRE(approx(rms(seg), 1.5, 1e-12));
	// No range -> rise/fall return NaN; no rising crossings -> period NaN.
	REQUIRE(std::isnan(rise_time_samples(seg)));
	REQUIRE(std::isnan(fall_time_samples(seg)));
	REQUIRE(std::isnan(period_samples(seg)));
	std::cout << "  all_equal_samples: passed\n";
}

void test_monotonic_no_falling_edge() {
	// Pure rising ramp: rise_time well-defined, fall_time should be NaN.
	std::vector<double> ramp(101);
	for (std::size_t n = 0; n <= 100; ++n)
		ramp[n] = static_cast<double>(n) / 100.0;
	std::span<const double> seg{ramp};
	REQUIRE(!std::isnan(rise_time_samples(seg)));
	REQUIRE(std::isnan(fall_time_samples(seg)));
	std::cout << "  monotonic_no_falling_edge: passed\n";
}

void test_invalid_thresholds_throw() {
	std::vector<double> dummy(10, 0.0);
	std::span<const double> seg{dummy};

	// Rise-time validation
	bool rise_lo = false, rise_hi = false, rise_eq = false;
	try { (void)rise_time_samples(seg, -0.1, 0.9); }
	catch (const std::invalid_argument&) { rise_lo = true; }
	try { (void)rise_time_samples(seg, 0.1, 1.1); }
	catch (const std::invalid_argument&) { rise_hi = true; }
	try { (void)rise_time_samples(seg, 0.5, 0.5); }
	catch (const std::invalid_argument&) { rise_eq = true; }
	REQUIRE(rise_lo);
	REQUIRE(rise_hi);
	REQUIRE(rise_eq);

	// Fall-time validation: same contract, must throw on the same inputs.
	bool fall_lo = false, fall_hi = false, fall_eq = false;
	try { (void)fall_time_samples(seg, -0.1, 0.9); }
	catch (const std::invalid_argument&) { fall_lo = true; }
	try { (void)fall_time_samples(seg, 0.1, 1.1); }
	catch (const std::invalid_argument&) { fall_hi = true; }
	try { (void)fall_time_samples(seg, 0.5, 0.5); }
	catch (const std::invalid_argument&) { fall_eq = true; }
	REQUIRE(fall_lo);
	REQUIRE(fall_hi);
	REQUIRE(fall_eq);
	std::cout << "  invalid_thresholds_throw: passed\n";
}

void test_invalid_sample_rate_throws() {
	std::vector<double> dummy(10, 0.0);
	std::span<const double> seg{dummy};
	bool threw = false;
	try { (void)frequency_hz(seg, 0.0); }
	catch (const std::invalid_argument&) { threw = true; }
	REQUIRE(threw);
	threw = false;
	try { (void)frequency_hz(seg, -1.0e6); }
	catch (const std::invalid_argument&) { threw = true; }
	REQUIRE(threw);
	std::cout << "  invalid_sample_rate_throws: passed\n";
}

// ============================================================================
// Mixed-precision sanity: float input still gives a reasonable measurement
// ============================================================================

void test_float_input_aggregations() {
	// Same square wave as the double test but with float samples.
	// Aggregations run in double internally, so the result should match
	// the double answer to within float epsilon scaled by N.
	std::vector<float> sq(100);
	for (std::size_t n = 0; n < 50; ++n) sq[n] = 2.0f;
	for (std::size_t n = 50; n < 100; ++n) sq[n] = -1.0f;
	std::span<const float> seg{sq};

	REQUIRE(approx(peak_to_peak(seg), 3.0, 1e-6));
	REQUIRE(approx(mean(seg), 0.5, 1e-6));
	REQUIRE(approx(rms(seg), std::sqrt(2.5), 1e-6));
	std::cout << "  float_input_aggregations: passed\n";
}

// ============================================================================
// main
// ============================================================================

int main() {
	try {
		std::cout << "test_instrument_measurements\n";

		test_square_wave_aggregations();
		test_sine_aggregations();
		test_linear_ramp_rise_time();
		test_linear_ramp_fall_time();
		test_steep_step_rise_time();
		test_steep_step_fall_time();
		test_period_and_frequency_sine();

		test_empty_segment_throws();
		test_single_sample();
		test_all_equal_samples();
		test_monotonic_no_falling_edge();
		test_invalid_thresholds_throw();
		test_invalid_sample_rate_throws();

		test_float_input_aggregations();

		std::cout << "all tests passed\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "FAILED: " << ex.what() << "\n";
		return 1;
	}
}
