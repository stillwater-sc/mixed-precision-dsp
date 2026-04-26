// test_instrument_calibration.cpp: tests for the front-end calibration /
// equalization framework.
//
// Coverage:
//   - CalibrationProfile: tabulated-point exact match, linear interpolation
//     between points, clamping below freq_min and above freq_max,
//     monotonic-frequency validation, length-mismatch validation, CSV
//     round-trip from a small file under tests/data/
//   - EqualizerFilter: synthetic-profile flattening test (design EQ
//     against a known sinusoidal-magnitude profile, verify the equalized
//     response is flat to within target dB across the calibrated band)
//   - Precision sweep: characterize equalization quality across the
//     three-scalar matrix (double / float / posit<32,2>) — the
//     intersection that all the library's number types support today
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)`.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <sw/dsp/instrument/calibration.hpp>
#include <sw/universal/number/posit/posit.hpp>

using namespace sw::dsp::instrument;

#define REQUIRE(cond) \
	do { if (!(cond)) throw std::runtime_error( \
		std::string("test failed: ") + #cond + \
		" at " __FILE__ ":" + std::to_string(__LINE__)); } while (0)

#define REQUIRE_NEAR(a, b, tol) \
	do { const double aa = (a), bb = (b), tt = (tol); \
		if (std::abs(aa - bb) > tt) \
			throw std::runtime_error(std::string("test failed: |") + \
				#a + " - " + #b + "| = " + std::to_string(std::abs(aa-bb)) + \
				" > " + std::to_string(tt) + " at " __FILE__ ":" + \
				std::to_string(__LINE__)); } while (0)

// ============================================================================
// CalibrationProfile — tabulated and interpolated lookups
// ============================================================================

void test_profile_exact_points() {
	CalibrationProfile cal({1.0, 10.0, 100.0},
	                                {0.0, -3.0, -10.0},
	                                {0.0, -0.5, -1.0});
	REQUIRE_NEAR(cal.gain_dB(1.0),    0.0, 1e-12);
	REQUIRE_NEAR(cal.gain_dB(10.0),  -3.0, 1e-12);
	REQUIRE_NEAR(cal.gain_dB(100.0),-10.0, 1e-12);
	REQUIRE_NEAR(cal.phase_rad(10.0),-0.5, 1e-12);
	std::cout << "  profile_exact_points: passed\n";
}

void test_profile_linear_interpolation() {
	CalibrationProfile cal({0.0, 100.0}, {0.0, -10.0}, {0.0, -1.0});
	// Halfway: -5 dB / -0.5 rad
	REQUIRE_NEAR(cal.gain_dB(50.0),  -5.0, 1e-12);
	REQUIRE_NEAR(cal.phase_rad(50.0),-0.5, 1e-12);
	// Quarter: -2.5 dB / -0.25 rad
	REQUIRE_NEAR(cal.gain_dB(25.0),  -2.5, 1e-12);
	REQUIRE_NEAR(cal.phase_rad(25.0),-0.25,1e-12);
	std::cout << "  profile_linear_interpolation: passed\n";
}

void test_profile_endpoint_clamping() {
	CalibrationProfile cal({10.0, 100.0}, {-1.0, -5.0}, {0.0, -2.0});
	// Below the lowest tabulated point — clamps to first
	REQUIRE_NEAR(cal.gain_dB(1.0),  -1.0, 1e-12);
	REQUIRE_NEAR(cal.phase_rad(1.0), 0.0, 1e-12);
	// Above the highest — clamps to last
	REQUIRE_NEAR(cal.gain_dB(1000.0), -5.0, 1e-12);
	REQUIRE_NEAR(cal.phase_rad(1000.0),-2.0, 1e-12);
	std::cout << "  profile_endpoint_clamping: passed\n";
}

void test_profile_validation() {
	bool threw;

	// Mismatched lengths
	threw = false;
	try { CalibrationProfile({1.0, 2.0}, {0.0}, {0.0, 0.0}); }
	catch (const std::invalid_argument&) { threw = true; }
	REQUIRE(threw);

	// Too few points
	threw = false;
	try { CalibrationProfile({1.0}, {0.0}, {0.0}); }
	catch (const std::invalid_argument&) { threw = true; }
	REQUIRE(threw);

	// Non-monotonic frequencies
	threw = false;
	try { CalibrationProfile({1.0, 5.0, 3.0}, {0,0,0}, {0,0,0}); }
	catch (const std::invalid_argument&) { threw = true; }
	REQUIRE(threw);

	std::cout << "  profile_validation: passed\n";
}

void test_profile_csv_round_trip() {
	const std::string path = std::string(TESTS_DATA_DIR) +
	                          "/example_frontend_cal.csv";
	auto cal = CalibrationProfile::from_csv(path);
	REQUIRE(cal.size() == 6);
	REQUIRE_NEAR(cal.freq_min(),     1000.0, 1e-9);
	REQUIRE_NEAR(cal.freq_max(), 50000000.0, 1e-9);
	REQUIRE_NEAR(cal.gain_dB(1000.0),    0.0, 1e-9);
	REQUIRE_NEAR(cal.gain_dB(10000000.0),-3.0,1e-9);
	std::cout << "  profile_csv_round_trip: passed\n";
}

void test_profile_csv_missing_file_throws() {
	const std::string path = std::string(TESTS_DATA_DIR) +
	                          "/this_file_does_not_exist.csv";
	bool threw = false;
	try {
		(void)CalibrationProfile::from_csv(path);
	} catch (const std::invalid_argument&) { threw = true; }
	REQUIRE(threw);
	std::cout << "  profile_csv_missing_file_throws: passed\n";
}

// ============================================================================
// EqualizerFilter — synthetic-profile flattening test
// ============================================================================

// Helper: measure the magnitude response of an EqualizerFilter at a given
// frequency by feeding a tone in and looking at the steady-state output.
template <class CoeffScalar>
double measure_response_dB(EqualizerFilter<CoeffScalar, CoeffScalar, CoeffScalar>& eq,
                            double freq_hz,
                            double sample_rate_hz,
                            std::size_t num_samples = 4096) {
	const double pi = std::numbers::pi_v<double>;
	double max_out = 0.0;
	// Skip the first num_taps samples so the filter's transient settles
	const std::size_t skip = eq.num_taps();
	for (std::size_t n = 0; n < num_samples; ++n) {
		const double phase = 2.0 * pi * freq_hz * n / sample_rate_hz;
		const auto x = static_cast<CoeffScalar>(std::cos(phase));
		const auto y = eq.process(x);
		if (n >= skip) {
			const double yd = static_cast<double>(y);
			if (std::abs(yd) > max_out) max_out = std::abs(yd);
		}
	}
	// Input amplitude is 1.0; output max is the magnitude response
	return 20.0 * std::log10(std::max(max_out, 1e-300));
}

void test_eq_flattens_synthetic_profile() {
	// Synthetic profile: a slow sinusoidal magnitude variation across the
	// band and a small linear-phase ramp. The equalizer should flatten
	// this to within a few dB across the calibrated band.
	const double fs = 1e6;
	const std::size_t N = 32;  // profile points
	std::vector<double> freqs(N), gains(N), phases(N);
	for (std::size_t i = 0; i < N; ++i) {
		freqs[i]  = i * fs / (2.0 * (N - 1));         // 0 .. fs/2
		// Magnitude varies between -3 and +3 dB over the band
		gains[i]  = 3.0 * std::sin(2.0 * std::numbers::pi_v<double> *
		                            i / (N - 1));
		phases[i] = 0.1 * static_cast<double>(i) / (N - 1);
	}
	CalibrationProfile profile(freqs, gains, phases);
	EqualizerFilter<double, double, double> eq(profile, /*num_taps=*/65, fs);

	// Measure the equalizer's response at a handful of test frequencies
	// well inside the calibrated band (away from DC and Nyquist where
	// frequency-sampling design has the most ringing). Each measured
	// response should approximately cancel the profile's gain at that
	// frequency.
	const std::array<double, 5> test_freqs = {
		fs * 0.05, fs * 0.10, fs * 0.20, fs * 0.30, fs * 0.40};
	for (double f : test_freqs) {
		const double profile_dB = profile.gain_dB(f);
		const double eq_dB      = measure_response_dB(eq, f, fs);
		const double net_dB     = profile_dB + eq_dB;
		// Within the "interior" of the calibrated band, the equalizer
		// should flatten to within a few dB. Frequency-sampling design
		// without a richer optimization will leave some ripple
		// between sample points, but ±2 dB is achievable.
		if (std::abs(net_dB) > 2.0)
			throw std::runtime_error(
				"test failed: net response at " + std::to_string(f) +
				" Hz = " + std::to_string(net_dB) +
				" dB (expected within ±2 dB of flat)");
	}
	std::cout << "  eq_flattens_synthetic_profile: passed\n";
}

void test_eq_clamps_inverse_at_deep_null() {
	// A profile with a -80 dB null at one frequency. With max_gain_dB=20,
	// the equalizer should clamp at +20 dB instead of trying to amplify
	// by 10000×. We check that the resulting tap magnitudes don't blow up
	// — they should be O(1) or smaller after windowing.
	std::vector<double> freqs = {0.0, 1.0e5, 5.0e5};
	std::vector<double> gains = {0.0, -80.0, 0.0};
	std::vector<double> phases = {0.0, 0.0, 0.0};
	CalibrationProfile profile(freqs, gains, phases);
	EqualizerFilter<double, double, double> eq(profile,
	                                            /*num_taps=*/65,
	                                            /*fs=*/1e6,
	                                            /*max_gain_dB=*/20.0);

	// Stream a unit-amplitude tone away from the null. Output should not
	// be wildly larger than the input. Skip the FIR's transient (one
	// num_taps span) so we measure steady-state amplitude.
	const double      pi   = std::numbers::pi_v<double>;
	const std::size_t skip = eq.num_taps();
	double max_out = 0.0;
	for (std::size_t n = 0; n < 1024; ++n) {
		const double x = std::cos(2.0 * pi * 5.0e4 * n / 1e6);
		const double y = eq.process(x);
		if (n > skip && std::abs(y) > max_out) max_out = std::abs(y);
	}
	// With max_gain_dB=20, the equalizer's response near the null is
	// bounded by 10× input. Including FIR ripple and overshoot, allow up
	// to ~20× as a sanity bound; the point is that it's not infinite.
	REQUIRE(max_out < 20.0);
	std::cout << "  eq_clamps_inverse_at_deep_null: passed (max_out=" <<
	             max_out << ")\n";
}

// ============================================================================
// Precision sweep — characterize cross-precision equalization quality
// ============================================================================

template <class T>
double measure_eq_error_dB() {
	// Design an equalizer for a known-good profile at type T, run a tone
	// through it, and measure how much the output deviates from the
	// expected unit gain (in dB). The profile's gain at the test
	// frequency is +1 dB, so a perfect equalizer would output 0 dB
	// (-1 dB equalizer × +1 dB profile = 0 dB net). We measure just the
	// equalizer here against the expected unit input gain.
	const double fs = 1e6;
	std::vector<double> freqs  = {0.0, fs / 2.0};
	std::vector<double> gains  = {1.0, 1.0};   // flat +1 dB across band
	std::vector<double> phases = {0.0, 0.0};
	CalibrationProfile profile(freqs, gains, phases);
	EqualizerFilter<T, T, T> eq(profile, /*num_taps=*/65, fs);

	// Equalizer's expected gain: -1 dB everywhere
	const double expected_dB = -1.0;
	const double measured_dB = measure_response_dB(eq, fs * 0.25, fs);
	return std::abs(measured_dB - expected_dB);
}

void test_precision_sweep() {
	const double err_double = measure_eq_error_dB<double>();
	const double err_float  = measure_eq_error_dB<float>();
	const double err_p32    = measure_eq_error_dB<sw::universal::posit<32, 2>>();

	std::cout << "  precision sweep: equalizer error vs target -1 dB\n";
	std::cout << "    double:        " << err_double << " dB\n";
	std::cout << "    float:         " << err_float  << " dB\n";
	std::cout << "    posit<32,2>:   " << err_p32    << " dB\n";

	// Frequency-sampling design has intrinsic ripple between sample
	// points, so even the double reference deviates by a fraction of a
	// dB. The cross-precision degradation we care about is whether float
	// / posit32 measurably worsen this. Loose bounds:
	REQUIRE(err_double < 1.0);   // double should be within ±1 dB of target
	REQUIRE(err_float  < 1.5);   // float adds modest streaming-arith noise
	REQUIRE(err_p32    < 1.5);   // posit32 in same neighborhood

	std::cout << "  precision_sweep: passed\n";
}

// ============================================================================
// main
// ============================================================================

int main() {
	try {
		std::cout << "test_instrument_calibration\n";

		test_profile_exact_points();
		test_profile_linear_interpolation();
		test_profile_endpoint_clamping();
		test_profile_validation();
		test_profile_csv_round_trip();
		test_profile_csv_missing_file_throws();

		test_eq_flattens_synthetic_profile();
		test_eq_clamps_inverse_at_deep_null();

		test_precision_sweep();

		std::cout << "all tests passed\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "FAILED: " << ex.what() << "\n";
		return 1;
	}
}
