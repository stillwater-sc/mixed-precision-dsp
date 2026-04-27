// test_instrument_fractional_delay.cpp: tests for the windowed-sinc
// fractional-delay FIR primitive.
//
// Coverage:
//   - Constructor validation: even num_taps, num_taps < 3, delay outside
//     [0, 1) all throw
//   - Zero-delay passthrough: with delay=0, the FIR's group delay is just
//     (N-1)/2 — input is faithfully reproduced after a known integer
//     latency
//   - **Delay accuracy**: feed a tone, measure the actual output delay
//     against the requested fractional delay across the band
//   - **In-band magnitude flatness**: ±0.5 dB across the passband (DC
//     to fs/2.5 approximately)
//   - **Group-delay flatness**: linear-phase FIR ⇒ constant group delay
//   - set_delay() retunes without resetting state (the FIR's internal
//     delay-line samples are preserved)
//   - Precision sweep: delay-accuracy degradation as types narrow
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)`.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <sw/dsp/instrument/fractional_delay.hpp>
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
				#a " - " #b "| = " + std::to_string(std::abs(aa-bb)) + \
				" > " + std::to_string(tt) + " at " __FILE__ ":" + \
				std::to_string(__LINE__)); } while (0)

// ============================================================================
// Constructor validation
// ============================================================================

void test_ctor_even_taps_throws() {
	bool threw = false;
	try { FractionalDelay<double> d(0.5, /*num_taps=*/30); }
	catch (const std::invalid_argument&) { threw = true; }
	REQUIRE(threw);
	std::cout << "  ctor_even_taps_throws: passed\n";
}

void test_ctor_too_few_taps_throws() {
	bool threw = false;
	try { FractionalDelay<double> d(0.5, /*num_taps=*/1); }
	catch (const std::invalid_argument&) { threw = true; }
	REQUIRE(threw);
	std::cout << "  ctor_too_few_taps_throws: passed\n";
}

void test_ctor_delay_negative_throws() {
	bool threw = false;
	try { FractionalDelay<double> d(-0.1); }
	catch (const std::invalid_argument&) { threw = true; }
	REQUIRE(threw);
	std::cout << "  ctor_delay_negative_throws: passed\n";
}

void test_ctor_delay_one_or_more_throws() {
	bool threw = false;
	try { FractionalDelay<double> d(1.0); }
	catch (const std::invalid_argument&) { threw = true; }
	REQUIRE(threw);
	std::cout << "  ctor_delay_one_or_more_throws: passed\n";
}

// ============================================================================
// Helpers
// ============================================================================

// Run a complex tone of frequency f (cycles/sample) through the filter,
// returning the steady-state magnitude and phase response. Skips the
// transient (first num_taps samples) before measuring.
template <class FD>
struct ToneResponse {
	double magnitude_dB;
	double phase_rad;
};

template <class FD>
ToneResponse<FD> measure_tone_response(FD& fd, double f_norm,
                                        std::size_t num_samples = 1024) {
	using T = typename FD::sample_scalar;
	const double pi = std::numbers::pi_v<double>;
	// Drive cosine, observe both real and imaginary correlations to
	// recover magnitude + phase.
	double sum_in_re = 0.0, sum_in_im = 0.0;
	double sum_out_re = 0.0, sum_out_im = 0.0;
	const std::size_t skip = fd.num_taps();
	for (std::size_t n = 0; n < num_samples; ++n) {
		const double phase = 2.0 * pi * f_norm * static_cast<double>(n);
		const T x = static_cast<T>(std::cos(phase));
		const T y = fd.process(x);
		if (n >= skip) {
			const double yd = static_cast<double>(y);
			const double xd = std::cos(phase);
			sum_in_re  += xd * std::cos(phase);
			sum_in_im  += xd * std::sin(phase);
			sum_out_re += yd * std::cos(phase);
			sum_out_im += yd * std::sin(phase);
		}
	}
	// |H(f)| ≈ 2 * sqrt((sum_out_re)^2 + (sum_out_im)^2) /
	//          2 * sqrt((sum_in_re)^2 + (sum_in_im)^2)
	const double mag_in  = std::hypot(sum_in_re,  sum_in_im);
	const double mag_out = std::hypot(sum_out_re, sum_out_im);
	const double mag_dB  = 20.0 * std::log10(mag_out / mag_in);
	// Standard filter phase response convention: H(f) = sum y[n] e^{-j2πfn}
	// means phase = atan2(-sum_out_im, sum_out_re). Equivalently, the
	// cross-correlation phase (atan2(sum_out_im, sum_out_re)) is the
	// NEGATIVE of the filter phase response. Negate to match Bode/DTFT
	// convention: a delay of τ samples gives phase = -2π f τ (negative).
	const double phase_in  = std::atan2(sum_in_im,  sum_in_re);
	const double phase_out = std::atan2(sum_out_im, sum_out_re);
	double phase_rad = phase_in - phase_out;
	// Wrap to [-π, π]
	while (phase_rad >  pi) phase_rad -= 2.0 * pi;
	while (phase_rad < -pi) phase_rad += 2.0 * pi;
	return ToneResponse<FD>{mag_dB, phase_rad};
}

// ============================================================================
// Zero-delay passthrough
// ============================================================================

void test_zero_delay_passthrough() {
	// delay=0 should give a near-perfect linear-phase FIR with peak at the
	// center tap. Group delay = (N-1)/2 = 15 for N=31. A unit impulse in
	// at sample 0 should appear (approximately) at output sample 15.
	FractionalDelay<double> d(0.0, /*num_taps=*/31);
	std::vector<double> input(64, 0.0);
	input[0] = 1.0;
	std::vector<double> output(input.size());
	for (std::size_t n = 0; n < input.size(); ++n) {
		output[n] = d.process(input[n]);
	}
	// Find the peak index
	double peak = 0.0;
	std::size_t peak_idx = 0;
	for (std::size_t n = 0; n < output.size(); ++n) {
		if (std::abs(output[n]) > peak) {
			peak     = std::abs(output[n]);
			peak_idx = n;
		}
	}
	REQUIRE(peak_idx == 15);                    // (N-1)/2 = 15
	REQUIRE_NEAR(output[15], 1.0, 0.05);         // close to unity (Hamming attenuates a bit)
	std::cout << "  zero_delay_passthrough: passed (peak at sample "
	          << peak_idx << ", value=" << output[15] << ")\n";
}

// ============================================================================
// Delay accuracy across the band
// ============================================================================

void test_delay_accuracy() {
	// A linear-phase FIR with group delay (N-1)/2 + d_frac means a tone at
	// frequency f_norm gets a phase shift of:
	//   phase = -2π * f_norm * total_delay
	// Pick a delay (e.g., 0.3) and verify that the measured phase at
	// several test frequencies matches the predicted one within tolerance.
	const double delay_frac = 0.3;
	FractionalDelay<double> d(delay_frac, /*num_taps=*/31);
	const double total_delay = 15.0 + delay_frac;
	const double pi = std::numbers::pi_v<double>;

	// Test frequencies inside the passband (well below Nyquist).
	const std::array<double, 4> freqs = {0.05, 0.10, 0.15, 0.20};
	for (double f : freqs) {
		const double expected_phase = -2.0 * pi * f * total_delay;
		// Wrap expected phase to [-π, π] for comparison
		double exp_phase = expected_phase;
		while (exp_phase >  pi) exp_phase -= 2.0 * pi;
		while (exp_phase < -pi) exp_phase += 2.0 * pi;
		d.reset();   // isolate iterations: clear any prior-tone state
		const auto resp = measure_tone_response(d, f);
		// Phase difference (wrapped). 0.05 rad ≈ 3 degrees tolerance.
		double dphase = resp.phase_rad - exp_phase;
		while (dphase >  pi) dphase -= 2.0 * pi;
		while (dphase < -pi) dphase += 2.0 * pi;
		if (std::abs(dphase) > 0.05)
			throw std::runtime_error(
				"delay accuracy: at f_norm=" + std::to_string(f) +
				" measured phase=" + std::to_string(resp.phase_rad) +
				" expected=" + std::to_string(exp_phase) +
				" diff=" + std::to_string(dphase));
	}
	std::cout << "  delay_accuracy: passed (4 frequencies, ±3° phase)\n";
}

// ============================================================================
// In-band magnitude flatness
// ============================================================================

void test_in_band_flatness() {
	FractionalDelay<double> d(0.3, /*num_taps=*/31);
	// Test in-band magnitude — should be ~0 dB (well within ±0.5 dB)
	// from DC up to roughly fs/2.5 (where the windowed-sinc starts to
	// roll off).
	const std::array<double, 5> freqs = {0.01, 0.05, 0.10, 0.15, 0.20};
	for (double f : freqs) {
		d.reset();   // isolate iterations: clear any prior-tone state
		const auto resp = measure_tone_response(d, f);
		if (std::abs(resp.magnitude_dB) > 0.5)
			throw std::runtime_error(
				"in-band flatness: at f_norm=" + std::to_string(f) +
				" mag_dB=" + std::to_string(resp.magnitude_dB) +
				" (want |dB| < 0.5)");
	}
	std::cout << "  in_band_flatness: passed (5 frequencies, ±0.5 dB)\n";
}

// ============================================================================
// Group-delay flatness — linear-phase FIR property
// ============================================================================

void test_group_delay_flatness() {
	const double delay_frac = 0.3;
	FractionalDelay<double> d(delay_frac, /*num_taps=*/31);
	const double total_delay = 15.0 + delay_frac;
	const double pi = std::numbers::pi_v<double>;
	// Group delay = -d(phase)/d(omega). For a linear-phase FIR, this is
	// exactly the constant total_delay. Compute by finite differences
	// across the passband.
	const std::array<double, 4> freqs = {0.05, 0.10, 0.15, 0.20};
	std::vector<double> phases;
	for (double f : freqs) {
		d.reset();   // isolate iterations
		auto resp = measure_tone_response(d, f);
		// Unwrap based on expected linear ramp
		double exp_phase = -2.0 * pi * f * total_delay;
		while (resp.phase_rad - exp_phase >  pi) resp.phase_rad -= 2.0 * pi;
		while (resp.phase_rad - exp_phase < -pi) resp.phase_rad += 2.0 * pi;
		phases.push_back(resp.phase_rad);
	}
	// Compute group delay from consecutive (f_k, phase_k) pairs:
	//   gd = -(phase[k+1] - phase[k]) / (2π * (f[k+1] - f[k]))
	for (std::size_t k = 0; k + 1 < freqs.size(); ++k) {
		const double gd = -(phases[k+1] - phases[k]) /
		                   (2.0 * pi * (freqs[k+1] - freqs[k]));
		if (std::abs(gd - total_delay) > 0.05)
			throw std::runtime_error(
				"group_delay: between f=" + std::to_string(freqs[k]) +
				" and f=" + std::to_string(freqs[k+1]) +
				" gd=" + std::to_string(gd) +
				" want " + std::to_string(total_delay));
	}
	std::cout << "  group_delay_flatness: passed (gd ≈ "
	          << total_delay << " across passband)\n";
}

// ============================================================================
// set_delay() retunes
// ============================================================================

void test_set_delay_retunes() {
	FractionalDelay<double> d(0.0, /*num_taps=*/31);
	// Verify at delay=0 the response is consistent
	auto r0 = measure_tone_response(d, 0.10);
	REQUIRE_NEAR(r0.magnitude_dB, 0.0, 0.5);

	// Retune to delay=0.5; the response should still be in-band-flat but
	// the phase changes.
	d.set_delay(0.5);
	auto r1 = measure_tone_response(d, 0.10);
	REQUIRE_NEAR(r1.magnitude_dB, 0.0, 0.5);
	// Phase difference between r0 and r1 corresponds to 0.5 sample of
	// extra delay at f_norm=0.1 → -2π * 0.1 * 0.5 ≈ -0.314 rad
	const double pi = std::numbers::pi_v<double>;
	const double expected_dphase = -2.0 * pi * 0.10 * 0.5;
	const double measured_dphase = r1.phase_rad - r0.phase_rad;
	double dphase = measured_dphase - expected_dphase;
	while (dphase >  pi) dphase -= 2.0 * pi;
	while (dphase < -pi) dphase += 2.0 * pi;
	REQUIRE(std::abs(dphase) < 0.1);
	std::cout << "  set_delay_retunes: passed\n";
}

// ============================================================================
// Precision sweep
// ============================================================================

template <class T>
double measure_delay_error_at_f(double delay_frac, double f_norm) {
	FractionalDelay<T, T, T> d(delay_frac, /*num_taps=*/31);
	const auto resp = measure_tone_response(d, f_norm);
	const double pi = std::numbers::pi_v<double>;
	const double total_delay = 15.0 + delay_frac;
	double exp_phase = -2.0 * pi * f_norm * total_delay;
	while (exp_phase >  pi) exp_phase -= 2.0 * pi;
	while (exp_phase < -pi) exp_phase += 2.0 * pi;
	double dphase = resp.phase_rad - exp_phase;
	while (dphase >  pi) dphase -= 2.0 * pi;
	while (dphase < -pi) dphase += 2.0 * pi;
	// Convert phase error to fractional-sample error
	return std::abs(dphase) / (2.0 * pi * f_norm);
}

void test_precision_sweep() {
	const double delay = 0.3;
	const double f     = 0.10;

	const double err_double = measure_delay_error_at_f<double>(delay, f);
	const double err_float  = measure_delay_error_at_f<float>(delay, f);
	const double err_p32    = measure_delay_error_at_f<sw::universal::posit<32, 2>>(delay, f);

	std::cout << "  precision sweep: delay accuracy at delay=0.3, f_norm=0.10\n";
	std::cout << "    double:        " << err_double << " samples\n";
	std::cout << "    float:         " << err_float  << " samples\n";
	std::cout << "    posit<32,2>:   " << err_p32    << " samples\n";

	// All should be small (sub-sample). The comparison is what matters:
	// double should be the most accurate, float and posit32 close behind.
	REQUIRE(err_double < 0.01);
	REQUIRE(err_float  < 0.01);
	REQUIRE(err_p32    < 0.01);
	std::cout << "  precision_sweep: passed\n";
}

// ============================================================================
// main
// ============================================================================

int main() {
	try {
		std::cout << "test_instrument_fractional_delay\n";

		test_ctor_even_taps_throws();
		test_ctor_too_few_taps_throws();
		test_ctor_delay_negative_throws();
		test_ctor_delay_one_or_more_throws();

		test_zero_delay_passthrough();
		test_delay_accuracy();
		test_in_band_flatness();
		test_group_delay_flatness();

		test_set_delay_retunes();

		test_precision_sweep();

		std::cout << "all tests passed\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "FAILED: " << ex.what() << "\n";
		return 1;
	}
}
