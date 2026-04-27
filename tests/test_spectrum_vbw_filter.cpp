// test_spectrum_vbw_filter.cpp: tests for the spectrum-analyzer VBW
// (video bandwidth) post-detector LPF.
//
// Coverage:
//   - DC gain = 1: long constant input settles to the constant
//   - Cutoff accuracy: |H(fc)| ~= 1/sqrt(2) (-3 dB) within 5% at
//     three representative cutoffs (fc = fs/100, fs/10, fs/1000)
//   - Stability at low cutoffs (fc = fs/1000): output stays bounded
//     and converges to DC for a constant input
//   - Bumpless retune: state preserved across set_cutoff(); output
//     continuous (no discontinuity at the retune sample)
//   - reset() clears state
//   - Validation: zero/negative cutoff, zero/negative fs, fc above
//     Nyquist all throw
//   - Length mismatch in process_block(input, output) throws
//   - Mixed-precision sanity: float SampleScalar produces a settled
//     output close to the double reference
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)`.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <sw/dsp/spectrum/vbw_filter.hpp>

using namespace sw::dsp::spectrum;
using F = VBWFilter<double>;

#define REQUIRE(cond) \
	do { if (!(cond)) throw std::runtime_error( \
		std::string("test failed: ") + #cond + \
		" at " __FILE__ ":" + std::to_string(__LINE__)); } while (0)

static bool approx(double a, double b, double tol) {
	return std::abs(a - b) <= tol;
}

// ============================================================================
// DC gain == 1
// ============================================================================

void test_dc_gain_unity() {
	// Long run of constant input. After enough sweeps to settle the
	// IIR (many time constants), the output equals the input.
	F vbw(/*fc=*/1000.0, /*fs=*/100000.0);   // fc = fs/100
	const double constant = 7.5;
	double y = 0.0;
	for (int i = 0; i < 5000; ++i) y = vbw.process(constant);
	REQUIRE(approx(y, constant, 1e-3));
	std::cout << "  dc_gain_unity: passed (settled to " << y << ")\n";
}

// ============================================================================
// Cutoff accuracy: |H(fc)| ~= 1/sqrt(2) (-3 dB) within 5%
// ============================================================================

// Drive the filter with a sine at frequency `f_in_hz`, run for enough
// samples to reach steady state, then measure the peak amplitude over
// the last few cycles. Compare to the input amplitude to get |H|.
static double measure_magnitude(F& vbw, double f_in_hz, double fs) {
	const double pi = std::numbers::pi_v<double>;
	const double amp = 1.0;
	const std::size_t cycles_warmup = 50;
	const std::size_t cycles_measure = 20;
	const double samples_per_cycle = fs / f_in_hz;
	const std::size_t n_warmup =
		static_cast<std::size_t>(cycles_warmup * samples_per_cycle);
	const std::size_t n_measure =
		static_cast<std::size_t>(cycles_measure * samples_per_cycle);

	for (std::size_t n = 0; n < n_warmup; ++n) {
		const double x = amp * std::sin(2.0 * pi * f_in_hz
		                                 * static_cast<double>(n) / fs);
		(void)vbw.process(x);
	}
	double peak = 0.0;
	for (std::size_t n = 0; n < n_measure; ++n) {
		const double t = static_cast<double>(n_warmup + n) / fs;
		const double x = amp * std::sin(2.0 * pi * f_in_hz * t);
		const double y = vbw.process(x);
		if (std::abs(y) > peak) peak = std::abs(y);
	}
	return peak / amp;
}

void test_cutoff_minus3db() {
	// Three representative cutoffs spanning the typical VBW range.
	// At each, |H(fc)| should be 1/sqrt(2) ~= 0.7071 within 5%.
	const double sqrt_half = 1.0 / std::sqrt(2.0);
	struct Case { double fs; double fc; const char* name; };
	const Case cases[] = {
		{1.0e6, 1.0e3, "fs=1MHz fc=1kHz"},     // fc = fs/1000
		{1.0e6, 1.0e4, "fs=1MHz fc=10kHz"},    // fc = fs/100
		{1.0e6, 1.0e5, "fs=1MHz fc=100kHz"},   // fc = fs/10
	};
	for (const auto& c : cases) {
		F vbw(c.fc, c.fs);
		const double mag = measure_magnitude(vbw, c.fc, c.fs);
		const double err_pct = std::abs(mag - sqrt_half) / sqrt_half * 100.0;
		if (!(err_pct < 5.0))
			throw std::runtime_error(
				std::string("cutoff accuracy out of spec: ") + c.name
				+ " |H(fc)|=" + std::to_string(mag)
				+ " err=" + std::to_string(err_pct) + "%");
		std::cout << "  cutoff_minus3db [" << c.name << "]: |H(fc)|="
		          << mag << " err=" << err_pct << "%\n";
	}
}

// ============================================================================
// Stability at low cutoffs (fc = fs/1000)
// ============================================================================

void test_stable_at_low_cutoff() {
	F vbw(/*fc=*/1.0, /*fs=*/1000.0);   // fc = fs/1000
	// Push a long bounded input and verify the output stays bounded.
	double max_abs = 0.0;
	for (int i = 0; i < 100000; ++i) {
		const double x = (i % 2 == 0) ? 1.0 : -1.0;   // square wave at fs/2
		const double y = vbw.process(x);
		max_abs = std::max(max_abs, std::abs(y));
		REQUIRE(std::isfinite(y));
	}
	// At fc = fs/1000 the filter heavily attenuates fs/2 signal — the
	// output should be much smaller than 1 in magnitude.
	REQUIRE(max_abs < 0.1);
	std::cout << "  stable_at_low_cutoff: passed (max|y|=" << max_abs << ")\n";
}

// ============================================================================
// Bumpless retune
// ============================================================================

void test_bumpless_retune() {
	// "Bumpless" means set_cutoff() preserves y_prev_ across the
	// coefficient change — the IIR state isn't reset. Pushing x=0
	// after a retune is a clean test: with alpha_new and x=0, the
	// output is simply (1 - alpha_new) * y_prev. So if y_prev was
	// preserved, y_after equals (1 - alpha_new) times the immediately-
	// pre-retune output.
	F vbw(/*fc=*/1000.0, /*fs=*/100000.0);
	double pre_retune = 0.0;
	for (int i = 0; i < 5; ++i) pre_retune = vbw.process(1.0);   // partial settle
	REQUIRE(pre_retune > 0.0);   // sanity: filter is responding

	const double new_fc = 500.0;
	const double new_fs = 100000.0;
	// Recompute alpha for the post-retune cutoff using the same formula
	// the filter uses internally. This couples the test to the design
	// formula, but that's the right coupling — bumpless is precisely
	// the property that the state passes through the formula change
	// untouched.
	const double pi = std::numbers::pi_v<double>;
	const double alpha_new = 1.0 - std::exp(-2.0 * pi * new_fc / new_fs);
	vbw.set_cutoff(new_fc);
	const double y_after = vbw.process(0.0);
	const double expected = (1.0 - alpha_new) * pre_retune;
	// Tolerance loose enough to absorb the 1e-8 denormal AC injection.
	REQUIRE(approx(y_after, expected, 1e-7));
	std::cout << "  bumpless_retune: passed (y_after=" << y_after
	          << " vs expected " << expected
	          << " from preserved y_prev=" << pre_retune << ")\n";
}

// ============================================================================
// reset() clears state
// ============================================================================

void test_reset_clears_state() {
	F vbw(1000.0, 100000.0);
	for (int i = 0; i < 100; ++i) vbw.process(10.0);   // build up state
	vbw.reset();
	// Push a zero sample. Without state, output should be 0 plus the
	// 1e-8 denormal AC injection (no-op for posit/fixpnt; ~1e-8 on
	// IEEE float/double). Tolerance loose enough to absorb that.
	const double y0 = vbw.process(0.0);
	REQUIRE(approx(y0, 0.0, 1e-7));
	std::cout << "  reset_clears_state: passed\n";
}

// ============================================================================
// Validation
// ============================================================================

void test_validation() {
	bool t1=false, t2=false, t3=false, t4=false, t5=false, t6=false,
	     t7=false, t8=false, t9=false, t10=false;

	try { F(0.0, 1e6); } catch (const std::invalid_argument&) { t1 = true; }
	REQUIRE(t1);
	try { F(-1.0, 1e6); } catch (const std::invalid_argument&) { t2 = true; }
	REQUIRE(t2);
	try { F(1000.0, 0.0); } catch (const std::invalid_argument&) { t3 = true; }
	REQUIRE(t3);
	try { F(1000.0, -1.0); } catch (const std::invalid_argument&) { t4 = true; }
	REQUIRE(t4);
	// fc above Nyquist (fs/2)
	try { F(/*fc=*/600000.0, /*fs=*/1e6); } catch (const std::invalid_argument&) { t5 = true; }
	REQUIRE(t5);
	// set_cutoff with invalid value also throws
	F vbw(1000.0, 1e6);
	try { vbw.set_cutoff(-100.0); } catch (const std::invalid_argument&) { t6 = true; }
	REQUIRE(t6);

	// Non-finite inputs: NaN and +inf must both throw. NaN is caught
	// by `> 0.0` (NaN comparisons are false); +inf is the one that
	// would otherwise sneak past `> 0.0` and produce a useless
	// "y stuck at y_prev" filter (alpha = 1 - exp(-x/inf) = 0).
	const double NaN = std::numeric_limits<double>::quiet_NaN();
	const double INF = std::numeric_limits<double>::infinity();
	try { F(NaN, 1e6); } catch (const std::invalid_argument&) { t7 = true; }
	REQUIRE(t7);
	try { F(INF, 1e6); } catch (const std::invalid_argument&) { t8 = true; }
	REQUIRE(t8);
	try { F(1000.0, NaN); } catch (const std::invalid_argument&) { t9 = true; }
	REQUIRE(t9);
	try { F(1000.0, INF); } catch (const std::invalid_argument&) { t10 = true; }
	REQUIRE(t10);
	std::cout << "  validation: passed\n";
}

void test_length_mismatch_throws() {
	F vbw(1000.0, 100000.0);
	std::array<double, 8> in{};
	std::array<double, 7> out{};
	bool threw = false;
	try {
		vbw.process_block(std::span<const double>{in},
		                   std::span<double>{out});
	} catch (const std::invalid_argument&) { threw = true; }
	REQUIRE(threw);
	std::cout << "  length_mismatch_throws: passed\n";
}

// ============================================================================
// Mixed-precision sanity
// ============================================================================

void test_float_settles_close_to_double() {
	// Three instantiations:
	//   - all-float        VBWFilter<float, float, float>
	//   - all-double       VBWFilter<double, double, double>
	//   - mixed-precision  VBWFilter<double, double, float>
	//                      (high-precision coefficients + state, float
	//                      streaming I/O — the FPGA-pragmatic mix that
	//                      the scope_demo's eq_float_storage_fx16 plan
	//                      uses for its calibration FIR)
	// All three should settle to ~7.5 for a constant input. Float
	// drift bounded by float epsilon times ~5000 iterations of leaky
	// integration is well under 1e-3.
	using FF = VBWFilter<float,  float,  float>;
	using FD = VBWFilter<double, double, double>;
	using FM = VBWFilter<double, double, float>;
	FF vbw_f(1000.0, 100000.0);
	FD vbw_d(1000.0, 100000.0);
	FM vbw_m(1000.0, 100000.0);
	float  y_f = 0.0f;
	double y_d = 0.0;
	float  y_m = 0.0f;
	for (int i = 0; i < 5000; ++i) {
		y_f = vbw_f.process(7.5f);
		y_d = vbw_d.process(7.5);
		y_m = vbw_m.process(7.5f);
	}
	REQUIRE(approx(static_cast<double>(y_f), y_d, 1e-3));
	REQUIRE(approx(static_cast<double>(y_m), y_d, 1e-3));
	std::cout << "  float_settles_close_to_double: passed (float="
	          << y_f << " mixed=" << y_m << " double=" << y_d << ")\n";
}

// ============================================================================
// main
// ============================================================================

int main() {
	try {
		std::cout << "test_spectrum_vbw_filter\n";

		test_dc_gain_unity();
		test_cutoff_minus3db();
		test_stable_at_low_cutoff();
		test_bumpless_retune();
		test_reset_clears_state();

		test_validation();
		test_length_mismatch_throws();

		test_float_settles_close_to_double();

		std::cout << "all tests passed\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "FAILED: " << ex.what() << "\n";
		return 1;
	}
}
