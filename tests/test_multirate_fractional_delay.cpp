// test_multirate_fractional_delay.cpp: tests for the polyphase
// fractional-delay line with runtime-variable delay.
//
// This class had no unit test at all before #208, which is exactly why its
// default taps_per_phase could sit at an even value its own validator
// rejected: the demo passes an explicit odd 15, so nothing ever constructed
// it the documented way. The first test below is that regression, and the
// rest cover the behaviour a caller actually depends on.
//
// Coverage:
//   - **Default construction succeeds** (the #208 regression) and every
//     value the parameter documentation recommends is accepted
//   - Constructor validation: even / too-short taps_per_phase and L == 0
//     all throw, and they throw from the initializer list rather than
//     after a half-built object
//   - **Phase 0 is an exact passthrough**: the property the odd-length
//     constraint exists to guarantee, and the reason the fix was to move
//     the default rather than relax the check
//   - Fractional-delay accuracy across the phase grid, measured as the
//     DC group delay of the impulse response, plus the 1/L quantization
//   - Requests below the group-delay floor clamp to the floor
//   - Requests beyond max_int_delay throw
//   - reset() clears the line
//   - Precision sweep: delay accuracy as the scalar types narrow
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)` —
// never assert(), which CI strips in Release.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <cstdint>   // std::uint32_t, for the cfloat block type
#include <iostream>
#include <stdexcept>
#include <string>

#include <sw/dsp/multirate/fractional_delay.hpp>
#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/posit/posit.hpp>

using namespace sw::dsp::multirate;

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
// #208 regression: the documented defaults must construct
// ============================================================================

void test_default_construction() {
	// The whole of issue #208: taps_per_phase defaulted to 12, which
	// design_bank rejects, so this line threw for every caller who took
	// the documented defaults.
	FractionalDelay<double> fd(64);
	REQUIRE(fd.L() == 64);
	REQUIRE(fd.taps_per_phase() % 2 == 1);
	REQUIRE(fd.taps_per_phase() >= 3);
	REQUIRE(fd.num_taps() == fd.L() * fd.taps_per_phase());
	// An odd length puts the group-delay floor on an integer sample.
	const double base = fd.base_group_delay_samples();
	REQUIRE_NEAR(base, std::round(base), 1e-12);
	std::cout << "  default_construction: passed (K = "
	          << fd.taps_per_phase() << ", group delay floor " << base
	          << ")\n";
}

void test_documented_values_construct() {
	// Everything the parameter documentation recommends must actually
	// work. The pre-#208 comment named 12, 16 and 24 — all even, all
	// rejected — so this test is the doc comment's own regression.
	for (std::size_t K : {std::size_t{3}, std::size_t{11}, std::size_t{15},
	                      std::size_t{21}, std::size_t{25}}) {
		FractionalDelay<double> fd(32, K);
		REQUIRE(fd.taps_per_phase() == K);
	}
	std::cout << "  documented_values_construct: passed\n";
}

// ============================================================================
// Constructor validation
// ============================================================================

void test_even_taps_per_phase_throws() {
	for (std::size_t K : {std::size_t{4}, std::size_t{12}, std::size_t{16}}) {
		bool threw = false;
		try { FractionalDelay<double> fd(32, K); }
		catch (const std::invalid_argument&) { threw = true; }
		REQUIRE(threw);
	}
	std::cout << "  even_taps_per_phase_throws: passed\n";
}

void test_short_taps_per_phase_throws() {
	for (std::size_t K : {std::size_t{0}, std::size_t{1}, std::size_t{2}}) {
		bool threw = false;
		try { FractionalDelay<double> fd(32, K); }
		catch (const std::invalid_argument&) { threw = true; }
		REQUIRE(threw);
	}
	std::cout << "  short_taps_per_phase_throws: passed\n";
}

void test_zero_L_throws() {
	// L == 0 used to be checked in the constructor BODY, which runs only
	// after design_bank and the ring-buffer sizing have already run on
	// the bad value. The check now lives in design_bank, so it fires
	// first; this test pins that down for L == 0 combined with a
	// perfectly valid K.
	bool threw = false;
	try { FractionalDelay<double> fd(0, 11); }
	catch (const std::invalid_argument&) { threw = true; }
	REQUIRE(threw);
	std::cout << "  zero_L_throws: passed\n";
}

// ============================================================================
// Phase 0 is an exact passthrough — why K must be odd
// ============================================================================

void test_phase_zero_is_exact_passthrough() {
	// With odd K the group delay (K-1)/2 is an integer, so phase 0 is
	// sinc(k - center) sampled on the integers: one non-zero tap. A
	// request at exactly the group-delay floor is therefore unfiltered.
	//
	// This is the property that decided issue #208 in favour of moving
	// the default to an odd value rather than relaxing the validator.
	// With even K the floor is a half-integer and no request lands on an
	// unfiltered tap at all.
	for (std::size_t K : {std::size_t{11}, std::size_t{15}, std::size_t{21}}) {
		FractionalDelay<double> fd(64, K);
		const double base = fd.base_group_delay_samples();
		const int    lat  = static_cast<int>(base);
		const int    impulse_at = 20;

		double worst = 0.0;
		for (int n = 0; n < 200; ++n) {
			const double x = (n == impulse_at) ? 1.0 : 0.0;
			const double y = fd.delay(x, base);
			const double want = (n == impulse_at + lat) ? 1.0 : 0.0;
			worst = std::max(worst, std::abs(y - want));
		}
		std::cout << "    K = " << K << ": worst phase-0 error = "
		          << worst << "\n";
		REQUIRE(worst < 1e-12);
	}
	std::cout << "  phase_zero_is_exact_passthrough: passed\n";
}

// ============================================================================
// Delay accuracy
// ============================================================================

// Measure the delay a filter actually applies, as the DC group delay of its
// impulse response: sum k*h[k] / sum h[k].
//
// That first moment IS -dphi/domega at omega = 0 for any real FIR, so this is
// the group delay itself rather than an estimate of it. The obvious
// alternative — correlating a delayed tone against the input — cannot be used
// here: a sinusoid's autocorrelation is periodic, so the measurement is
// ambiguous modulo the tone's period, and at any frequency low enough to sit
// inside the passband that period is longer than the delays under test.
// Broadband noise removes the ambiguity but leaves a correlation peak too
// broad for sub-sample refinement (measured, ~7e-02 samples of bias against
// 6e-04 for the moment).
template <typename Filter>
double measured_group_delay(Filter& fd, double requested) {
	const int span = 400, impulse_at = 5;
	double m1 = 0.0, m0 = 0.0;
	for (int n = 0; n < span; ++n) {
		const double y = static_cast<double>(
			fd.delay(static_cast<typename Filter::sample_scalar>(
				n == impulse_at ? 1.0 : 0.0), requested));
		m1 += static_cast<double>(n - impulse_at) * y;
		m0 += y;
	}
	if (!(std::abs(m0) > 1e-12))
		throw std::runtime_error("measured_group_delay: impulse response "
		                         "has no DC gain");
	return m1 / m0;
}

void test_delay_accuracy() {
	const std::size_t L = 64, K = 15;
	FractionalDelay<double> probe(L, K);
	const double base = probe.base_group_delay_samples();

	// Sweep the phase grid: quarter-sample steps on top of the floor, an
	// exact 1/L step, and a couple of integer-shift levels so the
	// int_shift path is exercised too.
	double worst = 0.0;
	for (double extra : {0.0, 1.0 / 64.0, 0.25, 0.5, 0.75, 1.0, 2.5, 7.25}) {
		FractionalDelay<double> fd(L, K);
		const double got = measured_group_delay(fd, base + extra);
		worst = std::max(worst, std::abs(got - (base + extra)));
	}
	std::cout << "    worst group-delay error over the phase grid: "
	          << worst << " samples\n";
	// Well inside the 1/L = 0.0156 sample delay grid, so what is left is
	// the window's effect on the pulse shape, not a placement error.
	REQUIRE(worst < 5e-3);
	std::cout << "  delay_accuracy: passed\n";
}

void test_delay_is_quantized_to_the_phase_grid() {
	// The class rounds a request to the nearest 1/L. A request halfway
	// between two phases must therefore land on one of them, not
	// somewhere in between — this is what makes L the delay resolution.
	const std::size_t L = 16, K = 11;
	FractionalDelay<double> fd(L, K);
	const double base = fd.base_group_delay_samples();
	const double step = 1.0 / static_cast<double>(L);

	const double got = measured_group_delay(fd, base + 0.4 * step);
	const double nearest = base + std::round(0.4) * step;   // == base
	REQUIRE_NEAR(got, nearest, 5e-3);
	std::cout << "  delay_is_quantized_to_the_phase_grid: passed\n";
}

// ============================================================================
// Boundary behaviour
// ============================================================================

void test_below_floor_clamps() {
	// A filter cannot reconstruct samples from the future, so requests
	// below the group-delay floor round up to it rather than throwing.
	// Documented behaviour — pin it, because silently clamping is the
	// kind of thing a refactor turns into a throw.
	const std::size_t L = 32, K = 11;
	FractionalDelay<double> at_floor(L, K), below(L, K), negative(L, K);
	const double base = at_floor.base_group_delay_samples();

	for (int n = 0; n < 100; ++n) {
		const double x = (n == 10) ? 1.0 : 0.0;
		const double a = at_floor.delay(x, base);
		const double b = below.delay(x, 0.0);
		const double c = negative.delay(x, -5.0);
		REQUIRE_NEAR(a, b, 1e-15);
		REQUIRE_NEAR(a, c, 1e-15);
	}
	std::cout << "  below_floor_clamps: passed\n";
}

void test_beyond_max_int_delay_throws() {
	const std::size_t L = 32, K = 11, max_int = 4;
	FractionalDelay<double> fd(L, K, max_int);
	const double base = fd.base_group_delay_samples();

	// Inside the window is fine...
	fd.delay(1.0, base + static_cast<double>(max_int));
	// ...one integer step beyond it is not: the ring no longer holds the
	// history the phase filter would read.
	bool threw = false;
	try { fd.delay(1.0, base + static_cast<double>(max_int) + 1.0); }
	catch (const std::runtime_error&) { threw = true; }
	REQUIRE(threw);
	std::cout << "  beyond_max_int_delay_throws: passed\n";
}

void test_reset_clears() {
	FractionalDelay<double> fd(32, 11);
	const double base = fd.base_group_delay_samples();
	for (int n = 0; n < 50; ++n) fd.delay(1.0, base);
	fd.reset();
	// After a reset the line is all zeros, so the first outputs must be
	// zero until the new input has propagated through the K taps.
	REQUIRE_NEAR(fd.delay(0.0, base), 0.0, 1e-15);
	std::cout << "  reset_clears: passed\n";
}

// ============================================================================
// Precision sweep
// ============================================================================

template <typename T>
double delay_error_for(double base, double extra, std::size_t L, std::size_t K) {
	FractionalDelay<double, T, T> fd(L, K);
	return std::abs(measured_group_delay(fd, base + extra) - (base + extra));
}

void test_precision_sweep() {
	using posit32  = sw::universal::posit<32, 2>;
	using posit16  = sw::universal::posit<16, 2>;
	using cfloat32 = sw::universal::cfloat<32, 8, std::uint32_t, true, false, false>;

	const std::size_t L = 64, K = 15;
	const double extra = 0.375;

	FractionalDelay<double> probe(L, K);
	const double base = probe.base_group_delay_samples();

	const double e_double = delay_error_for<double>  (base, extra, L, K);
	const double e_float  = delay_error_for<float>   (base, extra, L, K);
	const double e_p32    = delay_error_for<posit32> (base, extra, L, K);
	const double e_cf32   = delay_error_for<cfloat32>(base, extra, L, K);
	const double e_p16    = delay_error_for<posit16> (base, extra, L, K);

	std::cout << "    group-delay error vs streaming precision (1/L = "
	          << (1.0 / static_cast<double>(L)) << " sample grid):\n";
	std::cout << "      double  : " << e_double << "\n";
	std::cout << "      float   : " << e_float  << "\n";
	std::cout << "      posit32 : " << e_p32    << "\n";
	std::cout << "      cfloat32: " << e_cf32   << "\n";
	std::cout << "      posit16 : " << e_p16    << "\n";

	// Every type should land well inside the delay grid. This filter is a
	// windowed sinc with no feedback, so there is no accumulator to drift
	// and no threshold to fall off — unlike the SDR synchronization loops,
	// narrowing here degrades smoothly. A defensive bound, not a tight one.
	REQUIRE(e_double < 5e-3);
	REQUIRE(e_float  < 5e-3);
	REQUIRE(e_p32    < 5e-3);
	REQUIRE(e_cf32   < 5e-3);
	REQUIRE(e_p16    < 5e-2);
	std::cout << "  precision_sweep: passed\n";
}

// ============================================================================
// main
// ============================================================================

int main() {
	try {
		std::cout << "test_multirate_fractional_delay\n";

		test_default_construction();
		test_documented_values_construct();

		test_even_taps_per_phase_throws();
		test_short_taps_per_phase_throws();
		test_zero_L_throws();

		test_phase_zero_is_exact_passthrough();
		test_delay_accuracy();
		test_delay_is_quantized_to_the_phase_grid();

		test_below_floor_clamps();
		test_beyond_max_int_delay_throws();
		test_reset_clears();

		test_precision_sweep();

		std::cout << "all tests passed\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "FAILED: " << ex.what() << "\n";
		return 1;
	}
}
