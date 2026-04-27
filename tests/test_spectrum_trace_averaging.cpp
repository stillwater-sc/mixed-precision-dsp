// test_spectrum_trace_averaging.cpp: tests for the spectrum-analyzer
// trace-averaging modes (linear / exponential / max-hold / min-hold /
// max-hold-N).
//
// Coverage:
//   - Linear convergence: cumulative mean approaches the true mean
//     when the input is signal + zero-mean noise
//   - Exponential settling: alpha=0.1 step response settles toward the
//     new value at a known rate
//   - MaxHold preserves peaks across sweeps
//   - MinHold preserves troughs across sweeps
//   - MaxHoldN forgets a peak after the rolling window slides past it
//   - reset() clears state; sweeps_accumulated() reports correctly
//   - Validation: empty input, length mismatch, invalid alpha, invalid N
//   - Mixed-precision sanity: float input gives reasonable averages
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
#include <random>
#include <span>
#include <stdexcept>
#include <string>

#include <sw/dsp/spectrum/trace_averaging.hpp>

using namespace sw::dsp::spectrum;
using DA = TraceAverager<double>;

#define REQUIRE(cond) \
	do { if (!(cond)) throw std::runtime_error( \
		std::string("test failed: ") + #cond + \
		" at " __FILE__ ":" + std::to_string(__LINE__)); } while (0)

static bool approx(double a, double b, double tol) {
	return std::abs(a - b) <= tol;
}

// ============================================================================
// Linear: cumulative mean converges to the true signal under noise
// ============================================================================

void test_linear_converges_to_truth() {
	const std::size_t N_BINS  = 8;
	const std::size_t N_SWEEPS = 1000;
	const double truth = 5.0;

	DA avg(N_BINS, DA::Mode::Linear);
	std::mt19937 rng(0xACDC);
	std::normal_distribution<double> noise(0.0, 1.0);

	std::array<double, 8> sweep{};
	for (std::size_t s = 0; s < N_SWEEPS; ++s) {
		for (std::size_t i = 0; i < N_BINS; ++i)
			sweep[i] = truth + noise(rng);
		avg.accept_sweep(std::span<const double>{sweep});
	}
	REQUIRE(avg.sweeps_accumulated() == N_SWEEPS);
	auto out = avg.current_trace();
	for (std::size_t i = 0; i < N_BINS; ++i) {
		// stddev of the sample mean is sigma / sqrt(N) = 1/sqrt(1000) ~= 0.032,
		// so |mean - truth| should be well under 0.2 for any one bin.
		REQUIRE(approx(out[i], truth, 0.2));
	}
	std::cout << "  linear_converges_to_truth: passed (out[0]=" << out[0] << ")\n";
}

// ============================================================================
// Exponential: settles toward a new value at a known rate
// ============================================================================

void test_exponential_step_settling() {
	// alpha=0.1: after k sweeps of constant input, the running output is
	// 1 - (1-alpha)^k of the way from 0 to the input. After ~50 sweeps,
	// (1-0.1)^50 ~= 0.005, so the output should be within 0.5% of input.
	//
	// First-sweep seeding rule: y[0] = x[0] (no blend with zero), so we
	// expect y[0] = step value exactly.
	const double STEP = 10.0;
	DA avg(1, DA::Mode::Exponential, /*alpha=*/0.1);

	std::array<double, 1> sweep = {STEP};
	avg.accept_sweep(std::span<const double>{sweep});
	REQUIRE(approx(avg.current_trace()[0], STEP, 1e-12));   // first-sweep seed

	// 50 more sweeps at the same value: output stays at STEP. Tolerance
	// is loose because the denormal AC injection (alternating +/- 1e-8
	// per step) introduces small transient ripple even at steady state;
	// the mean error is zero but the instantaneous error is bounded by
	// ~1e-7 for alpha = 0.1.
	for (int k = 0; k < 50; ++k)
		avg.accept_sweep(std::span<const double>{sweep});
	REQUIRE(approx(avg.current_trace()[0], STEP, 1e-6));

	// Now drop the input to 0; after 50 sweeps the output should be
	// near (1-0.1)^50 * STEP ~= 0.052.
	std::array<double, 1> zero = {0.0};
	for (int k = 0; k < 50; ++k)
		avg.accept_sweep(std::span<const double>{zero});
	const double expected = std::pow(0.9, 50) * STEP;
	REQUIRE(approx(avg.current_trace()[0], expected, expected * 0.01));   // 1% tol
	std::cout << "  exponential_step_settling: passed (decay= "
	          << avg.current_trace()[0] << " vs expected " << expected << ")\n";
}

// ============================================================================
// MaxHold: preserves peaks across sweeps
// ============================================================================

void test_max_hold_preserves_peaks() {
	// Three sweeps, each with a peak at a different bin position.
	// Output after all three should have all three peaks.
	const std::size_t N = 5;
	DA avg(N, DA::Mode::MaxHold);
	std::array<double, 5> a = {1.0, 0.0, 0.0, 0.0, 0.0};
	std::array<double, 5> b = {0.0, 0.0, 2.0, 0.0, 0.0};
	std::array<double, 5> c = {0.0, 0.0, 0.0, 0.0, 3.0};
	avg.accept_sweep(std::span<const double>{a});
	avg.accept_sweep(std::span<const double>{b});
	avg.accept_sweep(std::span<const double>{c});
	auto out = avg.current_trace();
	REQUIRE(out[0] == 1.0);
	REQUIRE(out[1] == 0.0);
	REQUIRE(out[2] == 2.0);
	REQUIRE(out[3] == 0.0);
	REQUIRE(out[4] == 3.0);
	std::cout << "  max_hold_preserves_peaks: passed\n";
}

// ============================================================================
// MinHold: preserves troughs (negative peaks)
// ============================================================================

void test_min_hold_preserves_troughs() {
	const std::size_t N = 4;
	DA avg(N, DA::Mode::MinHold);
	std::array<double, 4> a = {0.0, -1.0, 0.0, 0.0};
	std::array<double, 4> b = {-2.0, 0.0, 0.0, 0.0};
	std::array<double, 4> c = {0.0, 0.0, 0.0, -3.0};
	avg.accept_sweep(std::span<const double>{a});
	avg.accept_sweep(std::span<const double>{b});
	avg.accept_sweep(std::span<const double>{c});
	auto out = avg.current_trace();
	REQUIRE(out[0] == -2.0);
	REQUIRE(out[1] == -1.0);
	REQUIRE(out[2] == 0.0);
	REQUIRE(out[3] == -3.0);
	std::cout << "  min_hold_preserves_troughs: passed\n";
}

// ============================================================================
// MaxHoldN: forgets old peaks after the rolling window slides past
// ============================================================================

void test_max_hold_n_forgets_old_peaks() {
	// Window N=3: a peak inserted at sweep 0 should disappear from the
	// output after 3 sweeps of zero, because the ring no longer contains
	// the peak. (Sweeps 0, 1, 2 cover the peak; from sweep 3 onward the
	// peak has been overwritten.)
	const std::size_t N_BINS = 4;
	DA avg(N_BINS, DA::Mode::MaxHoldN, /*window=*/3.0);

	std::array<double, 4> peak  = {0.0, 0.0, 5.0, 0.0};
	std::array<double, 4> zero4 = {0.0, 0.0, 0.0, 0.0};

	// Sweep 0: peak. Output reflects peak.
	avg.accept_sweep(std::span<const double>{peak});
	REQUIRE(avg.current_trace()[2] == 5.0);

	// Sweeps 1-2: zeros. Window still includes sweep 0; peak persists.
	avg.accept_sweep(std::span<const double>{zero4});
	avg.accept_sweep(std::span<const double>{zero4});
	REQUIRE(avg.current_trace()[2] == 5.0);

	// Sweep 3: zeros. Ring is now [zero, zero, zero] (sweep 0 evicted).
	// Peak is gone.
	avg.accept_sweep(std::span<const double>{zero4});
	REQUIRE(avg.current_trace()[2] == 0.0);
	REQUIRE(avg.sweeps_accumulated() == 4);
	std::cout << "  max_hold_n_forgets_old_peaks: passed\n";
}

// ============================================================================
// reset() clears state
// ============================================================================

void test_reset_clears_state() {
	DA avg(3, DA::Mode::Linear);
	std::array<double, 3> a = {1.0, 2.0, 3.0};
	avg.accept_sweep(std::span<const double>{a});
	avg.accept_sweep(std::span<const double>{a});
	REQUIRE(avg.sweeps_accumulated() == 2);
	REQUIRE(approx(avg.current_trace()[1], 2.0, 1e-12));

	avg.reset();
	REQUIRE(avg.sweeps_accumulated() == 0);
	for (auto v : avg.current_trace()) REQUIRE(v == 0.0);

	// Re-accumulating after reset starts from zero state.
	std::array<double, 3> b = {10.0, 20.0, 30.0};
	avg.accept_sweep(std::span<const double>{b});
	REQUIRE(approx(avg.current_trace()[0], 10.0, 1e-12));
	REQUIRE(avg.sweeps_accumulated() == 1);
	std::cout << "  reset_clears_state: passed\n";
}

// ============================================================================
// Validation
// ============================================================================

void test_construction_validation() {
	bool t1=false, t2=false, t3=false, t4=false, t5=false, t6=false, t7=false;

	try { DA(0, DA::Mode::Linear); }
	catch (const std::invalid_argument&) { t1 = true; }
	REQUIRE(t1);

	// Exponential: alpha out of range (zero, negative, > 1).
	try { DA(4, DA::Mode::Exponential, 0.0); }
	catch (const std::invalid_argument&) { t2 = true; }
	REQUIRE(t2);

	try { DA(4, DA::Mode::Exponential, -0.1); }
	catch (const std::invalid_argument&) { t3 = true; }
	REQUIRE(t3);

	try { DA(4, DA::Mode::Exponential, 1.5); }
	catch (const std::invalid_argument&) { t4 = true; }
	REQUIRE(t4);

	// MaxHoldN: window N < 1
	try { DA(4, DA::Mode::MaxHoldN, 0.0); }
	catch (const std::invalid_argument&) { t5 = true; }
	REQUIRE(t5);

	// MaxHoldN: fractional window (would silently truncate via static_cast
	// without explicit integer-valued validation).
	try { DA(4, DA::Mode::MaxHoldN, 2.5); }
	catch (const std::invalid_argument&) { t6 = true; }
	REQUIRE(t6);

	// MaxHoldN: NaN window (NaN >= 1 is false, so the first check fires;
	// belt-and-suspenders).
	try { DA(4, DA::Mode::MaxHoldN,
	         std::numeric_limits<double>::quiet_NaN()); }
	catch (const std::invalid_argument&) { t7 = true; }
	REQUIRE(t7);

	std::cout << "  construction_validation: passed\n";
}

void test_accept_sweep_length_mismatch_throws() {
	DA avg(8, DA::Mode::MaxHold);
	std::array<double, 7> wrong{};
	bool threw = false;
	try { avg.accept_sweep(std::span<const double>{wrong}); }
	catch (const std::invalid_argument&) { threw = true; }
	REQUIRE(threw);
	std::cout << "  accept_sweep_length_mismatch_throws: passed\n";
}

// ============================================================================
// Mixed-precision sanity: float input
// ============================================================================

void test_float_linear() {
	using FA = TraceAverager<float>;
	FA avg(4, FA::Mode::Linear);
	std::array<float, 4> a = {1.0f, 2.0f, 3.0f, 4.0f};
	std::array<float, 4> b = {3.0f, 4.0f, 5.0f, 6.0f};
	avg.accept_sweep(std::span<const float>{a});
	avg.accept_sweep(std::span<const float>{b});
	auto out = avg.current_trace();
	// Means should be (1+3)/2=2, (2+4)/2=3, (3+5)/2=4, (4+6)/2=5.
	REQUIRE(approx(static_cast<double>(out[0]), 2.0, 1e-6));
	REQUIRE(approx(static_cast<double>(out[1]), 3.0, 1e-6));
	REQUIRE(approx(static_cast<double>(out[2]), 4.0, 1e-6));
	REQUIRE(approx(static_cast<double>(out[3]), 5.0, 1e-6));
	std::cout << "  float_linear: passed\n";
}

// ============================================================================
// main
// ============================================================================

int main() {
	try {
		std::cout << "test_spectrum_trace_averaging\n";

		test_linear_converges_to_truth();
		test_exponential_step_settling();
		test_max_hold_preserves_peaks();
		test_min_hold_preserves_troughs();
		test_max_hold_n_forgets_old_peaks();

		test_reset_clears_state();

		test_construction_validation();
		test_accept_sweep_length_mismatch_throws();

		test_float_linear();

		std::cout << "all tests passed\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "FAILED: " << ex.what() << "\n";
		return 1;
	}
}
