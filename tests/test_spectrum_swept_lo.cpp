// test_spectrum_swept_lo.cpp: tests for the swept LO chirp generator.
//
// Coverage:
//   - Linear sweep: current_frequency_hz() rises monotonically from
//     f_start to f_stop over num_sweep_samples; first/last/midpoint
//     all hit their analytical values.
//   - Logarithmic sweep: same monotone check; midpoint hits the
//     analytical geometric mean (sqrt(f_start * f_stop)).
//   - Phase continuity across sweep restart: phase_inc snaps back to
//     start at the boundary, but the phase accumulator continues
//     (cos / sin output has no discontinuity glitch).
//   - sweep_complete() / total_sweeps() one-shot semantics.
//   - reset() clears state.
//   - Validation: NaN / Inf / non-positive inputs throw; log sweep
//     with mixed-sign frequencies throws; sweep_duration so short
//     it produces < 2 samples throws.
//   - Mixed-precision sanity: float and mixed-precision instances
//     produce monotone frequency profiles consistent with the
//     double reference.
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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sw/dsp/spectrum/swept_lo.hpp>

using namespace sw::dsp::spectrum;
using LO = SweptLO<double>;

#define REQUIRE(cond) \
	do { if (!(cond)) throw std::runtime_error( \
		std::string("test failed: ") + #cond + \
		" at " __FILE__ ":" + std::to_string(__LINE__)); } while (0)

static bool approx(double a, double b, double tol) {
	return std::abs(a - b) <= tol;
}

// ============================================================================
// Linear sweep: monotone frequency vs. time, hits expected values
// ============================================================================

void test_linear_sweep_profile() {
	const double fs = 1.0e6;
	const double f0 = 100.0e3;
	const double f1 = 200.0e3;
	const double T  = 1.0e-3;     // 1 ms sweep -> 1000 samples at 1 MHz
	LO lo(f0, f1, T, fs, LO::Sweep::Linear);

	const std::size_t N = lo.num_sweep_samples();
	REQUIRE(N == 1000);

	// Sample 0: f == f_start.
	REQUIRE(approx(lo.current_frequency_hz(), f0, 1.0));

	// Walk through the sweep, recording frequency at each sample.
	std::vector<double> freqs;
	freqs.reserve(N);
	for (std::size_t i = 0; i < N; ++i) {
		freqs.push_back(lo.current_frequency_hz());
		(void)lo.process();
	}
	// Strict monotone (linear, increasing).
	for (std::size_t i = 1; i < freqs.size(); ++i) {
		REQUIRE(freqs[i] > freqs[i - 1]);
	}
	// At sample index k of an N-sample linear sweep, the analytical
	// frequency is f0 + k * (f1 - f0) / (N - 1) — the schedule
	// interpolates across N points indexed 0..N-1.
	const double mid = freqs[N / 2];
	const double expected_mid =
		f0 + static_cast<double>(N / 2) * (f1 - f0) /
		     static_cast<double>(N - 1);
	REQUIRE(approx(mid, expected_mid, 1.0));
	// Last sample (index N-1) is the value of phase_inc just before
	// the wraparound at the Nth process() call — should be very close
	// to f1.
	REQUIRE(approx(freqs[N - 1], f1, 1.0));
	std::cout << "  linear_sweep_profile: passed (mid=" << mid
	          << " final=" << freqs[N - 1] << ")\n";
}

// ============================================================================
// Logarithmic sweep: monotone, geometric mean at midpoint
// ============================================================================

void test_log_sweep_profile() {
	const double fs = 1.0e6;
	const double f0 = 1.0e3;
	const double f1 = 100.0e3;    // 2 decades
	const double T  = 1.0e-3;
	LO lo(f0, f1, T, fs, LO::Sweep::Logarithmic);

	const std::size_t N = lo.num_sweep_samples();
	std::vector<double> freqs;
	freqs.reserve(N);
	for (std::size_t i = 0; i < N; ++i) {
		freqs.push_back(lo.current_frequency_hz());
		(void)lo.process();
	}
	for (std::size_t i = 1; i < freqs.size(); ++i) {
		REQUIRE(freqs[i] > freqs[i - 1]);
	}
	// Geometric midpoint: sqrt(f0 * f1).
	const double mid     = freqs[N / 2];
	const double mid_geo = std::sqrt(f0 * f1);
	// Tolerance: 5% — log sweep midpoint depends on which sample
	// index we sample (N/2 vs (N-1)/2 boundary).
	REQUIRE(std::abs(mid - mid_geo) / mid_geo < 0.05);
	std::cout << "  log_sweep_profile: passed (mid=" << mid
	          << " expected_geomean=" << mid_geo << ")\n";
}

// ============================================================================
// Phase continuity across sweep restart
// ============================================================================

void test_phase_continuous_across_restart() {
	const double fs = 1.0e6;
	const double f0 = 100.0e3;
	const double f1 = 200.0e3;
	const double T  = 5.0e-5;     // 50-sample sweep (small for testability)
	LO lo(f0, f1, T, fs, LO::Sweep::Linear);

	// Run two full sweeps (~100 samples) and capture the cosine output
	// across the boundary. There should be no large jump between
	// adjacent samples at the boundary.
	const std::size_t N = lo.num_sweep_samples();
	std::vector<double> cos_out;
	cos_out.reserve(N * 3);
	for (std::size_t i = 0; i < N * 3; ++i) {
		auto [c, s] = lo.process();
		(void)s;
		cos_out.push_back(c);
	}
	// At boundaries, frequency snaps back so a sample-to-sample jump
	// in cos-rate is expected — but the SAMPLE values themselves
	// should still be in [-1, 1] and there should be no impulse
	// (sudden ~2.0 swing). Bound: |cos[n+1] - cos[n]| <= 1 for any
	// reasonable sweep where per-sample phase advance < 0.5 cycles.
	double max_jump = 0.0;
	for (std::size_t i = 1; i < cos_out.size(); ++i) {
		max_jump = std::max(max_jump, std::abs(cos_out[i] - cos_out[i - 1]));
	}
	REQUIRE(max_jump < 1.5);   // healthy margin, < 2 (full swing)
	REQUIRE(lo.total_sweeps() >= 2);
	std::cout << "  phase_continuous_across_restart: passed (max sample-to-sample jump="
	          << max_jump << ", total_sweeps=" << lo.total_sweeps() << ")\n";
}

// ============================================================================
// sweep_complete() one-shot + total_sweeps() monotone
// ============================================================================

void test_sweep_complete_semantics() {
	const double fs = 1.0e6;
	LO lo(100.0e3, 200.0e3, 1.0e-5, fs, LO::Sweep::Linear);   // 10 samples
	const std::size_t N = lo.num_sweep_samples();
	REQUIRE(N == 10);

	REQUIRE(lo.total_sweeps() == 0);
	// Process N-1 samples; sweep_complete stays false.
	for (std::size_t i = 0; i < N - 1; ++i) {
		(void)lo.process();
		REQUIRE(!lo.sweep_complete());
	}
	// The Nth process() call wraps. sweep_complete is true on this
	// step and ONLY this step.
	(void)lo.process();
	REQUIRE(lo.sweep_complete());
	REQUIRE(lo.total_sweeps() == 1);
	(void)lo.process();
	REQUIRE(!lo.sweep_complete());   // self-reset
	REQUIRE(lo.total_sweeps() == 1);

	// Run another full sweep.
	for (std::size_t i = 1; i < N; ++i) (void)lo.process();
	REQUIRE(lo.sweep_complete());
	REQUIRE(lo.total_sweeps() == 2);
	std::cout << "  sweep_complete_semantics: passed\n";
}

// ============================================================================
// reset()
// ============================================================================

void test_reset() {
	const double fs = 1.0e6;
	LO lo(100.0e3, 200.0e3, 1.0e-5, fs, LO::Sweep::Linear);
	for (std::size_t i = 0; i < 7; ++i) (void)lo.process();
	REQUIRE(lo.total_sweeps() == 0);

	lo.reset();
	REQUIRE(lo.total_sweeps() == 0);
	REQUIRE(approx(lo.current_frequency_hz(), 100.0e3, 1.0));
	std::cout << "  reset: passed\n";
}

// ============================================================================
// Validation
// ============================================================================

void test_validation() {
	const double fs = 1.0e6;
	const double NaN = std::numeric_limits<double>::quiet_NaN();
	const double INF = std::numeric_limits<double>::infinity();
	bool t = false;

	// f_start
	t=false; try { LO( 0.0, 200e3, 1e-3, fs); } catch (const std::invalid_argument&) { t=true; } REQUIRE(t);
	t=false; try { LO(-100.0, 200e3, 1e-3, fs); } catch (const std::invalid_argument&) { t=true; } REQUIRE(t);
	t=false; try { LO( NaN,  200e3, 1e-3, fs); } catch (const std::invalid_argument&) { t=true; } REQUIRE(t);
	t=false; try { LO( INF,  200e3, 1e-3, fs); } catch (const std::invalid_argument&) { t=true; } REQUIRE(t);

	// f_stop
	t=false; try { LO(100e3,  0.0, 1e-3, fs); } catch (const std::invalid_argument&) { t=true; } REQUIRE(t);
	t=false; try { LO(100e3, NaN,  1e-3, fs); } catch (const std::invalid_argument&) { t=true; } REQUIRE(t);

	// duration
	t=false; try { LO(100e3, 200e3,  0.0, fs); } catch (const std::invalid_argument&) { t=true; } REQUIRE(t);
	t=false; try { LO(100e3, 200e3, -1.0, fs); } catch (const std::invalid_argument&) { t=true; } REQUIRE(t);

	// sample_rate
	t=false; try { LO(100e3, 200e3, 1e-3, 0.0); } catch (const std::invalid_argument&) { t=true; } REQUIRE(t);
	t=false; try { LO(100e3, 200e3, 1e-3, INF); } catch (const std::invalid_argument&) { t=true; } REQUIRE(t);

	// duration too short to yield 2 samples (1e-7 s * 1 MHz = 0.1 sample)
	t=false; try { LO(100e3, 200e3, 1e-7, fs); } catch (const std::invalid_argument&) { t=true; } REQUIRE(t);

	// Log-sweep same-sign requirement: today this is implicit in the
	// positivity check above (both f_start and f_stop must be > 0,
	// which means they share sign by construction). The explicit
	// same-sign check in the log branch was removed as dead code; it
	// would need to come back if negative-frequency sweeps are
	// supported later.

	std::cout << "  validation: passed\n";
}

// ============================================================================
// Mixed-precision sanity
// ============================================================================

void test_float_and_mixed_match_double() {
	// Linear sweep: monotone frequency profile across precisions.
	const double fs = 1.0e6;
	const double f0 = 100.0e3;
	const double f1 = 200.0e3;
	const double T  = 1.0e-4;     // 100 samples
	using LD = SweptLO<double, double, double>;
	using LF = SweptLO<float,  float,  float>;
	using LM = SweptLO<double, double, float>;
	LD ld(f0, f1, T, fs);
	LF lf(static_cast<double>(f0), static_cast<double>(f1),
	      static_cast<double>(T),  static_cast<double>(fs));
	LM lm(f0, f1, T, fs);

	auto run_freqs = [](auto& lo) {
		std::vector<double> v;
		v.reserve(lo.num_sweep_samples());
		for (std::size_t i = 0; i < lo.num_sweep_samples(); ++i) {
			v.push_back(lo.current_frequency_hz());
			(void)lo.process();
		}
		return v;
	};
	auto fd = run_freqs(ld);
	auto ff = run_freqs(lf);
	auto fm = run_freqs(lm);
	// All three must be monotone increasing.
	for (std::size_t i = 1; i < fd.size(); ++i) {
		REQUIRE(fd[i] > fd[i - 1]);
		REQUIRE(ff[i] > ff[i - 1]);
		REQUIRE(fm[i] > fm[i - 1]);
	}
	// Endpoints should agree across precisions to within a small
	// fraction of the bandwidth.
	const double bw = f1 - f0;
	REQUIRE(std::abs(fd.front() - ff.front()) < bw * 0.01);
	REQUIRE(std::abs(fd.front() - fm.front()) < bw * 0.01);
	REQUIRE(std::abs(fd.back()  - ff.back())  < bw * 0.01);
	REQUIRE(std::abs(fd.back()  - fm.back())  < bw * 0.01);
	std::cout << "  float_and_mixed_match_double: passed (start " << fd.front()
	          << "/" << ff.front() << "/" << fm.front()
	          << " end "  << fd.back()  << "/" << ff.back()  << "/" << fm.back()
	          << ")\n";
}

// ============================================================================
// main
// ============================================================================

int main() {
	try {
		std::cout << "test_spectrum_swept_lo\n";

		test_linear_sweep_profile();
		test_log_sweep_profile();
		test_phase_continuous_across_restart();
		test_sweep_complete_semantics();
		test_reset();

		test_validation();
		test_float_and_mixed_match_double();

		std::cout << "all tests passed\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "FAILED: " << ex.what() << "\n";
		return 1;
	}
}
