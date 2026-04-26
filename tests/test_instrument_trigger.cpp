// test_instrument_trigger.cpp: tests for the trigger primitive library.
//
// Coverage:
//   - EdgeTrigger: rising / falling / either, hysteresis, samples_since_trigger
//   - LevelTrigger: ergonomic alias for EdgeTrigger w/ Slope::Either
//   - SlopeTrigger: positive / negative / either signs, threshold gating
//   - HoldoffWrapper: suppresses fires within window; inner state stays sane
//   - QualifierAnd: coincidence within window; non-coincidence stays silent
//   - QualifierOr: per-sample disjunction
//   - Precision sweep: minimum-detectable-edge degradation as SampleScalar
//                       narrows (double / float / posit32 / posit16)
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)` rather
// than assert(); CI runs in Release where assert is stripped.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sw/dsp/instrument/trigger.hpp>
#include <sw/universal/number/posit/posit.hpp>

using namespace sw::dsp::instrument;

#define REQUIRE(cond) \
	do { if (!(cond)) throw std::runtime_error( \
		std::string("test failed: ") + #cond + \
		" at " __FILE__ ":" + std::to_string(__LINE__)); } while (0)

// ============================================================================
// EdgeTrigger — basic rising / falling / either
// ============================================================================

void test_edge_rising() {
	EdgeTrigger<double> t(/*level=*/0.5, Slope::Rising);
	// Below the level — no fires
	REQUIRE(!t.process(0.0));
	REQUIRE(!t.process(0.2));
	REQUIRE(!t.process(0.4));
	// Cross up — fires
	REQUIRE(t.process(0.7));
	// Stay above — no re-fire
	REQUIRE(!t.process(0.8));
	REQUIRE(!t.process(1.0));
	// Cross back down — no fire (Rising slope only)
	REQUIRE(!t.process(0.3));
	// Cross up again — fires
	REQUIRE(t.process(0.6));
	std::cout << "  edge_rising: passed\n";
}

void test_edge_falling() {
	EdgeTrigger<double> t(/*level=*/0.5, Slope::Falling);
	// Start above — no fire on entry (state was Unknown, transitioning to Above)
	REQUIRE(!t.process(0.7));
	// Cross down — fires
	REQUIRE(t.process(0.3));
	// Stay below — no re-fire
	REQUIRE(!t.process(0.0));
	// Cross up — no fire (Falling slope only)
	REQUIRE(!t.process(0.8));
	// Cross down again — fires
	REQUIRE(t.process(0.2));
	std::cout << "  edge_falling: passed\n";
}

void test_edge_either() {
	EdgeTrigger<double> t(/*level=*/0.5, Slope::Either);
	REQUIRE(!t.process(0.0));   // Below
	REQUIRE(t.process(0.7));    // Up — fires
	REQUIRE(t.process(0.2));    // Down — fires
	REQUIRE(t.process(0.8));    // Up — fires
	REQUIRE(!t.process(0.9));   // Stay above — no fire
	std::cout << "  edge_either: passed\n";
}

// ============================================================================
// EdgeTrigger — hysteresis
// ============================================================================

void test_edge_hysteresis() {
	// Hysteresis = 0.2 means: upper boundary = 0.6, lower = 0.4.
	// Signal must cross OUT of one band and INTO the other; samples in
	// [0.4, 0.6] don't change the region.
	EdgeTrigger<double> t(/*level=*/0.5, Slope::Either, /*hysteresis=*/0.2);
	REQUIRE(!t.process(0.0));   // Below
	REQUIRE(!t.process(0.45));  // Inside band — state stays Below
	REQUIRE(!t.process(0.55));  // Inside band — state still Below
	REQUIRE(t.process(0.7));    // Out the top — fires (Below -> Above)
	REQUIRE(!t.process(0.55));  // Inside band — state stays Above
	REQUIRE(!t.process(0.45));  // Inside band — still Above
	REQUIRE(t.process(0.3));    // Out the bottom — fires (Above -> Below)
	std::cout << "  edge_hysteresis: passed\n";
}

void test_edge_hysteresis_suppresses_noise() {
	// Without hysteresis, repeated noise around level fires every crossing.
	// With hysteresis, dwell within the band is silent.
	EdgeTrigger<double> nonoise(/*level=*/0.5, Slope::Either, /*hyst=*/0.0);
	EdgeTrigger<double> hyst(/*level=*/0.5, Slope::Either,    /*hyst=*/0.2);
	const std::vector<double> noisy = {0.0, 0.51, 0.49, 0.51, 0.49, 0.51, 1.0};
	int nonoise_fires = 0;
	int hyst_fires    = 0;
	for (double x : noisy) {
		if (nonoise.process(x)) ++nonoise_fires;
		if (hyst.process(x))    ++hyst_fires;
	}
	REQUIRE(nonoise_fires >= 4);   // every noise wiggle counts
	REQUIRE(hyst_fires == 1);      // only the final clean cross matters
	std::cout << "  edge_hysteresis_suppresses_noise: passed\n";
}

void test_edge_negative_hysteresis_throws() {
	bool threw = false;
	try { EdgeTrigger<double>(0.5, Slope::Rising, -0.1); }
	catch (const std::invalid_argument&) { threw = true; }
	REQUIRE(threw);
	std::cout << "  edge_negative_hysteresis_throws: passed\n";
}

// ============================================================================
// EdgeTrigger — samples_since_trigger
// ============================================================================

void test_samples_since() {
	EdgeTrigger<double> t(0.5, Slope::Rising);
	REQUIRE(t.samples_since_trigger() == 0);
	t.process(0.0);  // 1
	t.process(0.0);  // 2
	t.process(0.0);  // 3
	REQUIRE(t.samples_since_trigger() == 3);
	t.process(0.7);  // fires; counter resets
	REQUIRE(t.samples_since_trigger() == 0);
	t.process(0.8);  // 1
	t.process(0.9);  // 2
	REQUIRE(t.samples_since_trigger() == 2);
	std::cout << "  samples_since: passed\n";
}

// ============================================================================
// LevelTrigger
// ============================================================================

void test_level_trigger() {
	LevelTrigger<double> t(0.5);
	REQUIRE(!t.process(0.0));
	REQUIRE(t.process(0.7));   // up
	REQUIRE(t.process(0.3));   // down
	REQUIRE(t.process(0.7));   // up
	std::cout << "  level_trigger: passed\n";
}

// ============================================================================
// SlopeTrigger
// ============================================================================

void test_slope_positive() {
	SlopeTrigger<double> t(/*threshold=*/0.5, Slope::Rising);
	t.process(0.0);            // first sample, no fire
	REQUIRE(!t.process(0.3));  // diff = +0.3, below threshold
	REQUIRE(t.process(0.9));   // diff = +0.6, above threshold, positive — fires
	REQUIRE(t.process(1.5));   // diff = +0.6 again — independent step, fires too
	REQUIRE(!t.process(1.7));  // diff = +0.2, below threshold
	REQUIRE(!t.process(0.5));  // diff = -1.2, large but Rising slope only
	std::cout << "  slope_positive: passed\n";
}

void test_slope_negative() {
	SlopeTrigger<double> t(0.5, Slope::Falling);
	t.process(1.0);
	REQUIRE(!t.process(0.7));  // diff = -0.3
	REQUIRE(t.process(0.0));   // diff = -0.7 — fires
	REQUIRE(!t.process(0.5));  // diff = +0.5, but Falling slope ignores positives
	std::cout << "  slope_negative: passed\n";
}

void test_slope_either() {
	SlopeTrigger<double> t(0.5, Slope::Either);
	t.process(0.0);
	REQUIRE(t.process(0.7));   // +0.7 — fires
	REQUIRE(t.process(0.0));   // -0.7 — fires
	REQUIRE(!t.process(0.2));  // +0.2 — below threshold
	std::cout << "  slope_either: passed\n";
}

void test_slope_negative_threshold_throws() {
	bool threw = false;
	try { SlopeTrigger<double>(-0.1); }
	catch (const std::invalid_argument&) { threw = true; }
	REQUIRE(threw);
	std::cout << "  slope_negative_threshold_throws: passed\n";
}

// ============================================================================
// HoldoffWrapper
// ============================================================================

void test_holdoff_basic() {
	// Edge-rising-trigger that would otherwise fire on every up-crossing,
	// wrapped with a 5-sample holdoff.
	using Inner = EdgeTrigger<double>;
	HoldoffWrapper<Inner> t(Inner(/*level=*/0.5, Slope::Either), /*holdoff=*/5);

	REQUIRE(!t.process(0.0));
	REQUIRE(t.process(0.7));        // first fire
	// Next 5 samples: any crossings should NOT fire
	REQUIRE(!t.process(0.3));       // would normally fire (Either)
	REQUIRE(!t.process(0.8));       // would normally fire
	REQUIRE(!t.process(0.2));
	REQUIRE(!t.process(0.9));
	REQUIRE(!t.process(0.0));
	// 6th sample after fire — holdoff expired, fire is allowed again
	REQUIRE(t.process(0.7));
	std::cout << "  holdoff_basic: passed\n";
}

void test_holdoff_drives_inner() {
	// Critical: the inner trigger MUST be driven during holdoff so its
	// state (region for EdgeTrigger) is current. If we skipped, the very
	// first sample after holdoff would see a stale Region and could fire
	// spuriously.
	using Inner = EdgeTrigger<double>;
	HoldoffWrapper<Inner> t(Inner(0.5, Slope::Rising), /*holdoff=*/3);

	REQUIRE(!t.process(0.0));
	REQUIRE(t.process(0.7));        // first fire — state goes Above
	// Inside holdoff: signal goes back below, then up. If we skipped the
	// inner trigger, the inner's state would still be Above and the next
	// up-crossing wouldn't register correctly.
	REQUIRE(!t.process(0.0));  // 1: drives inner -> Below
	REQUIRE(!t.process(0.7));  // 2: drives inner -> Above (would have been a fire if not in holdoff)
	REQUIRE(!t.process(0.0));  // 3: drives inner -> Below
	// Holdoff expired. Now an up-cross should fire normally.
	REQUIRE(t.process(0.7));
	std::cout << "  holdoff_drives_inner: passed\n";
}

// ============================================================================
// QualifierAnd
// ============================================================================

void test_qualifier_and_coincidence() {
	using TA = EdgeTrigger<double>;
	using TB = EdgeTrigger<double>;
	QualifierAnd<TA, TB> q(TA(0.5, Slope::Rising),
	                       TB(0.5, Slope::Rising),
	                       /*window=*/3);

	// Both go up on the SAME sample
	REQUIRE(!q.process(0.0, 0.0));
	REQUIRE(q.process(0.7, 0.7));   // both fire same sample, AND fires
	std::cout << "  qualifier_and_coincidence: passed\n";
}

void test_qualifier_and_within_window() {
	using TA = EdgeTrigger<double>;
	using TB = EdgeTrigger<double>;
	QualifierAnd<TA, TB> q(TA(0.5, Slope::Rising),
	                       TB(0.5, Slope::Rising),
	                       /*window=*/3);

	REQUIRE(!q.process(0.0, 0.0));
	REQUIRE(!q.process(0.7, 0.0));   // A fires; B has not fired
	REQUIRE(!q.process(0.8, 0.0));   // 1 sample later
	REQUIRE(!q.process(0.9, 0.0));   // 2 samples later
	REQUIRE(q.process(0.9, 0.7));    // 3 samples after A's fire — B fires now, AND fires (A still inside window)
	std::cout << "  qualifier_and_within_window: passed\n";
}

void test_qualifier_and_outside_window() {
	using TA = EdgeTrigger<double>;
	using TB = EdgeTrigger<double>;
	QualifierAnd<TA, TB> q(TA(0.5, Slope::Rising),
	                       TB(0.5, Slope::Rising),
	                       /*window=*/2);

	REQUIRE(!q.process(0.7, 0.0));   // A fires
	REQUIRE(!q.process(0.8, 0.0));   // 1
	REQUIRE(!q.process(0.9, 0.0));   // 2 (still in window)
	REQUIRE(!q.process(0.9, 0.0));   // 3 (out of window)
	REQUIRE(!q.process(0.9, 0.7));   // B fires now, but A is outside window
	std::cout << "  qualifier_and_outside_window: passed\n";
}

// ============================================================================
// QualifierOr
// ============================================================================

void test_qualifier_or() {
	using TA = EdgeTrigger<double>;
	using TB = EdgeTrigger<double>;
	QualifierOr<TA, TB> q(TA(0.5, Slope::Rising),
	                      TB(0.5, Slope::Rising));

	REQUIRE(!q.process(0.0, 0.0));
	REQUIRE(q.process(0.7, 0.0));    // A fires
	REQUIRE(q.process(0.0, 0.7));    // first sample for B; but B was below -> above, fires
	// After both have fired, neither is currently above on next call
	REQUIRE(!q.process(0.0, 0.0));   // both below
	REQUIRE(q.process(0.7, 0.7));    // both fire same sample
	std::cout << "  qualifier_or: passed\n";
}

// ============================================================================
// Precision sweep — minimum-detectable-edge across types
// ============================================================================

template <class T>
double measure_min_detectable_edge_near_unity() {
	// Find the smallest amplitude A such that an EdgeTrigger at level=1.0
	// fires reliably when the signal swings from (1.0 - A) to (1.0 + A).
	// Centered at unity to expose the precision differences: every type's
	// ULP varies by location, and unity is the standard reference point.
	// (A test centered at zero is too easy — every type has dense precision
	// near zero, so all types appear identical.)
	auto fires_at = [](double amplitude) {
		EdgeTrigger<T> t(static_cast<T>(1.0), Slope::Rising);
		t.process(static_cast<T>(1.0 - amplitude));  // establish Below
		return t.process(static_cast<T>(1.0 + amplitude));
	};

	double lo = 1e-20;
	double hi = 0.5;
	for (int i = 0; i < 200; ++i) {
		double mid = std::sqrt(lo * hi);
		if (fires_at(mid)) hi = mid;
		else               lo = mid;
		if (hi / lo < 1.001) break;
	}
	return hi;
}

void test_precision_sweep_minimum_detectable() {
	const double me_double = measure_min_detectable_edge_near_unity<double>();
	const double me_float  = measure_min_detectable_edge_near_unity<float>();
	const double me_p32    = measure_min_detectable_edge_near_unity<
		sw::universal::posit<32, 2>>();
	const double me_p16    = measure_min_detectable_edge_near_unity<
		sw::universal::posit<16, 2>>();

	std::cout << "  precision sweep: min detectable edge near level=1.0\n";
	std::cout << "    double:        " << me_double << "\n";
	std::cout << "    float:         " << me_float  << "\n";
	std::cout << "    posit<32,2>:   " << me_p32    << "\n";
	std::cout << "    posit<16,2>:   " << me_p16    << "\n";

	// At level=1.0, the floor is set by each type's ULP near unity:
	//   double:      2^-52 ≈ 2.2e-16
	//   float:       2^-23 ≈ 1.2e-7
	//   posit<32,2>: ~28 mantissa bits near unity → ~3.7e-9
	//   posit<16,2>: ~12 mantissa bits near unity → ~2.4e-4
	REQUIRE(me_double < 1e-14);   // close to double ULP at 1.0
	REQUIRE(me_float  < 1e-6);    // close to float ULP at 1.0
	REQUIRE(me_p32    < 1e-7);    // posit32 better than float near unity
	REQUIRE(me_p16    < 1e-2);    // posit16 ~12 mantissa bits → ms-scale floor

	// Confirm the expected ordering: narrower types detect coarser edges.
	REQUIRE(me_double <= me_float);
	REQUIRE(me_p32    <= me_float);   // posit32 ties or beats float at unity
	REQUIRE(me_p16    >  me_float);   // posit16 is meaningfully worse than float
	std::cout << "  precision_sweep_minimum_detectable: passed\n";
}

// ============================================================================
// main
// ============================================================================

int main() {
	try {
		std::cout << "test_instrument_trigger\n";

		test_edge_rising();
		test_edge_falling();
		test_edge_either();
		test_edge_hysteresis();
		test_edge_hysteresis_suppresses_noise();
		test_edge_negative_hysteresis_throws();
		test_samples_since();
		test_level_trigger();
		test_slope_positive();
		test_slope_negative();
		test_slope_either();
		test_slope_negative_threshold_throws();
		test_holdoff_basic();
		test_holdoff_drives_inner();
		test_qualifier_and_coincidence();
		test_qualifier_and_within_window();
		test_qualifier_and_outside_window();
		test_qualifier_or();
		test_precision_sweep_minimum_detectable();

		std::cout << "all tests passed\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "FAILED: " << ex.what() << "\n";
		return 1;
	}
}
