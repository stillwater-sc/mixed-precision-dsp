// test_instrument_peak_detect.cpp: tests for the peak-detect (min/max)
// decimator.
//
// Coverage:
//   - Streaming API: emits std::nullopt while accumulating, pair on
//     window completion
//   - Block API: returns floor(N/R) outputs; partial trailing window
//     correctly dropped
//   - Edge cases: R=1 (passthrough), R > N (no outputs), reset() clears
//     window state
//   - **Glitch survival**: a 50 MHz square wave with a 5-sample-wide
//     narrow positive glitch buried in it. The glitch's peak amplitude
//     must show up in the max stream at every decimation factor we test
//     (R = 2, 4, 8, 16, 32). This is THE acceptance criterion that
//     distinguishes peak-detect decimation from generic averaging or
//     polyphase decimation.
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)`.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sw/dsp/instrument/peak_detect.hpp>

using namespace sw::dsp::instrument;

#define REQUIRE(cond) \
	do { if (!(cond)) throw std::runtime_error( \
		std::string("test failed: ") + #cond + \
		" at " __FILE__ ":" + std::to_string(__LINE__)); } while (0)

// ============================================================================
// Streaming API
// ============================================================================

void test_streaming_basic() {
	PeakDetectDecimator<int> d(/*R=*/4);
	// First three samples — accumulating, no output yet.
	REQUIRE(!d.process(3).has_value());
	REQUIRE(!d.process(7).has_value());
	REQUIRE(!d.process(1).has_value());
	// Fourth sample — completes the window, returns (min, max) = (1, 7).
	auto p = d.process(5);
	REQUIRE(p.has_value());
	REQUIRE(p->first  == 1);
	REQUIRE(p->second == 7);
	// Next window starts fresh.
	REQUIRE(!d.process(10).has_value());
	REQUIRE(!d.process(2).has_value());
	REQUIRE(!d.process(9).has_value());
	auto p2 = d.process(8);
	REQUIRE(p2.has_value());
	REQUIRE(p2->first  == 2);
	REQUIRE(p2->second == 10);
	std::cout << "  streaming_basic: passed\n";
}

void test_streaming_negative_values() {
	// Make sure min/max work for signed values across zero.
	PeakDetectDecimator<double> d(3);
	REQUIRE(!d.process(-1.5).has_value());
	REQUIRE(!d.process( 0.0).has_value());
	auto p = d.process(2.5);
	REQUIRE(p.has_value());
	REQUIRE(p->first  == -1.5);
	REQUIRE(p->second ==  2.5);
	std::cout << "  streaming_negative_values: passed\n";
}

void test_streaming_decimation_one_passthrough() {
	// R=1: every sample becomes (sample, sample) immediately.
	PeakDetectDecimator<float> d(1);
	for (float x : {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}) {
		auto p = d.process(x);
		REQUIRE(p.has_value());
		REQUIRE(p->first  == x);
		REQUIRE(p->second == x);
	}
	std::cout << "  streaming_decimation_one_passthrough: passed\n";
}

void test_constructor_zero_throws() {
	bool threw = false;
	try { PeakDetectDecimator<int>(0); }
	catch (const std::invalid_argument&) { threw = true; }
	REQUIRE(threw);
	std::cout << "  constructor_zero_throws: passed\n";
}

// ============================================================================
// Block API
// ============================================================================

void test_block_basic() {
	std::vector<int> in = {3, 7, 1, 5, 10, 2, 9, 8};
	PeakDetectDecimator<int> d(4);
	auto env = d.process_block(std::span<const int>(in.data(), in.size()));
	REQUIRE(env.mins.size() == 2);
	REQUIRE(env.maxs.size() == 2);
	REQUIRE(env.mins[0] == 1);
	REQUIRE(env.maxs[0] == 7);
	REQUIRE(env.mins[1] == 2);
	REQUIRE(env.maxs[1] == 10);
	std::cout << "  block_basic: passed\n";
}

void test_block_drops_partial_trailing_window() {
	std::vector<int> in = {1, 2, 3, 4, 5, 6, 7};   // 7 samples, R=4 → 1 output
	PeakDetectDecimator<int> d(4);
	auto env = d.process_block(std::span<const int>(in.data(), in.size()));
	REQUIRE(env.mins.size() == 1);
	REQUIRE(env.mins[0] == 1);
	REQUIRE(env.maxs[0] == 4);
	// The trailing 3 samples (5,6,7) are dropped — they form a partial window.
	std::cout << "  block_drops_partial_trailing_window: passed\n";
}

void test_block_input_smaller_than_factor() {
	// Input length 3, R=4 — no complete window, zero outputs.
	std::vector<int> in = {1, 2, 3};
	PeakDetectDecimator<int> d(4);
	auto env = d.process_block(std::span<const int>(in.data(), in.size()));
	REQUIRE(env.mins.size() == 0);
	REQUIRE(env.maxs.size() == 0);
	std::cout << "  block_input_smaller_than_factor: passed\n";
}

void test_separate_min_max_block_apis() {
	// process_block_min / process_block_max should agree with
	// the unified process_block.
	std::vector<int> in = {3, 7, 1, 5, 10, 2, 9, 8};
	PeakDetectDecimator<int> d_a(4);
	auto mins = d_a.process_block_min(std::span<const int>(in.data(), in.size()));
	PeakDetectDecimator<int> d_b(4);
	auto maxs = d_b.process_block_max(std::span<const int>(in.data(), in.size()));
	REQUIRE(mins.size() == 2 && maxs.size() == 2);
	REQUIRE(mins[0] == 1 && maxs[0] == 7);
	REQUIRE(mins[1] == 2 && maxs[1] == 10);
	std::cout << "  separate_min_max_block_apis: passed\n";
}

// ============================================================================
// reset()
// ============================================================================

void test_reset_drops_partial_window() {
	PeakDetectDecimator<int> d(4);
	d.process(100);   // partial window
	d.process(200);
	REQUIRE(d.samples_in_window() == 2);
	d.reset();
	REQUIRE(d.samples_in_window() == 0);
	// Now run a fresh window — the previous samples should NOT be in it.
	REQUIRE(!d.process(1).has_value());
	REQUIRE(!d.process(2).has_value());
	REQUIRE(!d.process(3).has_value());
	auto p = d.process(4);
	REQUIRE(p.has_value());
	REQUIRE(p->first  == 1);   // not 1 vs 100 — reset cleared the window
	REQUIRE(p->second == 4);
	std::cout << "  reset_drops_partial_window: passed\n";
}

// ============================================================================
// GLITCH-SURVIVAL TEST — the headline acceptance criterion
//
// Synthesize a 50 MHz square wave at fs=1 GSPS (so 20 samples per period,
// 10 high + 10 low) with a 5-sample-wide narrow positive glitch buried
// somewhere in the LOW phase. The glitch's peak amplitude is +1.5
// (compared to the square wave's ±1.0). We then run peak-detect at
// decimation factors 2, 4, 8, 16, 32 and verify that the glitch's
// amplitude shows up in the max output at every factor.
//
// This is what distinguishes a scope's peak-detect decimator from a
// generic averaging or polyphase decimator: those would average the
// glitch out at sufficient decimation; peak-detect MUST preserve it.
// ============================================================================

std::vector<float> make_square_wave_with_glitch() {
	const std::size_t N = 4096;        // total samples
	std::vector<float> stream(N, 0.0f);

	// 50 MHz square wave at fs=1 GSPS = 20 samples per period
	for (std::size_t n = 0; n < N; ++n) {
		stream[n] = ((n / 10) % 2 == 0) ? 1.0f : -1.0f;
	}
	// 5-sample-wide positive glitch buried in a LOW phase (e.g., samples
	// 510..514 within a low region, so the surrounding values are -1.0).
	for (std::size_t n = 510; n < 515; ++n) {
		stream[n] = 1.5f;   // peaks above the +1.0 high level
	}
	return stream;
}

void test_glitch_survives_at_all_decimation_factors() {
	const auto stream = make_square_wave_with_glitch();
	const float glitch_peak = 1.5f;

	for (std::size_t R : {std::size_t{2}, std::size_t{4}, std::size_t{8},
	                       std::size_t{16}, std::size_t{32}}) {
		PeakDetectDecimator<float> d(R);
		auto env = d.process_block(
			std::span<const float>(stream.data(), stream.size()));

		// Find the max-stream peak. With the glitch present, this should
		// be exactly the glitch's amplitude (1.5), not the square wave's
		// (1.0).
		float observed_peak = -1e9f;
		for (std::size_t i = 0; i < env.maxs.size(); ++i) {
			if (env.maxs[i] > observed_peak) observed_peak = env.maxs[i];
		}

		// The glitch's amplitude must be visible — observed_peak should
		// be at or very close to glitch_peak. With a generic averaging
		// decimator at R=32, observed_peak would drop toward (5 * 1.5 +
		// 27 * -1) / 32 ≈ -0.61, but with peak-detect it stays at 1.5.
		if (!(std::abs(observed_peak - glitch_peak) < 1e-5f))
			throw std::runtime_error(
				"glitch lost at R=" + std::to_string(R) + ": observed_peak=" +
				std::to_string(observed_peak) + " want=" +
				std::to_string(glitch_peak));
	}
	std::cout << "  glitch_survives_at_all_decimation_factors: passed (R=2,4,8,16,32)\n";
}

// ============================================================================
// main
// ============================================================================

int main() {
	try {
		std::cout << "test_instrument_peak_detect\n";

		test_streaming_basic();
		test_streaming_negative_values();
		test_streaming_decimation_one_passthrough();
		test_constructor_zero_throws();

		test_block_basic();
		test_block_drops_partial_trailing_window();
		test_block_input_smaller_than_factor();
		test_separate_min_max_block_apis();

		test_reset_drops_partial_window();

		test_glitch_survives_at_all_decimation_factors();

		std::cout << "all tests passed\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "FAILED: " << ex.what() << "\n";
		return 1;
	}
}
