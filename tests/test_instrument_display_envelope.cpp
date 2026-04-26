// test_instrument_display_envelope.cpp: tests for the display-rate min/max
// envelope reducer.
//
// Coverage:
//   - Exact division: input length is a clean multiple of pixel_width
//   - Inexact division: remainder distributed across leading pixels
//   - Sparse: input shorter than pixel_width (passthrough + tail padding)
//   - Single-pixel output: one (min, max) covering the whole input
//   - Empty input: returns empty envelope; pixel_width=0 throws
//   - **Glitch survival**: 5-sample-wide narrow glitch in a 10000-sample
//     square wave reduced to 100 pixels — the glitch's peak amplitude
//     must show up in its corresponding pixel's max.
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)`.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>

#include <sw/dsp/instrument/display_envelope.hpp>

using namespace sw::dsp::instrument;

#define REQUIRE(cond) \
	do { if (!(cond)) throw std::runtime_error( \
		std::string("test failed: ") + #cond + \
		" at " __FILE__ ":" + std::to_string(__LINE__)); } while (0)

// ============================================================================
// Exact division
// ============================================================================

void test_exact_division() {
	// 8 samples, 4 pixels — each pixel covers 2 samples.
	std::array<int, 8> in = {1, 5, 2, 6, 7, 3, 8, 4};
	auto env = render_envelope<int>(std::span<const int>{in}, 4);
	REQUIRE(env.mins.size() == 4);
	REQUIRE(env.maxs.size() == 4);
	// pixel 0: {1, 5} → (1, 5)
	// pixel 1: {2, 6} → (2, 6)
	// pixel 2: {7, 3} → (3, 7)
	// pixel 3: {8, 4} → (4, 8)
	REQUIRE(env.mins[0] == 1 && env.maxs[0] == 5);
	REQUIRE(env.mins[1] == 2 && env.maxs[1] == 6);
	REQUIRE(env.mins[2] == 3 && env.maxs[2] == 7);
	REQUIRE(env.mins[3] == 4 && env.maxs[3] == 8);
	std::cout << "  exact_division: passed\n";
}

// ============================================================================
// Inexact division — remainder distributed across leading pixels
// ============================================================================

void test_inexact_division() {
	// 7 samples, 3 pixels — base_span=2, remainder=1
	// Leading 1 pixel gets span=3, trailing 2 pixels get span=2.
	// Pixel 0: samples[0..2]   = {1,2,3} → (1, 3)
	// Pixel 1: samples[3..4]   = {4, 5}  → (4, 5)
	// Pixel 2: samples[5..6]   = {6, 7}  → (6, 7)
	std::array<int, 7> in = {1, 2, 3, 4, 5, 6, 7};
	auto env = render_envelope<int>(std::span<const int>{in}, 3);
	REQUIRE(env.mins.size() == 3 && env.maxs.size() == 3);
	REQUIRE(env.mins[0] == 1 && env.maxs[0] == 3);
	REQUIRE(env.mins[1] == 4 && env.maxs[1] == 5);
	REQUIRE(env.mins[2] == 6 && env.maxs[2] == 7);
	std::cout << "  inexact_division: passed\n";
}

void test_inexact_division_large_remainder() {
	// 11 samples, 4 pixels — base_span=2, remainder=3
	// Leading 3 pixels get span=3 (cover 3 samples each), trailing 1
	// pixel gets span=2. Verifies the extra-sample distribution is
	// correctly applied to ALL `remainder` leading pixels, not just one.
	// Pixel 0: samples[0..2]   = {0,1,2}   → (0, 2)
	// Pixel 1: samples[3..5]   = {3,4,5}   → (3, 5)
	// Pixel 2: samples[6..8]   = {6,7,8}   → (6, 8)
	// Pixel 3: samples[9..10]  = {9,10}    → (9, 10)
	std::array<int, 11> in = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	auto env = render_envelope<int>(std::span<const int>{in}, 4);
	REQUIRE(env.mins.size() == 4 && env.maxs.size() == 4);
	REQUIRE(env.mins[0] == 0 && env.maxs[0] == 2);
	REQUIRE(env.mins[1] == 3 && env.maxs[1] == 5);
	REQUIRE(env.mins[2] == 6 && env.maxs[2] == 8);
	REQUIRE(env.mins[3] == 9 && env.maxs[3] == 10);
	std::cout << "  inexact_division_large_remainder: passed\n";
}

// ============================================================================
// Sparse: input shorter than pixel_width
// ============================================================================

void test_sparse_input_shorter() {
	// 3 samples → 8 pixels. First 3 pixels each get one sample (min == max);
	// trailing 5 pixels are padded with the last sample's value.
	std::array<int, 3> in = {10, 20, 30};
	auto env = render_envelope<int>(std::span<const int>{in}, 8);
	REQUIRE(env.mins.size() == 8 && env.maxs.size() == 8);
	REQUIRE(env.mins[0] == 10 && env.maxs[0] == 10);
	REQUIRE(env.mins[1] == 20 && env.maxs[1] == 20);
	REQUIRE(env.mins[2] == 30 && env.maxs[2] == 30);
	for (std::size_t p = 3; p < 8; ++p) {
		REQUIRE(env.mins[p] == 30 && env.maxs[p] == 30);
	}
	std::cout << "  sparse_input_shorter: passed\n";
}

void test_sparse_equal_width() {
	// N == pixel_width — each sample becomes one pixel; no padding.
	std::array<int, 4> in = {7, 8, 9, 10};
	auto env = render_envelope<int>(std::span<const int>{in}, 4);
	REQUIRE(env.mins.size() == 4);
	for (std::size_t i = 0; i < 4; ++i) {
		REQUIRE(env.mins[i] == in[i]);
		REQUIRE(env.maxs[i] == in[i]);
	}
	std::cout << "  sparse_equal_width: passed\n";
}

// ============================================================================
// Single-pixel output
// ============================================================================

void test_single_pixel() {
	// pixel_width=1: one (min, max) covering everything.
	std::array<int, 5> in = {3, 7, 1, 9, 5};
	auto env = render_envelope<int>(std::span<const int>{in}, 1);
	REQUIRE(env.mins.size() == 1 && env.maxs.size() == 1);
	REQUIRE(env.mins[0] == 1);
	REQUIRE(env.maxs[0] == 9);
	std::cout << "  single_pixel: passed\n";
}

// ============================================================================
// Edge cases: empty input, zero pixel width
// ============================================================================

void test_empty_input() {
	std::span<const int> empty;
	auto env = render_envelope<int>(empty, 100);
	// Empty envelope — explicit signal that there's nothing to render.
	REQUIRE(env.mins.size() == 0);
	REQUIRE(env.maxs.size() == 0);
	std::cout << "  empty_input: passed\n";
}

void test_zero_pixel_width_throws() {
	std::array<int, 4> in = {1, 2, 3, 4};
	bool threw = false;
	try {
		(void)render_envelope<int>(std::span<const int>{in}, 0);
	} catch (const std::invalid_argument&) { threw = true; }
	REQUIRE(threw);
	std::cout << "  zero_pixel_width_throws: passed\n";
}

// ============================================================================
// GLITCH-SURVIVAL TEST — the headline acceptance criterion
//
// 50 MHz square wave at fs=1 GSPS = 20 samples/period, 10000 samples
// total. Buried in a low phase: a 5-sample-wide narrow positive glitch
// (peak +1.5, square wave ±1.0). Render to 100 pixels — input/output
// ratio is 100:1 and the glitch is only 5 samples wide, so a generic
// averaging or polyphase decimator would average it out below the
// noise floor. Min/max envelope keeps the peak at 1.5 in whichever
// pixel the glitch happened to fall in.
// ============================================================================

constexpr std::size_t kStreamSize = 10000;

std::array<float, kStreamSize> make_square_wave_with_glitch() {
	std::array<float, kStreamSize> stream{};
	for (std::size_t n = 0; n < kStreamSize; ++n) {
		stream[n] = ((n / 10) % 2 == 0) ? 1.0f : -1.0f;
	}
	// Glitch in samples [5010..5014] — within a low-phase region.
	for (std::size_t n = 5010; n < 5015; ++n) {
		stream[n] = 1.5f;
	}
	return stream;
}

void test_glitch_survives_at_100_pixels() {
	const auto stream = make_square_wave_with_glitch();
	const float glitch_peak = 1.5f;

	auto env = render_envelope<float>(
		std::span<const float>{stream}, /*pixel_width=*/100);
	REQUIRE(env.mins.size() == 100);
	REQUIRE(env.maxs.size() == 100);

	// Find the peak of the max stream — should be exactly the glitch
	// amplitude. With averaging/polyphase, this would have averaged
	// down toward (5*1.5 + 95*-1)/100 ≈ -0.875 — far below 1.0,
	// invisible against the square wave's ±1.0.
	float observed_peak = -1e9f;
	std::size_t glitch_pixel = 0;
	for (std::size_t i = 0; i < env.maxs.size(); ++i) {
		if (env.maxs[i] > observed_peak) {
			observed_peak = env.maxs[i];
			glitch_pixel = i;
		}
	}
	if (!(std::abs(observed_peak - glitch_peak) < 1e-5f))
		throw std::runtime_error(
			"glitch lost: observed_peak=" + std::to_string(observed_peak) +
			" want=" + std::to_string(glitch_peak));

	// Sanity: glitch was at samples ~5010-5014 in a 10000-sample stream
	// reduced to 100 pixels (each pixel covers 100 samples). So the
	// glitch should land in pixel 50 (samples 5000..5099).
	if (!(glitch_pixel == 50))
		throw std::runtime_error(
			"glitch in pixel " + std::to_string(glitch_pixel) +
			", expected pixel 50");
	std::cout << "  glitch_survives_at_100_pixels: passed (pixel "
	          << glitch_pixel << ", peak=" << observed_peak << ")\n";
}

// ============================================================================
// main
// ============================================================================

int main() {
	try {
		std::cout << "test_instrument_display_envelope\n";

		test_exact_division();
		test_inexact_division();
		test_inexact_division_large_remainder();
		test_sparse_input_shorter();
		test_sparse_equal_width();
		test_single_pixel();
		test_empty_input();
		test_zero_pixel_width_throws();
		test_glitch_survives_at_100_pixels();

		std::cout << "all tests passed\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "FAILED: " << ex.what() << "\n";
		return 1;
	}
}
