// test_spectrum_waterfall.cpp: tests for the waterfall buffer (circular
// 2D trace memory for spectrogram displays).
//
// Coverage:
//   - Construction validates non-zero num_bins / num_frames
//   - push_frame: length mismatch throws
//   - Partial fill: num_frames_filled() grows up to capacity, then
//     saturates
//   - Wrap point: after capacity + k pushes, the oldest stored frame
//     is the (k+1)-th push (the first k were evicted)
//   - frame_at: zero-copy single-frame access; out-of-range throws
//   - last_frames: chronological order, count clamping at fill,
//     count==0 returns empty span
//   - clear() resets state and lets fresh pushes start over
//   - Mixed-precision storage: float frames stored and retrieved
//     bit-exactly (this is pure storage; no arithmetic)
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
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <sw/dsp/spectrum/waterfall_buffer.hpp>

using namespace sw::dsp::spectrum;
using W = WaterfallBuffer<double>;

#define REQUIRE(cond) \
	do { if (!(cond)) throw std::runtime_error( \
		std::string("test failed: ") + #cond + \
		" at " __FILE__ ":" + std::to_string(__LINE__)); } while (0)

// Helper: build a frame of length B where each sample equals
// frame_id * 100 + bin (gives unique values for verification).
static std::vector<double> make_frame(std::size_t B, double frame_id) {
	std::vector<double> f(B);
	for (std::size_t i = 0; i < B; ++i)
		f[i] = frame_id * 100.0 + static_cast<double>(i);
	return f;
}

// ============================================================================
// Construction
// ============================================================================

void test_construction_validation() {
	bool t1 = false, t2 = false, t3 = false;
	try { W(0, 4); } catch (const std::invalid_argument&) { t1 = true; }
	REQUIRE(t1);
	try { W(8, 0); } catch (const std::invalid_argument&) { t2 = true; }
	REQUIRE(t2);

	// Multiplication overflow guard: num_bins * num_frames must fit in
	// size_t. Use values whose product overflows on any 64-bit system.
	const std::size_t huge = std::numeric_limits<std::size_t>::max() / 2 + 1;
	try { W(huge, 4); } catch (const std::length_error&) { t3 = true; }
	REQUIRE(t3);

	// Valid construction: empty initial state.
	W w(8, 4);
	REQUIRE(w.num_bins() == 8);
	REQUIRE(w.num_frames_capacity() == 4);
	REQUIRE(w.num_frames_filled() == 0);
	std::cout << "  construction_validation: passed\n";
}

// ============================================================================
// push_frame: length mismatch
// ============================================================================

void test_push_frame_length_mismatch_throws() {
	W w(8, 4);
	std::array<double, 7> wrong{};
	bool threw = false;
	try { w.push_frame(std::span<const double>{wrong}); }
	catch (const std::invalid_argument&) { threw = true; }
	REQUIRE(threw);
	std::cout << "  push_frame_length_mismatch_throws: passed\n";
}

// ============================================================================
// Partial fill: num_frames_filled grows then saturates
// ============================================================================

void test_partial_fill_saturates_at_capacity() {
	const std::size_t B = 4;
	const std::size_t C = 3;
	W w(B, C);

	for (std::size_t k = 0; k < C; ++k) {
		auto f = make_frame(B, static_cast<double>(k));
		w.push_frame(std::span<const double>{f});
		REQUIRE(w.num_frames_filled() == k + 1);
	}
	// Capacity reached. Push 5 more — num_frames_filled stays at C.
	for (std::size_t k = 0; k < 5; ++k) {
		auto f = make_frame(B, static_cast<double>(C + k));
		w.push_frame(std::span<const double>{f});
		REQUIRE(w.num_frames_filled() == C);
	}
	std::cout << "  partial_fill_saturates_at_capacity: passed\n";
}

// ============================================================================
// Wrap point: oldest survives correctly
// ============================================================================

void test_wrap_oldest_survivor() {
	// Capacity 3. Push frames with ids 0..6. After 7 pushes, the
	// stored frames are the last 3: ids 4, 5, 6. frame_at(0) = id 4,
	// frame_at(1) = id 5, frame_at(2) = id 6.
	const std::size_t B = 4;
	const std::size_t C = 3;
	W w(B, C);
	for (std::size_t id = 0; id < 7; ++id) {
		auto f = make_frame(B, static_cast<double>(id));
		w.push_frame(std::span<const double>{f});
	}
	REQUIRE(w.num_frames_filled() == C);
	for (std::size_t i = 0; i < C; ++i) {
		auto f = w.frame_at(i);
		const double expected_id = static_cast<double>(4 + i);
		REQUIRE(f[0] == expected_id * 100.0 + 0.0);
		REQUIRE(f[B - 1] == expected_id * 100.0 + static_cast<double>(B - 1));
	}
	std::cout << "  wrap_oldest_survivor: passed\n";
}

// ============================================================================
// frame_at: out-of-range throws
// ============================================================================

void test_frame_at_out_of_range_throws() {
	W w(4, 3);
	bool threw_empty = false, threw_full = false, threw_far = false;

	// Empty: any index throws.
	try { (void)w.frame_at(0); } catch (const std::out_of_range&) { threw_empty = true; }
	REQUIRE(threw_empty);

	// Push 2 frames. frame_at(0) and frame_at(1) work; frame_at(2)
	// (just past the filled count) and frame_at(7) (well past it,
	// even past capacity) both throw.
	auto f = make_frame(4, 1.0);
	w.push_frame(std::span<const double>{f});
	w.push_frame(std::span<const double>{f});
	(void)w.frame_at(0);
	(void)w.frame_at(1);
	try { (void)w.frame_at(2); } catch (const std::out_of_range&) { threw_full = true; }
	REQUIRE(threw_full);
	try { (void)w.frame_at(7); } catch (const std::out_of_range&) { threw_far = true; }
	REQUIRE(threw_far);
	std::cout << "  frame_at_out_of_range_throws: passed\n";
}

// ============================================================================
// last_frames: chronological order + clamping
// ============================================================================

void test_last_frames_chronological() {
	const std::size_t B = 4;
	const std::size_t C = 5;
	W w(B, C);
	// Push 7 frames (ids 0..6). Stored: ids 2..6.
	for (std::size_t id = 0; id < 7; ++id) {
		auto f = make_frame(B, static_cast<double>(id));
		w.push_frame(std::span<const double>{f});
	}
	// last_frames(3) should return ids 4, 5, 6 in that order.
	auto view = w.last_frames(3);
	REQUIRE(view.size() == 3 * B);
	for (std::size_t k = 0; k < 3; ++k) {
		const double expected_id = static_cast<double>(4 + k);
		for (std::size_t i = 0; i < B; ++i) {
			const std::size_t flat = k * B + i;
			REQUIRE(view[flat] == expected_id * 100.0 + static_cast<double>(i));
		}
	}
	std::cout << "  last_frames_chronological: passed\n";
}

void test_last_frames_clamping() {
	const std::size_t B = 4;
	const std::size_t C = 5;
	W w(B, C);
	// Push only 2 frames; ask for 10. Should return 2.
	for (std::size_t id = 0; id < 2; ++id) {
		auto f = make_frame(B, static_cast<double>(id));
		w.push_frame(std::span<const double>{f});
	}
	auto view = w.last_frames(10);
	REQUIRE(view.size() == 2 * B);
	// Frame 0 first, frame 1 next.
	REQUIRE(view[0]                == 0.0 * 100.0 + 0.0);
	REQUIRE(view[B + 0]            == 1.0 * 100.0 + 0.0);

	// last_frames(0) returns empty.
	auto empty = w.last_frames(0);
	REQUIRE(empty.empty());
	std::cout << "  last_frames_clamping: passed\n";
}

// ============================================================================
// clear() resets state
// ============================================================================

void test_clear_resets_state() {
	W w(4, 3);
	for (std::size_t id = 0; id < 5; ++id) {
		auto f = make_frame(4, static_cast<double>(id));
		w.push_frame(std::span<const double>{f});
	}
	REQUIRE(w.num_frames_filled() == 3);

	w.clear();
	REQUIRE(w.num_frames_filled() == 0);
	REQUIRE(w.last_frames(5).empty());

	// Push fresh frame: shows up at index 0.
	auto f = make_frame(4, 99.0);
	w.push_frame(std::span<const double>{f});
	REQUIRE(w.num_frames_filled() == 1);
	auto frame = w.frame_at(0);
	REQUIRE(frame[0] == 99.0 * 100.0 + 0.0);
	std::cout << "  clear_resets_state: passed\n";
}

// ============================================================================
// Mixed-precision storage: round-trip exactness
// ============================================================================

void test_float_storage_round_trip() {
	using WF = WaterfallBuffer<float>;
	WF wf(4, 3);
	std::array<float, 4> a = {1.5f, 2.25f, -3.125f, 4.75f};
	wf.push_frame(std::span<const float>{a});
	auto f = wf.frame_at(0);
	for (std::size_t i = 0; i < 4; ++i) REQUIRE(f[i] == a[i]);
	std::cout << "  float_storage_round_trip: passed\n";
}

// ============================================================================
// main
// ============================================================================

int main() {
	try {
		std::cout << "test_spectrum_waterfall\n";

		test_construction_validation();
		test_push_frame_length_mismatch_throws();
		test_partial_fill_saturates_at_capacity();
		test_wrap_oldest_survivor();
		test_frame_at_out_of_range_throws();
		test_last_frames_chronological();
		test_last_frames_clamping();
		test_clear_resets_state();
		test_float_storage_round_trip();

		std::cout << "all tests passed\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "FAILED: " << ex.what() << "\n";
		return 1;
	}
}
