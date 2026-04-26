// test_instrument_ring_buffer.cpp: tests for the trigger ring buffer +
// segmented capture primitives.
//
// Coverage:
//   - TriggerRingBuffer: basic capture, partial pre-fill, post=0, pre=0,
//     ring wrap-around, late trigger after long pre-fill, drop-after-complete,
//     rearm preserves ring, multiple sequential captures
//   - SegmentedCapture: back-to-back captures, max_segments cap,
//     accessor bounds checking, integration with the trigger primitives
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)`.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <array>
#include <initializer_list>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>

#include <sw/dsp/instrument/ring_buffer.hpp>
#include <sw/dsp/instrument/trigger.hpp>

using namespace sw::dsp::instrument;

#define REQUIRE(cond) \
	do { if (!(cond)) throw std::runtime_error( \
		std::string("test failed: ") + #cond + \
		" at " __FILE__ ":" + std::to_string(__LINE__)); } while (0)

// ============================================================================
// Helpers
// ============================================================================

// Compare a captured span to an expected sequence. Takes the expected
// sequence as a std::initializer_list so brace-init call sites stay
// concise without forcing std::vector heap allocations.
template <class T>
void require_segment_equals(std::span<const T> got,
                            std::initializer_list<T> expected,
                            const char* tag) {
	if (got.size() != expected.size())
		throw std::runtime_error(std::string(tag) +
			": segment size mismatch, got=" +
			std::to_string(got.size()) + " want=" +
			std::to_string(expected.size()));
	std::size_t i = 0;
	for (const T& want : expected) {
		if (got[i] != want)
			throw std::runtime_error(std::string(tag) +
				": segment[" + std::to_string(i) + "] = " +
				std::to_string(got[i]) + " want " +
				std::to_string(want));
		++i;
	}
}

// ============================================================================
// TriggerRingBuffer — happy path
// ============================================================================

void test_basic_capture() {
	// pre=4, post=3 — total segment = 4+1+3 = 8
	TriggerRingBuffer<int> buf(4, 3);
	for (int i = 1; i <= 4; ++i) buf.push(i);  // ring fills with 1,2,3,4
	REQUIRE(!buf.capture_complete());
	buf.push_trigger(99);                       // segment so far: 1,2,3,4,99
	REQUIRE(!buf.capture_complete());
	buf.push(100);
	buf.push(101);
	REQUIRE(!buf.capture_complete());
	buf.push(102);                              // post-trigger now full
	REQUIRE(buf.capture_complete());
	require_segment_equals<int>(buf.captured_segment(),
	                            {1, 2, 3, 4, 99, 100, 101, 102},
	                            "basic_capture");
	std::cout << "  basic_capture: passed\n";
}

void test_ring_wraparound() {
	// pre=3 — push 7 samples then trigger; ring should hold the last 3.
	TriggerRingBuffer<int> buf(3, 2);
	for (int i = 1; i <= 7; ++i) buf.push(i);   // ring: 5,6,7
	buf.push_trigger(99);
	buf.push(100);
	buf.push(101);
	REQUIRE(buf.capture_complete());
	require_segment_equals<int>(buf.captured_segment(),
	                            {5, 6, 7, 99, 100, 101},
	                            "ring_wraparound");
	std::cout << "  ring_wraparound: passed\n";
}

// ============================================================================
// Partial pre-fill: trigger arrives before pre-trigger buffer is full
// ============================================================================

void test_partial_prefill() {
	TriggerRingBuffer<int> buf(5, 2);
	buf.push(10);
	buf.push(20);
	// Only 2 pre-trigger samples accumulated; trigger arrives early.
	buf.push_trigger(99);
	buf.push(100);
	buf.push(101);
	REQUIRE(buf.capture_complete());
	// Captured segment: 2 pre-trigger + trigger + 2 post = 5 total
	require_segment_equals<int>(buf.captured_segment(),
	                            {10, 20, 99, 100, 101},
	                            "partial_prefill");
	std::cout << "  partial_prefill: passed\n";
}

void test_immediate_trigger() {
	// Trigger on the very first sample — no pre-context at all.
	TriggerRingBuffer<int> buf(8, 3);
	buf.push_trigger(42);
	buf.push(1);
	buf.push(2);
	buf.push(3);
	REQUIRE(buf.capture_complete());
	require_segment_equals<int>(buf.captured_segment(),
	                            {42, 1, 2, 3},
	                            "immediate_trigger");
	std::cout << "  immediate_trigger: passed\n";
}

// ============================================================================
// Edge cases: zero-size pre or post
// ============================================================================

void test_post_zero() {
	// post_size=0: capture completes immediately on push_trigger.
	TriggerRingBuffer<int> buf(3, 0);
	buf.push(1);
	buf.push(2);
	buf.push(3);
	buf.push_trigger(99);
	REQUIRE(buf.capture_complete());
	require_segment_equals<int>(buf.captured_segment(),
	                            {1, 2, 3, 99},
	                            "post_zero");
	std::cout << "  post_zero: passed\n";
}

void test_pre_zero() {
	// pre_size=0: trigger is the first sample of every capture.
	TriggerRingBuffer<int> buf(0, 3);
	buf.push(1);   // dropped — no ring
	buf.push(2);   // dropped
	buf.push_trigger(99);
	buf.push(100);
	buf.push(101);
	buf.push(102);
	REQUIRE(buf.capture_complete());
	require_segment_equals<int>(buf.captured_segment(),
	                            {99, 100, 101, 102},
	                            "pre_zero");
	std::cout << "  pre_zero: passed\n";
}

void test_pre_zero_post_zero() {
	// degenerate: capture is just the trigger sample
	TriggerRingBuffer<int> buf(0, 0);
	buf.push_trigger(42);
	REQUIRE(buf.capture_complete());
	require_segment_equals<int>(buf.captured_segment(), {42}, "pre_zero_post_zero");
	std::cout << "  pre_zero_post_zero: passed\n";
}

// ============================================================================
// Drop-after-complete and rearm semantics
// ============================================================================

void test_drop_after_complete() {
	TriggerRingBuffer<int> buf(2, 1);
	buf.push(1); buf.push(2);
	buf.push_trigger(99);
	buf.push(100);
	REQUIRE(buf.capture_complete());
	// Further pushes are silently dropped
	buf.push(200);
	buf.push(201);
	// Captured segment unchanged
	require_segment_equals<int>(buf.captured_segment(),
	                            {1, 2, 99, 100},
	                            "drop_after_complete");
	std::cout << "  drop_after_complete: passed\n";
}

void test_rearm_preserves_ring() {
	// After rearm(), the pre-trigger ring should still contain the previous
	// content so a near-immediate re-trigger doesn't lose pre-context.
	TriggerRingBuffer<int> buf(3, 2);
	buf.push(1); buf.push(2); buf.push(3);
	buf.push_trigger(99); buf.push(100); buf.push(101);
	REQUIRE(buf.capture_complete());
	buf.rearm();
	REQUIRE(!buf.capture_complete());
	// Trigger again WITHOUT new push() calls — pre-ring should still have
	// the post-trigger samples that flowed through it (100, 101) plus the
	// trigger sample that was rotated into the ring? No — the ring is only
	// fed by push(), not by post-trigger samples in the segment buffer.
	// So after the first capture, the ring still has 1,2,3.
	buf.push_trigger(200);
	buf.push(201); buf.push(202);
	REQUIRE(buf.capture_complete());
	require_segment_equals<int>(buf.captured_segment(),
	                            {1, 2, 3, 200, 201, 202},
	                            "rearm_preserves_ring");
	std::cout << "  rearm_preserves_ring: passed\n";
}

void test_reset_clears_everything() {
	TriggerRingBuffer<int> buf(3, 2);
	buf.push(1); buf.push(2); buf.push(3);
	buf.push_trigger(99); buf.push(100); buf.push(101);
	REQUIRE(buf.capture_complete());
	buf.reset();
	// Should be back to a fresh PreFill state.
	REQUIRE(!buf.capture_complete());
	buf.push_trigger(42);
	buf.push(43); buf.push(44);
	REQUIRE(buf.capture_complete());
	// Only the trigger + post — ring was cleared
	require_segment_equals<int>(buf.captured_segment(),
	                            {42, 43, 44},
	                            "reset_clears_everything");
	std::cout << "  reset_clears_everything: passed\n";
}

// ============================================================================
// Multiple sequential captures
// ============================================================================

void test_three_sequential_captures() {
	TriggerRingBuffer<int> buf(2, 1);
	for (int round = 0; round < 3; ++round) {
		buf.push(round * 10 + 1);
		buf.push(round * 10 + 2);
		buf.push_trigger(round * 10 + 9);
		buf.push(round * 10 + 3);
		REQUIRE(buf.capture_complete());
		auto seg = buf.captured_segment();
		REQUIRE(seg.size() == 4);
		REQUIRE(seg[2] == round * 10 + 9);
		buf.rearm();
	}
	std::cout << "  three_sequential_captures: passed\n";
}

// ============================================================================
// SegmentedCapture
// ============================================================================

void test_segmented_three() {
	SegmentedCapture<int> sc(/*pre=*/2, /*post=*/1, /*max_segments=*/3);

	// Capture 1
	sc.push(1); sc.push(2);
	sc.push_trigger(101);
	sc.push(3);
	REQUIRE(!sc.capture_complete());
	REQUIRE(sc.segment_count() == 1);

	// Capture 2 (note: ring still holds [1, 2] -> [2, ?] after the previous
	// capture's tail; let's just push fresh data)
	sc.push(4); sc.push(5);
	sc.push_trigger(102);
	sc.push(6);
	REQUIRE(!sc.capture_complete());
	REQUIRE(sc.segment_count() == 2);

	// Capture 3
	sc.push(7); sc.push(8);
	sc.push_trigger(103);
	sc.push(9);
	REQUIRE(sc.capture_complete());
	REQUIRE(sc.segment_count() == 3);

	// Check each segment's trigger sample lives at index 2
	REQUIRE(sc.segment(0)[2] == 101);
	REQUIRE(sc.segment(1)[2] == 102);
	REQUIRE(sc.segment(2)[2] == 103);

	// Subsequent pushes should be ignored
	sc.push(999);
	sc.push_trigger(998);
	REQUIRE(sc.segment_count() == 3);

	std::cout << "  segmented_three: passed\n";
}

void test_segmented_max_segments_zero_throws() {
	bool threw = false;
	try { SegmentedCapture<int>(2, 1, 0); }
	catch (const std::invalid_argument&) { threw = true; }
	REQUIRE(threw);
	std::cout << "  segmented_max_segments_zero_throws: passed\n";
}

void test_segmented_segment_oor_throws() {
	SegmentedCapture<int> sc(1, 1, 2);
	sc.push(0); sc.push_trigger(1); sc.push(2);
	REQUIRE(sc.segment_count() == 1);
	bool threw = false;
	try { (void)sc.segment(5); }
	catch (const std::out_of_range&) { threw = true; }
	REQUIRE(threw);
	std::cout << "  segmented_segment_oor_throws: passed\n";
}

void test_segmented_reset() {
	SegmentedCapture<int> sc(1, 1, 2);
	sc.push(0); sc.push_trigger(1); sc.push(2);
	REQUIRE(sc.segment_count() == 1);
	sc.reset();
	REQUIRE(sc.segment_count() == 0);
	REQUIRE(!sc.capture_complete());
	std::cout << "  segmented_reset: passed\n";
}

// ============================================================================
// Integration with the trigger primitives (#140)
// ============================================================================

void test_integration_with_edge_trigger() {
	// Drive a real EdgeTrigger and a TriggerRingBuffer side by side.
	// Stream long enough to: build pre-trigger ring, fire the trigger,
	// fill the post-trigger region.  Pre=8, Post=16 → need at least
	// (8 + 1 + 16) = 25 samples after some lead-in. 64 is comfortable.
	EdgeTrigger<float> trig(/*level=*/0.5f, Slope::Rising);
	TriggerRingBuffer<float> buf(/*pre=*/8, /*post=*/16);

	std::array<float, 64> stream{};
	for (std::size_t i = 0; i < stream.size(); ++i) {
		stream[i] = static_cast<float>(i) / 64.0f;  // ramp 0 -> ~1
	}

	for (float x : stream) {
		if (trig.process(x)) {
			buf.push_trigger(x);
		} else {
			buf.push(x);
		}
		if (buf.capture_complete()) break;
	}

	REQUIRE(buf.capture_complete());
	auto seg = buf.captured_segment();
	REQUIRE(seg.size() == 25);   // 8 pre + 1 trigger + 16 post

	// The trigger sample sits at index 8 (after 8 pre samples). EdgeTrigger
	// fires on the first sample > level after at least one sample < level,
	// so it's the first ramp value strictly greater than 0.5.
	REQUIRE(seg[8] > 0.5f);
	REQUIRE(seg[7] <= 0.5f);     // last pre-trigger sample is below threshold
	std::cout << "  integration_with_edge_trigger: passed\n";
}

// ============================================================================
// main
// ============================================================================

int main() {
	try {
		std::cout << "test_instrument_ring_buffer\n";

		test_basic_capture();
		test_ring_wraparound();
		test_partial_prefill();
		test_immediate_trigger();
		test_post_zero();
		test_pre_zero();
		test_pre_zero_post_zero();
		test_drop_after_complete();
		test_rearm_preserves_ring();
		test_reset_clears_everything();
		test_three_sequential_captures();

		test_segmented_three();
		test_segmented_max_segments_zero_throws();
		test_segmented_segment_oor_throws();
		test_segmented_reset();

		test_integration_with_edge_trigger();

		std::cout << "all tests passed\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "FAILED: " << ex.what() << "\n";
		return 1;
	}
}
