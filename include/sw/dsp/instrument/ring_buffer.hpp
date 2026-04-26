#pragma once
// ring_buffer.hpp: Pre/post-trigger ring buffer + segmented memory capture
// for instrument-style data acquisition.
//
// `TriggerRingBuffer<SampleScalar>`
//   A circular sample buffer that captures `pre_trigger_samples` of context
//   before a trigger event, the event sample itself, and
//   `post_trigger_samples` after. The captured segment is available via
//   `captured_segment()` once `capture_complete()` returns true.
//
// `SegmentedCapture<SampleScalar>`
//   Stores up to `max_segments` consecutive captures back-to-back without
//   re-arm dead time. Used by oscilloscopes for capturing rare events at
//   high duty cycle.
//
// Both classes integrate with the trigger primitives from
// `<sw/dsp/instrument/trigger.hpp>`: a typical pipeline calls
// `trigger.process(x)` first, then either `buf.push(x)` or
// `buf.push_trigger(x)` based on whether the trigger fired.
//
// The buffer is precision-insensitive (it only stores and returns samples,
// no arithmetic). The `SampleScalar` parameter exists for type-homogeneity
// with the upstream stream and downstream consumers.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp::instrument {

// =============================================================================
// TriggerRingBuffer
//
// Lifecycle (state machine):
//
//   ┌─────────────┐   push() x pre_size      ┌────────────────┐
//   │  PreFill    │ ───────────────────────► │  Armed         │
//   │             │                          │  (waiting for  │
//   │  Building   │                          │   trigger)     │
//   │  pre-trigger│                          │                │
//   │  context    │                          │                │
//   └─────────────┘                          └────────────────┘
//                                                    │
//                                            push_trigger(x)
//                                                    │
//                                                    ▼
//                                     ┌──────────────────────────────┐
//                                     │  Capturing                   │
//                                     │  Filling post_size samples   │
//                                     │  after the trigger event     │
//                                     └──────────────────────────────┘
//                                                    │
//                                              post_size pushes later
//                                                    │
//                                                    ▼
//                                          ┌────────────────────┐
//                                          │  Complete          │
//                                          │  captured_segment  │
//                                          │  is available      │
//                                          └────────────────────┘
//                                                    │
//                                                  rearm()
//                                                    │
//                                                    ▼
//                                            (back to PreFill)
//
// Notes:
//   - During PreFill, push() simply rotates the pre-trigger ring; once
//     pre_size samples have been pushed, the next call to push_trigger()
//     can capture full context.
//   - Calling push_trigger() during PreFill is allowed: the captured
//     segment will be shorter than the maximum (it just contains whatever
//     pre-context has been accumulated so far). This corresponds to
//     "trigger arrived before pre-trigger buffer was full" in real scopes.
//   - The captured segment is returned in time order: oldest pre-trigger
//     sample first, then the trigger sample, then the post-trigger samples.
// =============================================================================
template <DspScalar SampleScalar>
class TriggerRingBuffer {
public:
	using sample_scalar = SampleScalar;

	// pre_trigger_samples: capacity of the pre-trigger context buffer
	// post_trigger_samples: number of samples to capture after the trigger
	//
	// Either may be zero (a "trigger-only" capture sets pre=0; a
	// "pre-history-only" capture sets post=0). Both being zero is allowed
	// but unusual — captured_segment() will contain just the trigger sample.
	TriggerRingBuffer(std::size_t pre_trigger_samples,
	                  std::size_t post_trigger_samples)
		: pre_size_(pre_trigger_samples),
		  post_size_(post_trigger_samples),
		  // Pre-allocate the ring and segment buffers via the size ctor
		  // (mtl::vec::dense_vector does not support .resize() — fixed
		  // dimension sizes are decided at construction).
		  ring_(pre_size_),
		  segment_(pre_size_ + 1 + post_size_) {
		reset_state();
	}

	// Push a non-trigger sample. The sample joins the pre-trigger ring.
	// During Capturing, the sample joins the post-trigger region instead.
	// During Complete, additional pushes are silently dropped (caller
	// should rearm() before continuing).
	void push(SampleScalar x) {
		switch (state_) {
			case State::PreFill:
			case State::Armed:
				push_to_ring(x);
				break;
			case State::Capturing:
				segment_[segment_pos_++] = x;
				if (segment_pos_ >= target_length_) {
					captured_length_ = segment_pos_;
					state_ = State::Complete;
				}
				break;
			case State::Complete:
				// dropped; caller should rearm before pushing more
				break;
		}
	}

	// Push the sample that caused the trigger. Subsequent post_trigger_samples
	// pushes will be captured before capture_complete() returns true.
	//
	// If called during PreFill (pre-trigger buffer not yet full), the
	// captured segment will start with however many pre-context samples
	// have been accumulated so far. The total length of captured_segment()
	// reflects the actual capture, not the maximum.
	//
	// Calling push_trigger() during Capturing or Complete is silently
	// ignored (the trigger is already past); rearm first if you want to
	// capture again.
	void push_trigger(SampleScalar x) {
		if (state_ == State::Capturing || state_ == State::Complete) return;

		// Drain the ring into the segment in time order (oldest first).
		// The ring's logical content is `pre_count_` samples ending at
		// (write_pos_ - 1) mod pre_size_. We unwind that into segment_[0..].
		segment_pos_ = 0;
		if (pre_size_ > 0 && pre_count_ > 0) {
			std::size_t start = (write_pos_ + pre_size_ - pre_count_) % pre_size_;
			for (std::size_t i = 0; i < pre_count_; ++i) {
				segment_[segment_pos_++] = ring_[(start + i) % pre_size_];
			}
		}
		// Trigger sample
		segment_[segment_pos_++] = x;

		// Compute the target length for THIS capture: it depends on how
		// much pre-context we actually had, which may be less than
		// pre_size_ when the trigger arrived during PreFill.
		target_length_ = segment_pos_ + post_size_;

		// If post_size_ is zero, we're complete immediately.
		if (post_size_ == 0) {
			captured_length_ = segment_pos_;
			state_ = State::Complete;
		} else {
			state_ = State::Capturing;
		}
	}

	// True once the post-trigger region is full (or zero).
	bool capture_complete() const {
		return state_ == State::Complete;
	}

	// Returns the captured pre+trigger+post segment. Only valid when
	// capture_complete() is true; otherwise returns an empty span.
	std::span<const SampleScalar> captured_segment() const {
		if (state_ != State::Complete)
			return std::span<const SampleScalar>{};
		return std::span<const SampleScalar>(segment_.data(), captured_length_);
	}

	// Discard the captured segment and resume PreFill. The pre-trigger ring
	// retains its content so a near-immediate re-trigger doesn't lose
	// pre-context. (Real scopes work this way too: rearm doesn't blank the
	// front-end memory.)
	void rearm() {
		state_       = pre_count_ >= pre_size_ ? State::Armed : State::PreFill;
		segment_pos_ = 0;
		captured_length_ = 0;
	}

	// Discard everything (ring, captured segment) and return to a fresh
	// PreFill state. Useful for resetting between unrelated test scenarios.
	void reset() {
		reset_state();
	}

	// Capacity getters (mostly for tests / introspection)
	std::size_t pre_trigger_capacity()  const { return pre_size_; }
	std::size_t post_trigger_capacity() const { return post_size_; }

private:
	enum class State { PreFill, Armed, Capturing, Complete };

	void push_to_ring(SampleScalar x) {
		if (pre_size_ == 0) {
			// degenerate: no pre-trigger history requested
			return;
		}
		ring_[write_pos_] = x;
		write_pos_ = (write_pos_ + 1) % pre_size_;
		if (pre_count_ < pre_size_) ++pre_count_;
		if (state_ == State::PreFill && pre_count_ == pre_size_) {
			state_ = State::Armed;
		}
	}

	void reset_state() {
		state_       = pre_size_ == 0 ? State::Armed : State::PreFill;
		write_pos_   = 0;
		pre_count_   = 0;
		segment_pos_ = 0;
		target_length_   = 0;
		captured_length_ = 0;
	}

	std::size_t                          pre_size_;
	std::size_t                          post_size_;
	mtl::vec::dense_vector<SampleScalar> ring_;     // size == pre_size_
	mtl::vec::dense_vector<SampleScalar> segment_;  // size == pre_size_+1+post_size_
	std::size_t              write_pos_   = 0;  // ring write head
	std::size_t              pre_count_   = 0;  // valid samples in ring
	std::size_t              segment_pos_ = 0;  // next write index in segment_
	std::size_t              target_length_   = 0;  // pre-actual + 1 + post
	std::size_t              captured_length_ = 0;
	State                    state_ = State::PreFill;
};

// =============================================================================
// SegmentedCapture
//
// Stores up to max_segments consecutive triggered captures back-to-back.
// Each segment has the same pre/post sizes as a TriggerRingBuffer.
//
// Use case: an oscilloscope capturing rare events at high duty cycle
// (e.g., bus errors) without losing triggers to re-arm dead time.
//
// API mirrors TriggerRingBuffer's push() / push_trigger() but accumulates
// segments instead of stopping after one. capture_complete() returns true
// once max_segments have been captured.
// =============================================================================
template <DspScalar SampleScalar>
class SegmentedCapture {
public:
	using sample_scalar = SampleScalar;

	SegmentedCapture(std::size_t pre_trigger_samples,
	                 std::size_t post_trigger_samples,
	                 std::size_t max_segments)
		: max_segments_(max_segments),
		  buf_(pre_trigger_samples, post_trigger_samples) {
		if (max_segments == 0)
			throw std::invalid_argument(
				"SegmentedCapture: max_segments must be >= 1");
		segments_.reserve(max_segments_);
	}

	void push(SampleScalar x) {
		if (capture_complete()) return;
		buf_.push(x);
		harvest_if_complete();
	}

	void push_trigger(SampleScalar x) {
		if (capture_complete()) return;
		buf_.push_trigger(x);
		harvest_if_complete();
	}

	bool capture_complete() const {
		return segments_.size() >= max_segments_;
	}

	std::size_t segment_count() const { return segments_.size(); }
	std::size_t max_segments()  const { return max_segments_; }

	std::span<const SampleScalar> segment(std::size_t i) const {
		if (i >= segments_.size())
			throw std::out_of_range("SegmentedCapture::segment: index out of range");
		const auto& seg = segments_[i];
		return std::span<const SampleScalar>(seg.data(), seg.size());
	}

	void reset() {
		buf_.reset();
		segments_.clear();
	}

private:
	void harvest_if_complete() {
		if (buf_.capture_complete()) {
			auto seg = buf_.captured_segment();
			mtl::vec::dense_vector<SampleScalar> copy(seg.size());
			for (std::size_t i = 0; i < seg.size(); ++i) copy[i] = seg[i];
			segments_.emplace_back(std::move(copy));
			buf_.rearm();
		}
	}

	std::size_t                                       max_segments_;
	TriggerRingBuffer<SampleScalar>                   buf_;
	std::vector<mtl::vec::dense_vector<SampleScalar>> segments_;
};

} // namespace sw::dsp::instrument
