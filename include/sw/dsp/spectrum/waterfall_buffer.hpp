#pragma once
// waterfall_buffer.hpp: circular 2D trace memory for waterfall displays.
//
// Stores the last `num_frames` FFT magnitude frames produced by a
// streaming spectrum processor (RealtimeSpectrum, see
// realtime_spectrum.hpp). Each frame is `num_bins` samples wide; the
// storage is a circular buffer that overwrites the oldest frame when
// full, so the buffer always holds the most recent frames available.
//
// This is the data structure behind a waterfall / spectrogram display:
// time on one axis (frame number), frequency on the other (bin
// number), amplitude as color/intensity. The display refresh fetches
// the last K frames in chronological order; the buffer fills as new
// FFT outputs land.
//
// Naming note: the class is called WaterfallBuffer (not Spectrogram)
// to avoid colliding with sw::dsp::spectral::Spectrogram, which is the
// batch STFT in the general-spectral module. Spectral = general DSP
// transforms; Spectrum = analyzer-specific stages (this file).
//
// Mixed-precision contract:
//   Pure storage. The SampleScalar template parameter only affects
//   how amplitude values are stored; no arithmetic is performed on
//   them. push_frame copies the input span into the ring; reads
//   either return a zero-copy view of one frame (frame_at) or
//   compact a contiguous chronological slice into the internal
//   compaction buffer (last_frames). Same precision-blind story
//   as TriggerRingBuffer.
//
// API split between zero-copy and compacted views:
//   frame_at(i)        - single-frame, zero-copy span into the ring.
//                        Always works because each frame is num_bins
//                        consecutive entries in the ring.
//   last_frames(count) - multi-frame, non-const, returns a span into
//                        an internal compaction buffer because frames
//                        crossing the wrap point aren't contiguous in
//                        the ring. Compacts on call. Suitable for
//                        once-per-display-refresh use.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp::spectrum {

template <DspScalar SampleScalar>
class WaterfallBuffer {
public:
	using sample_scalar = SampleScalar;

	// num_bins:    number of frequency bins per frame (= FFT size for
	//              full spectrum, or fft_size/2 + 1 for one-sided).
	// num_frames:  ring capacity (how many frames to retain).
	WaterfallBuffer(std::size_t num_bins, std::size_t num_frames)
		: num_bins_(check_dims(num_bins, num_frames)),   // throws on bad input
		  num_frames_capacity_(num_frames),
		  ring_(num_bins * num_frames),
		  compact_(num_bins * num_frames) {}

	// Append a frame to the buffer. The frame must have exactly
	// num_bins samples; otherwise std::invalid_argument. When the
	// ring is full, the oldest frame is overwritten — total
	// num_frames_filled() saturates at capacity.
	void push_frame(std::span<const SampleScalar> magnitude) {
		if (magnitude.size() != num_bins_)
			throw std::invalid_argument(
				"WaterfallBuffer::push_frame: magnitude length "
				+ std::to_string(magnitude.size())
				+ " must equal num_bins " + std::to_string(num_bins_));
		const std::size_t offset = write_pos_ * num_bins_;
		for (std::size_t i = 0; i < num_bins_; ++i)
			ring_[offset + i] = magnitude[i];
		write_pos_ = (write_pos_ + 1) % num_frames_capacity_;
		if (num_frames_filled_ < num_frames_capacity_) ++num_frames_filled_;
	}

	// Read a single frame by chronological index. idx_from_oldest = 0
	// is the oldest available frame; idx_from_oldest = num_frames_
	// filled() - 1 is the newest (most recently pushed). The returned
	// span is a zero-copy view into the ring.
	[[nodiscard]] std::span<const SampleScalar>
	frame_at(std::size_t idx_from_oldest) const {
		if (idx_from_oldest >= num_frames_filled_)
			throw std::out_of_range(
				"WaterfallBuffer::frame_at: index "
				+ std::to_string(idx_from_oldest)
				+ " >= num_frames_filled() "
				+ std::to_string(num_frames_filled_));
		// Slot of the oldest frame: write_pos_ - num_frames_filled_,
		// modulo capacity. Then add idx_from_oldest.
		const std::size_t base =
			(write_pos_ + num_frames_capacity_ - num_frames_filled_)
			% num_frames_capacity_;
		const std::size_t slot = (base + idx_from_oldest) % num_frames_capacity_;
		const std::size_t offset = slot * num_bins_;
		return std::span<const SampleScalar>(
			ring_.data() + offset, num_bins_);
	}

	// Read the last `count` frames in chronological order (oldest
	// first), returned as a flat span of length count * num_bins_ in
	// row-major (frame-major) layout. count is clamped to
	// num_frames_filled() — fewer-than-requested frames are returned
	// when the buffer hasn't filled yet.
	//
	// Non-const because the implementation compacts into an internal
	// buffer (the ring's frames cross the wrap point and aren't
	// contiguous in storage). Suitable for once-per-display-refresh
	// use; not a hot-loop API.
	[[nodiscard]] std::span<const SampleScalar>
	last_frames(std::size_t count) {
		const std::size_t available = std::min(count, num_frames_filled_);
		if (available == 0) return {};

		// Walk frames from (num_frames_filled - available) to end,
		// copying each into the compaction buffer.
		const std::size_t start = num_frames_filled_ - available;
		for (std::size_t k = 0; k < available; ++k) {
			auto src = frame_at(start + k);
			const std::size_t dst_offset = k * num_bins_;
			for (std::size_t i = 0; i < num_bins_; ++i)
				compact_[dst_offset + i] = src[i];
		}
		return std::span<const SampleScalar>(
			compact_.data(), available * num_bins_);
	}

	// Discard all stored frames; capacity preserved. Subsequent reads
	// return empty until push_frame is called again.
	void clear() {
		write_pos_         = 0;
		num_frames_filled_ = 0;
		// Ring contents not zeroed — frame_at / last_frames bound their
		// reads by num_frames_filled_, so stale ring data is
		// unreachable.
	}

	[[nodiscard]] std::size_t num_bins()             const { return num_bins_; }
	[[nodiscard]] std::size_t num_frames_capacity()  const { return num_frames_capacity_; }
	[[nodiscard]] std::size_t num_frames_filled()    const { return num_frames_filled_; }

private:
	// Validates the constructor's dimension arguments. Returns
	// num_bins (so it can drive num_bins_'s init in the member-init
	// list) on success; throws on any of:
	//   - num_bins == 0
	//   - num_frames == 0
	//   - num_bins * num_frames overflows std::size_t. The ring and
	//     compact buffers are sized by this product, so a silent
	//     overflow would allocate a tiny buffer and let later index
	//     arithmetic walk off the end.
	static std::size_t check_dims(std::size_t num_bins,
	                              std::size_t num_frames) {
		if (num_bins == 0)
			throw std::invalid_argument(
				"WaterfallBuffer: num_bins must be > 0");
		if (num_frames == 0)
			throw std::invalid_argument(
				"WaterfallBuffer: num_frames must be > 0");
		// Compare against max/x rather than multiplying first.
		if (num_bins > std::numeric_limits<std::size_t>::max() / num_frames)
			throw std::length_error(
				"WaterfallBuffer: num_bins * num_frames overflows size_t (got "
				+ std::to_string(num_bins) + " * "
				+ std::to_string(num_frames) + ")");
		return num_bins;
	}

	std::size_t num_bins_;
	std::size_t num_frames_capacity_;
	std::size_t write_pos_         = 0;
	std::size_t num_frames_filled_ = 0;
	mtl::vec::dense_vector<SampleScalar> ring_;
	mtl::vec::dense_vector<SampleScalar> compact_;
};

} // namespace sw::dsp::spectrum
