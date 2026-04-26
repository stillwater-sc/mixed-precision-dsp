#pragma once
// peak_detect.hpp: Min/max peak-detect decimation for instrument-style
// data acquisition.
//
// PeakDetectDecimator emits one (min, max) pair for every R input samples.
// Both extremes are preserved, so a narrow glitch shorter than the
// decimation interval still shows up in the output: at any zoom level,
// the user sees a vertical line bounding the glitch's amplitude rather
// than the glitch being smoothed or aliased away. This is the defining
// difference between a scope-style decimator and a generic averaging or
// polyphase one — a generic decimator throws peaks away; a scope MUST
// preserve them.
//
// Used by both the capture pipeline (initial post-trigger rate reduction)
// and the display pipeline (final reduction toward N pixels — see
// display_envelope.hpp / issue #149).
//
// Precision-insensitive: min/max are reductions, not arithmetic. The
// SampleScalar parameter only controls the type of the values being
// compared and emitted.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp::instrument {

// =============================================================================
// PeakDetectDecimator
//
// One (min, max) pair per R input samples. The streaming `process()` returns
// std::nullopt while accumulating a window and the (min, max) pair on the
// sample that completes the window.
//
// Block APIs `process_block_min/max(input)` are convenience wrappers that
// return separate min and max vectors of length floor(input.size() / R).
// Any partial trailing window (input.size() % R != 0) is dropped — the
// streaming `process()` would still be in the std::nullopt state for those
// samples. The `process_block(input)` overload returns both vectors as a
// PeakDetectEnvelope struct in a single call.
// =============================================================================

template <DspScalar SampleScalar>
struct PeakDetectEnvelope {
	mtl::vec::dense_vector<SampleScalar> mins;
	mtl::vec::dense_vector<SampleScalar> maxs;
};

template <DspScalar SampleScalar>
class PeakDetectDecimator {
public:
	using sample_scalar = SampleScalar;

	// decimation_factor R: how many input samples per output (min, max)
	// pair. R=1 is degenerate but valid (passthrough — every sample
	// becomes a (sample, sample) pair).
	explicit PeakDetectDecimator(std::size_t decimation_factor)
		: R_(decimation_factor),
		  count_(0),
		  cur_min_(),
		  cur_max_() {
		if (R_ == 0)
			throw std::invalid_argument(
				"PeakDetectDecimator: decimation_factor must be >= 1");
	}

	// Push one input sample. Returns std::nullopt while accumulating
	// within a window; returns the (min, max) pair on the sample that
	// completes the current window.
	std::optional<std::pair<SampleScalar, SampleScalar>> process(SampleScalar x) {
		if (count_ == 0) {
			// First sample of a new window: seed both extremes.
			cur_min_ = x;
			cur_max_ = x;
		} else {
			if (x < cur_min_) cur_min_ = x;
			if (x > cur_max_) cur_max_ = x;
		}
		++count_;
		if (count_ == R_) {
			count_ = 0;
			return std::make_pair(cur_min_, cur_max_);
		}
		return std::nullopt;
	}

	// Block APIs. Each returns a vector of length equal to the number of
	// complete windows that this call (plus any prior partial-window
	// state from streaming process() calls) will close. That count is
	// (count_ + input.size()) / R_ — must include count_, otherwise a
	// caller mixing streaming and block usage gets an under-allocated
	// buffer.
	mtl::vec::dense_vector<SampleScalar>
	process_block_min(std::span<const SampleScalar> input) {
		const std::size_t n_out = (count_ + input.size()) / R_;
		mtl::vec::dense_vector<SampleScalar> out(n_out);
		std::size_t out_idx = 0;
		for (auto x : input) {
			if (auto p = process(x); p.has_value()) {
				out[out_idx++] = p->first;
			}
		}
		return out;
	}

	mtl::vec::dense_vector<SampleScalar>
	process_block_max(std::span<const SampleScalar> input) {
		const std::size_t n_out = (count_ + input.size()) / R_;
		mtl::vec::dense_vector<SampleScalar> out(n_out);
		std::size_t out_idx = 0;
		for (auto x : input) {
			if (auto p = process(x); p.has_value()) {
				out[out_idx++] = p->second;
			}
		}
		return out;
	}

	// Convenience: return both vectors in one pass.
	PeakDetectEnvelope<SampleScalar>
	process_block(std::span<const SampleScalar> input) {
		const std::size_t n_out = (count_ + input.size()) / R_;
		PeakDetectEnvelope<SampleScalar> env{
			mtl::vec::dense_vector<SampleScalar>(n_out),
			mtl::vec::dense_vector<SampleScalar>(n_out)};
		std::size_t out_idx = 0;
		for (auto x : input) {
			if (auto p = process(x); p.has_value()) {
				env.mins[out_idx] = p->first;
				env.maxs[out_idx] = p->second;
				++out_idx;
			}
		}
		return env;
	}

	// Re-arm the decimator: drop any partial window in progress.
	void reset() {
		count_   = 0;
		cur_min_ = SampleScalar{};
		cur_max_ = SampleScalar{};
	}

	std::size_t decimation_factor()    const { return R_; }
	std::size_t samples_in_window()    const { return count_; }

private:
	std::size_t  R_;        // decimation factor
	std::size_t  count_;    // samples seen in the current window
	SampleScalar cur_min_;
	SampleScalar cur_max_;
};

} // namespace sw::dsp::instrument
