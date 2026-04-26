#pragma once
// display_envelope.hpp: Reduce a captured signal segment to a fixed-pixel
// min/max envelope for display-rate rendering.
//
// Real scopes do this constantly: a 10-million-sample capture must fit on a
// 1024-pixel-wide screen. Each pixel column displays the (min, max) of the
// samples that fell in its time bin. The user's eye sees a vertical line
// bounding the signal's amplitude in each column — narrow glitches still
// show up as long as they appeared somewhere in the column's span.
//
// Distinct from PeakDetectDecimator (peak_detect.hpp) in two ways:
//
//   1. Variable decimation factor: input length is arbitrary, output length
//      is fixed (pixel_width). The decimation factor falls out of
//      (input.size() / pixel_width) and is in general not constant —
//      remainder samples are distributed across leading pixels.
//
//   2. One-shot reduction over a complete captured segment, not a streaming
//      decimator. Free function rather than a stateful class.
//
// Both primitives use the same min/max logic; this one just picks the
// per-bin span dynamically.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <span>
#include <stdexcept>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp::instrument {

template <DspScalar SampleScalar>
struct DisplayEnvelope {
	mtl::vec::dense_vector<SampleScalar> mins;  // length == pixel_width
	mtl::vec::dense_vector<SampleScalar> maxs;  // length == pixel_width
};

// =============================================================================
// render_envelope
//
// Reduce `samples` to exactly `pixel_width` (min, max) pairs.
//
// Cases:
//   - samples.size() == 0:
//       returns an empty envelope (mins.size() == maxs.size() == 0).
//       pixel_width must still be > 0; an empty envelope is the right
//       answer because there's nothing to render.
//   - samples.size() <= pixel_width:
//       each input sample becomes one pixel column with min == max == that
//       sample. Trailing pixels (if pixel_width > samples.size()) are
//       padded with the LAST sample's value, so the trace flat-lines
//       at the right edge rather than dropping to zero. (This is the
//       conventional choice; alternatives are NaN or 0, neither of which
//       renders sensibly.)
//   - samples.size() > pixel_width:
//       distribute samples across pixels. Floor-distribution leaves a
//       remainder of (samples.size() % pixel_width); spread it one extra
//       sample per leading pixel until exhausted. So the leading
//       (remainder) pixels each cover ⌈N/W⌉ samples and the trailing
//       (pixel_width - remainder) pixels each cover ⌊N/W⌋ samples.
// =============================================================================
template <DspScalar SampleScalar>
DisplayEnvelope<SampleScalar> render_envelope(
		std::span<const SampleScalar> samples,
		std::size_t                   pixel_width) {
	if (pixel_width == 0)
		throw std::invalid_argument(
			"render_envelope: pixel_width must be > 0");

	DisplayEnvelope<SampleScalar> env{
		mtl::vec::dense_vector<SampleScalar>(pixel_width),
		mtl::vec::dense_vector<SampleScalar>(pixel_width)};

	const std::size_t N = samples.size();

	// Empty input — return empty envelope (we still report pixel_width
	// in the allocated vectors, but they have no meaningful content).
	// Prefer to surface this by returning an envelope with size 0 so
	// callers can detect "nothing to render."
	if (N == 0) {
		env.mins = mtl::vec::dense_vector<SampleScalar>(0);
		env.maxs = mtl::vec::dense_vector<SampleScalar>(0);
		return env;
	}

	// Sparse case: one input sample per pixel (or fewer). Pad trailing
	// pixels with the last sample's value so the trace flat-lines at the
	// edge.
	if (N <= pixel_width) {
		for (std::size_t i = 0; i < N; ++i) {
			env.mins[i] = samples[i];
			env.maxs[i] = samples[i];
		}
		const SampleScalar last = samples[N - 1];
		for (std::size_t i = N; i < pixel_width; ++i) {
			env.mins[i] = last;
			env.maxs[i] = last;
		}
		return env;
	}

	// Dense case: each pixel covers ≥ 1 sample. Distribute the remainder
	// (N % pixel_width) across the leading pixels.
	const std::size_t base_span = N / pixel_width;
	const std::size_t remainder = N % pixel_width;

	std::size_t in_pos = 0;
	for (std::size_t p = 0; p < pixel_width; ++p) {
		const std::size_t span = base_span + (p < remainder ? 1 : 0);
		// At least one sample per pixel since N > pixel_width.
		SampleScalar mn = samples[in_pos];
		SampleScalar mx = samples[in_pos];
		for (std::size_t k = 1; k < span; ++k) {
			const SampleScalar x = samples[in_pos + k];
			if (x < mn) mn = x;
			if (x > mx) mx = x;
		}
		env.mins[p] = mn;
		env.maxs[p] = mx;
		in_pos += span;
	}
	return env;
}

} // namespace sw::dsp::instrument
