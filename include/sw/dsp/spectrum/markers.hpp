#pragma once
// markers.hpp: spectrum-analyzer marker / peak-find utilities.
//
// The "what's interesting in this trace?" stage. Three free functions
// answer the analyst's headline questions:
//
//   find_peaks         - top-N strongest spectral lines, with a
//                        minimum-separation rule to suppress local-max
//                        chatter on noisy traces. Sub-bin frequency
//                        position recovered via parabolic interpolation
//                        across the three samples around each peak.
//   harmonic_markers   - given a fundamental frequency, return markers
//                        at the bins nearest k*fundamental for k=2..N.
//                        No peak search around the target bin; the
//                        caller composes with find_peaks if they want
//                        peak-snapped harmonics.
//   make_delta_marker  - difference of two markers (frequency and
//                        amplitude). The delta-marker measurement
//                        every commercial analyzer ships.
//
// All functions are stateless reducers over a span of trace samples.
// The trace is typically already in a log scale (dB) by this stage,
// so amplitude comparisons are meaningful as-is.
//
// Mixed-precision contract:
//   - Amplitude comparisons (peak detection, top-N selection) run in
//     T per DspOrderedField.
//   - Sub-bin parabolic interpolation accumulates in double regardless
//     of T; the returned `frequency_hz` is always double.
//   - Marker / DeltaMarker fields are double for type-uniformity
//     across the analyzer's reporting layer.
//
// Edge cases:
//   - Empty trace -> returns empty vector (no peaks possible).
//   - top_n == 0 -> returns empty vector.
//   - harmonics == 0 -> returns empty vector.
//   - trace.size() == 1 -> the single bin is a peak iff trace[0] is
//     consulted (no neighbors to compare); we treat it as a peak with
//     no sub-bin offset.
//   - bin_freq_step_hz <= 0 -> std::invalid_argument.
//   - fundamental_hz <= 0 -> std::invalid_argument.
//   - Edge bins (i = 0 or N-1) are flagged as peaks if higher than
//     their single neighbor; sub-bin interpolation is skipped (no
//     parabola fits with only two points).
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp::spectrum {

// A single marker on the trace: which bin, what frequency, what amplitude.
//
// `bin_index` is the integer bin nearest the marker. `frequency_hz`
// reflects the sub-bin-interpolated position when applicable; for
// markers at edge bins (or with degenerate parabolic fits) it equals
// `bin_index * bin_freq_step_hz`.
struct Marker {
	std::size_t bin_index    = 0;
	double      frequency_hz = 0.0;
	double      amplitude    = 0.0;
};

// Two-marker measurement standard. Delta values are b minus a.
struct DeltaMarker {
	Marker a;
	Marker b;
	double delta_freq_hz   = 0.0;
	double delta_amplitude = 0.0;
};

namespace detail {

// Parabolic interpolation for sub-bin peak position. Given samples
// y_left = y[i-1], y_center = y[i], y_right = y[i+1] around a local
// max at bin i, the parabolic vertex sits at fractional offset
//
//     delta = 0.5 * (y_left - y_right) / (y_left - 2*y_center + y_right)
//
// from the center bin. delta is in [-0.5, +0.5] for a true local max.
// If the denominator is degenerate (flat top), returns 0.
inline double parabolic_offset(double y_left, double y_center, double y_right) {
	const double denom = y_left - 2.0 * y_center + y_right;
	// Exact-zero guard: see the same convention in
	// instrument/measurements::interp_crossing.
	if (denom == 0.0) return 0.0;
	const double offset = 0.5 * (y_left - y_right) / denom;
	// Clamp to [-0.5, +0.5] — values outside that range mean the
	// quadratic fit is no longer locally a maximum (e.g., due to
	// rounding noise on a flat region) and the integer bin position
	// is the most honest answer.
	if (offset < -0.5) return -0.5;
	if (offset >  0.5) return  0.5;
	return offset;
}

// Local-max detection: bin i is a peak iff it dominates its neighbors.
// Edge bins (i=0 or i=N-1) need special handling — only one neighbor.
template <typename T>
inline bool is_local_max(std::span<const T> trace, std::size_t i) {
	const std::size_t n = trace.size();
	if (n == 0) return false;
	if (n == 1) return true;
	if (i == 0)        return trace[0]   > trace[1];
	if (i == n - 1)    return trace[n-1] > trace[n-2];
	return trace[i] > trace[i-1] && trace[i] > trace[i+1];
}

} // namespace detail

// Top-N strongest peaks with a minimum-separation rule.
//
// Algorithm:
//   1. Identify all local maxima in the trace (including edges).
//   2. Sort them by amplitude descending.
//   3. Greedy-select: take the strongest, then walk down the list and
//      pick each next-strongest peak that's at least
//      `min_separation_bins` away from any already-selected peak.
//      Repeat until top_n peaks are selected or the list is exhausted.
//
// The greedy step is what suppresses local-max chatter on noisy
// traces — without it a single broad peak straddling several adjacent
// bins would consume all top_n slots.
//
// Returned markers are in descending amplitude order (selection order).
// Sub-bin frequency interpolation is applied via a parabolic fit
// across the three bins around each peak; edge bins skip interpolation.
template <DspOrderedField T>
	requires ConvertibleToDouble<T>
[[nodiscard]] std::vector<Marker> find_peaks(
		std::span<const T> trace,
		double bin_freq_step_hz,
		std::size_t top_n,
		std::size_t min_separation_bins = 3) {
	if (!(bin_freq_step_hz > 0.0))
		throw std::invalid_argument(
			"find_peaks: bin_freq_step_hz must be positive (got "
			+ std::to_string(bin_freq_step_hz) + ")");
	if (trace.empty() || top_n == 0) return {};

	// Step 1: collect (bin_index, amplitude) pairs for every local max.
	// Amplitude stored in T (not double) so the sort below preserves
	// the mixed-precision ordering that DspOrderedField promises — same
	// rationale as detect_peak / detect_negative_peak in detectors.hpp
	// and min_max_double in instrument/measurements.hpp.
	struct Candidate { std::size_t bin; T amp; };
	std::vector<Candidate> candidates;
	candidates.reserve(trace.size() / 4 + 4);   // rough upper bound
	for (std::size_t i = 0; i < trace.size(); ++i) {
		if (detail::is_local_max(trace, i))
			candidates.push_back({i, trace[i]});
	}

	// Step 2: sort by amplitude descending. Comparison is in T (per
	// DspOrderedField); ties broken by lower bin first for determinism.
	// Use only `>` so we don't require T to support `!=`.
	std::sort(candidates.begin(), candidates.end(),
	          [](const Candidate& a, const Candidate& b) {
	              if (a.amp > b.amp) return true;
	              if (b.amp > a.amp) return false;
	              return a.bin < b.bin;
	          });

	// Step 3: greedy-select with min-separation.
	std::vector<Marker> out;
	out.reserve(top_n);
	for (const auto& c : candidates) {
		if (out.size() >= top_n) break;
		bool too_close = false;
		for (const auto& sel : out) {
			const std::size_t d = c.bin > sel.bin_index
			                       ? c.bin - sel.bin_index
			                       : sel.bin_index - c.bin;
			if (d < min_separation_bins) { too_close = true; break; }
		}
		if (too_close) continue;

		// Sub-bin parabolic interpolation. Skip for edge bins. The
		// parabolic fit is a numerical operation that benefits from
		// double accumulation regardless of T, so we cast at the
		// boundary (same pattern as the Marker.amplitude output).
		Marker m;
		m.bin_index = c.bin;
		m.amplitude = static_cast<double>(c.amp);
		double offset = 0.0;
		if (c.bin > 0 && c.bin + 1 < trace.size()) {
			offset = detail::parabolic_offset(
				static_cast<double>(trace[c.bin - 1]),
				static_cast<double>(c.amp),
				static_cast<double>(trace[c.bin + 1]));
		}
		m.frequency_hz = (static_cast<double>(c.bin) + offset) * bin_freq_step_hz;
		out.push_back(m);
	}
	return out;
}

// Markers at the bins nearest k * fundamental_hz for k = 2..harmonics.
//
// No peak search around the target bin: this returns the trace value
// at the rounded bin position. If the harmonic has drifted off-bin
// (real-world noise or quantization), the caller composes this with
// find_peaks() to peak-snap each harmonic in a small neighborhood.
//
// Harmonics that fall past the trace's frequency range are silently
// omitted from the result (no out-of-range error).
template <DspOrderedField T>
	requires ConvertibleToDouble<T>
[[nodiscard]] std::vector<Marker> harmonic_markers(
		std::span<const T> trace,
		double bin_freq_step_hz,
		double fundamental_hz,
		std::size_t harmonics) {
	if (!(bin_freq_step_hz > 0.0))
		throw std::invalid_argument(
			"harmonic_markers: bin_freq_step_hz must be positive (got "
			+ std::to_string(bin_freq_step_hz) + ")");
	if (!(fundamental_hz > 0.0))
		throw std::invalid_argument(
			"harmonic_markers: fundamental_hz must be positive (got "
			+ std::to_string(fundamental_hz) + ")");
	if (trace.empty() || harmonics == 0) return {};

	std::vector<Marker> out;
	out.reserve(harmonics);
	for (std::size_t k = 2; k < 2 + harmonics; ++k) {
		const double target_freq = static_cast<double>(k) * fundamental_hz;
		const double target_bin_d = target_freq / bin_freq_step_hz;
		// Round to nearest. std::lround would also work; using
		// floor(x + 0.5) avoids platform-specific banker's rounding.
		const std::size_t target_bin =
			static_cast<std::size_t>(std::floor(target_bin_d + 0.5));
		if (target_bin >= trace.size()) break;   // remaining harmonics out of range

		Marker m;
		m.bin_index    = target_bin;
		m.frequency_hz = static_cast<double>(target_bin) * bin_freq_step_hz;
		m.amplitude    = static_cast<double>(trace[target_bin]);
		out.push_back(m);
	}
	return out;
}

// Two-marker delta. b minus a, in both frequency and amplitude.
[[nodiscard]] inline DeltaMarker make_delta_marker(const Marker& a,
                                                    const Marker& b) {
	DeltaMarker d;
	d.a = a;
	d.b = b;
	d.delta_freq_hz   = b.frequency_hz - a.frequency_hz;
	d.delta_amplitude = b.amplitude    - a.amplitude;
	return d;
}

} // namespace sw::dsp::spectrum
