#pragma once
// detectors.hpp: spectrum-analyzer detector modes.
//
// In a swept-tuned analyzer, the detector is the stage between the RBW
// filter and the trace memory: it reduces a window of post-RBW samples
// (the dwell time at one frequency bin) to a single trace value. In an
// FFT-based analyzer, the detector is applied to the per-bin samples
// produced by overlapping FFTs (or, equivalently, to the magnitude
// stream sampled at the FFT's hop rate).
//
// Five modes are supported:
//
//   Peak          - max-hold within the bin window
//   Sample        - the FIRST sample in the bin window (no averaging,
//                   no extreme-tracking; mirrors the "sample detector"
//                   as commonly defined in CISPR / Keysight references)
//   Average       - arithmetic mean of the bin samples (linear)
//   RMS           - sqrt(mean(x^2)) - the energy detector
//   NegativePeak  - min-hold within the bin window
//
// CISPR-22 quasi-peak is intentionally NOT in this set. Its decay-time
// semantics (charge-then-decay weighted average) want a stateful class,
// not a stateless reducer over a span; it lands in a separate sub-issue.
//
// Mixed-precision contract: all detectors return double. Sum, sum-of-
// squares, and other accumulators run in double internally regardless
// of SampleScalar - same convention as instrument/measurements.hpp.
// Min/max are comparison-only and precision-blind, but the result is
// cast to double for a uniform return type across modes.
//
// Edge-case convention (matches instrument/measurements):
//   - Empty span -> throws std::invalid_argument
//   - Single sample -> returns that sample (or |sample| for RMS)
//   - All-equal samples -> well-defined for every mode
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp::spectrum {

// Selector for `detect()`. Individual entry-point functions exist for
// each mode (detect_peak, detect_sample, ...) for callers that pick at
// compile time; this enum is the runtime-dispatch path.
enum class DetectorMode {
	Peak,
	Sample,
	Average,
	RMS,
	NegativePeak
};

namespace detail {

template <typename T>
inline void require_nonempty(std::span<const T> bin, const char* fn) {
	if (bin.empty())
		throw std::invalid_argument(
			std::string(fn) + ": bin samples span is empty");
}

} // namespace detail

// Peak detector: max(bin). Standard "peak" mode on commercial analyzers.
//
// Comparison-only - storage precision is preserved bit-exact (no
// arithmetic). Returned as double for type-uniformity across modes.
template <DspOrderedField T>
	requires ConvertibleToDouble<T>
[[nodiscard]] double detect_peak(std::span<const T> bin) {
	detail::require_nonempty(bin, "detect_peak");
	double hi = static_cast<double>(bin[0]);
	for (std::size_t i = 1; i < bin.size(); ++i) {
		const double v = static_cast<double>(bin[i]);
		if (v > hi) hi = v;
	}
	return hi;
}

// Negative-peak detector: min(bin). Useful for finding the deepest
// notch in a frequency response or the floor of a noise distribution.
template <DspOrderedField T>
	requires ConvertibleToDouble<T>
[[nodiscard]] double detect_negative_peak(std::span<const T> bin) {
	detail::require_nonempty(bin, "detect_negative_peak");
	double lo = static_cast<double>(bin[0]);
	for (std::size_t i = 1; i < bin.size(); ++i) {
		const double v = static_cast<double>(bin[i]);
		if (v < lo) lo = v;
	}
	return lo;
}

// Sample detector: the FIRST sample in the bin window.
//
// Conceptually a "no-detector" mode - it picks one representative time
// instant per bin and ignores the rest. Used when the bin dwell is
// short enough that one sample is good enough, or as a reference point
// for the others. Convention is to take the first sample (the analyzer
// has just started measuring at this frequency). The bin is still
// required to be non-empty.
template <DspScalar T>
	requires ConvertibleToDouble<T>
[[nodiscard]] double detect_sample(std::span<const T> bin) {
	detail::require_nonempty(bin, "detect_sample");
	return static_cast<double>(bin[0]);
}

// Average detector: arithmetic mean of the bin samples.
//
// Accumulation in double regardless of SampleScalar. For narrow types
// on long bins this preserves accuracy without forcing the caller to
// widen their input.
template <DspScalar T>
	requires ConvertibleToDouble<T>
[[nodiscard]] double detect_average(std::span<const T> bin) {
	detail::require_nonempty(bin, "detect_average");
	double sum = 0.0;
	for (const T& s : bin) sum += static_cast<double>(s);
	return sum / static_cast<double>(bin.size());
}

// RMS detector: sqrt(mean(x^2)). The energy detector.
//
// For a unit-amplitude sine: 1/sqrt(2) ~= 0.7071.
// For a unit-amplitude square wave: 1.0.
// Sum-of-squares is accumulated in double.
template <DspScalar T>
	requires ConvertibleToDouble<T>
[[nodiscard]] double detect_rms(std::span<const T> bin) {
	detail::require_nonempty(bin, "detect_rms");
	double sumsq = 0.0;
	for (const T& s : bin) {
		const double v = static_cast<double>(s);
		sumsq += v * v;
	}
	return std::sqrt(sumsq / static_cast<double>(bin.size()));
}

// Runtime-dispatch entry point. Chooses one of the five modes via the
// DetectorMode enum. For compile-time-known modes prefer the named
// detect_* functions above (one less branch).
//
// DspOrderedField is required (not just DspScalar) because Peak /
// NegativePeak need ordering. For Sample / Average / RMS the ordering
// requirement is vacuous but the concept is uniform across the five
// modes for a single dispatch entry-point.
template <DspOrderedField T>
	requires ConvertibleToDouble<T>
[[nodiscard]] double detect(std::span<const T> bin, DetectorMode mode) {
	switch (mode) {
		case DetectorMode::Peak:         return detect_peak(bin);
		case DetectorMode::Sample:       return detect_sample(bin);
		case DetectorMode::Average:      return detect_average(bin);
		case DetectorMode::RMS:          return detect_rms(bin);
		case DetectorMode::NegativePeak: return detect_negative_peak(bin);
	}
	// Unreachable for a valid enum value, but a defensive throw lets
	// the caller catch a corrupted-mode bug instead of silently
	// returning garbage.
	throw std::invalid_argument("detect: unknown DetectorMode value");
}

} // namespace sw::dsp::spectrum
