#pragma once
// measurements.hpp: Scope-style waveform measurements on captured segments.
//
// These are the numbers a digital oscilloscope displays in its bottom
// panel: amplitude, mean, RMS, frequency, period, rise/fall time. Each
// function takes a captured segment as a std::span<const T> and returns
// a double scalar — same one-shot contract as a scope's measurement
// readout.
//
// Mixed-precision contract: aggregations (sum, sum-of-squares, min/max,
// threshold-crossing interpolation) run in double internally regardless
// of SampleScalar. For narrow streaming types this preserves measurement
// accuracy without forcing the caller to widen their sample buffer.
// Returning double matches the precision concerns of the spectrum-
// analyzer RMS detector (#134) and the analysis primitives in
// sw/dsp/analysis/.
//
// Edge-case convention:
//   - Empty segment       -> throw std::invalid_argument
//   - Single sample       -> peak_to_peak=0, mean=value, rms=|value|;
//                            rise/fall/period/frequency return NaN
//                            (no transition to measure)
//   - All-equal samples   -> peak_to_peak=0, mean/rms well-defined;
//                            rise/fall/period/frequency return NaN
//
// NaN is preferred over throws for "no measurable transition" cases so
// that a caller computing all seven values on one segment does not have
// one undefined measurement abort the others.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp::instrument {

namespace detail {

template <typename T>
inline void require_nonempty(std::span<const T> segment, const char* fn) {
	if (segment.empty())
		throw std::invalid_argument(
			std::string(fn) + ": segment is empty");
}

// Linear interpolation: given x[i] = a, x[i+1] = b, find the fractional
// offset within sample i where the signal crosses `threshold`. Returns
// a value in [0, 1]. Caller adds `i` to get the absolute crossing time.
inline double interp_crossing(double a, double b, double threshold) {
	const double denom = b - a;
	if (std::abs(denom) < std::numeric_limits<double>::min())
		return 0.0;   // degenerate: a == b == threshold; pick the left edge
	return (threshold - a) / denom;
}

} // namespace detail

// Peak-to-peak amplitude: max(segment) - min(segment).
//
// For a unit-amplitude sine: 2.0. For a square wave of amplitude A: 2*A.
// Single-sample / all-equal segments return 0.
template <DspOrderedField T>
	requires ConvertibleToDouble<T>
[[nodiscard]] double peak_to_peak(std::span<const T> segment) {
	detail::require_nonempty(segment, "peak_to_peak");
	double lo = static_cast<double>(segment[0]);
	double hi = lo;
	for (std::size_t i = 1; i < segment.size(); ++i) {
		const double v = static_cast<double>(segment[i]);
		if (v < lo) lo = v;
		if (v > hi) hi = v;
	}
	return hi - lo;
}

// Arithmetic mean of the segment (DC level).
//
// Accumulates in double regardless of SampleScalar precision so narrow
// types do not lose accuracy on long segments.
template <DspScalar T>
	requires ConvertibleToDouble<T>
[[nodiscard]] double mean(std::span<const T> segment) {
	detail::require_nonempty(segment, "mean");
	double sum = 0.0;
	for (const T& s : segment) sum += static_cast<double>(s);
	return sum / static_cast<double>(segment.size());
}

// Root-mean-square of the segment.
//
// For a unit-amplitude sine: 1/sqrt(2) ~= 0.7071. For a unit-amplitude
// square wave: 1.0. Sum-of-squares is accumulated in double.
template <DspScalar T>
	requires ConvertibleToDouble<T>
[[nodiscard]] double rms(std::span<const T> segment) {
	detail::require_nonempty(segment, "rms");
	double sumsq = 0.0;
	for (const T& s : segment) {
		const double v = static_cast<double>(s);
		sumsq += v * v;
	}
	return std::sqrt(sumsq / static_cast<double>(segment.size()));
}

// Rise time in samples: the time for the signal to climb from
// low_pct * peak_to_peak above the segment minimum to high_pct *
// peak_to_peak above the minimum, measured on the first such rising
// transition in the segment. Returns NaN if no transition spans both
// thresholds (e.g., all-equal segment, monotonically falling, never
// reaches high_pct).
//
// Sub-sample crossings are computed by linear interpolation between
// the two samples that bracket each threshold, so the returned value
// is fractional. Caller converts to seconds via `result / sample_rate`.
template <DspOrderedField T>
	requires ConvertibleToDouble<T>
[[nodiscard]] double rise_time_samples(std::span<const T> segment,
                                       double low_pct = 0.1,
                                       double high_pct = 0.9) {
	detail::require_nonempty(segment, "rise_time_samples");
	if (!(low_pct >= 0.0 && low_pct < high_pct && high_pct <= 1.0))
		throw std::invalid_argument(
			"rise_time_samples: require 0 <= low_pct < high_pct <= 1");
	if (segment.size() < 2)
		return std::numeric_limits<double>::quiet_NaN();

	double lo = static_cast<double>(segment[0]);
	double hi = lo;
	for (std::size_t i = 1; i < segment.size(); ++i) {
		const double v = static_cast<double>(segment[i]);
		if (v < lo) lo = v;
		if (v > hi) hi = v;
	}
	const double range = hi - lo;
	if (range <= 0.0)
		return std::numeric_limits<double>::quiet_NaN();
	const double thr_lo = lo + low_pct  * range;
	const double thr_hi = lo + high_pct * range;

	// First rising crossing of thr_lo, then first subsequent rising
	// crossing of thr_hi. "Rising" = sample[i] < threshold,
	// sample[i+1] >= threshold.
	double t_lo = std::numeric_limits<double>::quiet_NaN();
	double t_hi = std::numeric_limits<double>::quiet_NaN();
	for (std::size_t i = 0; i + 1 < segment.size(); ++i) {
		const double a = static_cast<double>(segment[i]);
		const double b = static_cast<double>(segment[i + 1]);
		if (std::isnan(t_lo)) {
			if (a < thr_lo && b >= thr_lo)
				t_lo = static_cast<double>(i)
				     + detail::interp_crossing(a, b, thr_lo);
		} else if (std::isnan(t_hi)) {
			if (a < thr_hi && b >= thr_hi) {
				t_hi = static_cast<double>(i)
				     + detail::interp_crossing(a, b, thr_hi);
				break;
			}
		}
	}
	if (std::isnan(t_lo) || std::isnan(t_hi))
		return std::numeric_limits<double>::quiet_NaN();
	return t_hi - t_lo;
}

// Fall time in samples: mirror of rise_time_samples for the first
// falling transition from high_pct down to low_pct of peak-to-peak.
// Returns NaN if no falling transition spans both thresholds.
template <DspOrderedField T>
	requires ConvertibleToDouble<T>
[[nodiscard]] double fall_time_samples(std::span<const T> segment,
                                       double low_pct = 0.1,
                                       double high_pct = 0.9) {
	detail::require_nonempty(segment, "fall_time_samples");
	if (!(low_pct >= 0.0 && low_pct < high_pct && high_pct <= 1.0))
		throw std::invalid_argument(
			"fall_time_samples: require 0 <= low_pct < high_pct <= 1");
	if (segment.size() < 2)
		return std::numeric_limits<double>::quiet_NaN();

	double lo = static_cast<double>(segment[0]);
	double hi = lo;
	for (std::size_t i = 1; i < segment.size(); ++i) {
		const double v = static_cast<double>(segment[i]);
		if (v < lo) lo = v;
		if (v > hi) hi = v;
	}
	const double range = hi - lo;
	if (range <= 0.0)
		return std::numeric_limits<double>::quiet_NaN();
	const double thr_lo = lo + low_pct  * range;
	const double thr_hi = lo + high_pct * range;

	// First falling crossing of thr_hi (a >= thr_hi, b < thr_hi),
	// then first subsequent falling crossing of thr_lo.
	double t_hi = std::numeric_limits<double>::quiet_NaN();
	double t_lo = std::numeric_limits<double>::quiet_NaN();
	for (std::size_t i = 0; i + 1 < segment.size(); ++i) {
		const double a = static_cast<double>(segment[i]);
		const double b = static_cast<double>(segment[i + 1]);
		if (std::isnan(t_hi)) {
			if (a >= thr_hi && b < thr_hi)
				t_hi = static_cast<double>(i)
				     + detail::interp_crossing(a, b, thr_hi);
		} else if (std::isnan(t_lo)) {
			if (a >= thr_lo && b < thr_lo) {
				t_lo = static_cast<double>(i)
				     + detail::interp_crossing(a, b, thr_lo);
				break;
			}
		}
	}
	if (std::isnan(t_hi) || std::isnan(t_lo))
		return std::numeric_limits<double>::quiet_NaN();
	return t_lo - t_hi;
}

// Period in samples: average distance between consecutive rising
// threshold-crossings. Threshold defaults to T{0} (zero-crossing,
// natural for AC-coupled signals). Returns NaN if fewer than two
// rising crossings occur in the segment.
//
// Sub-sample crossing times use linear interpolation, so the returned
// period is fractional.
template <DspOrderedField T>
	requires ConvertibleToDouble<T>
[[nodiscard]] double period_samples(std::span<const T> segment,
                                    T threshold = T{0}) {
	detail::require_nonempty(segment, "period_samples");
	if (segment.size() < 2)
		return std::numeric_limits<double>::quiet_NaN();
	const double thr = static_cast<double>(threshold);

	// Collect all rising crossing times.
	double first_crossing = 0.0;
	double last_crossing  = 0.0;
	std::size_t crossings = 0;
	for (std::size_t i = 0; i + 1 < segment.size(); ++i) {
		const double a = static_cast<double>(segment[i]);
		const double b = static_cast<double>(segment[i + 1]);
		if (a < thr && b >= thr) {
			const double t = static_cast<double>(i)
			               + detail::interp_crossing(a, b, thr);
			if (crossings == 0) first_crossing = t;
			last_crossing = t;
			++crossings;
		}
	}
	if (crossings < 2)
		return std::numeric_limits<double>::quiet_NaN();
	// (n-1) intervals between n crossings.
	return (last_crossing - first_crossing)
	     / static_cast<double>(crossings - 1);
}

// Frequency in Hz: 1 / (period_samples * sample_period). Returns NaN
// if period cannot be measured (see period_samples).
template <DspOrderedField T>
	requires ConvertibleToDouble<T>
[[nodiscard]] double frequency_hz(std::span<const T> segment,
                                  double sample_rate,
                                  T threshold = T{0}) {
	if (!(sample_rate > 0.0))
		throw std::invalid_argument(
			"frequency_hz: sample_rate must be positive");
	const double T_samples = period_samples(segment, threshold);
	if (std::isnan(T_samples) || T_samples <= 0.0)
		return std::numeric_limits<double>::quiet_NaN();
	return sample_rate / T_samples;
}

} // namespace sw::dsp::instrument
