#pragma once
// compressor.hpp: dynamic range compressor
//
// Reduces the dynamic range of a signal above a threshold.
// Uses a peak envelope follower for level detection.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/conditioning/envelope.hpp>

namespace sw::dsp {

// Dynamic range compressor.
//
// Note: T must be convertible to double for dB computation.
//
// Parameters:
//   threshold_db: level above which compression starts
//   ratio:        compression ratio (e.g., 4.0 = 4:1)
//   attack_ms:    attack time for level detector
//   release_ms:   release time for level detector
//   makeup_db:    output gain to compensate for compression
template <DspField T>
	requires ConvertibleToDouble<T>
class Compressor {
public:
	Compressor() = default;

	void setup(double sample_rate, double threshold_db, double ratio,
	           double attack_ms, double release_ms, double makeup_db = 0.0,
	           double knee_db = 0.0) {
		if (ratio < 1.0)
			throw std::invalid_argument("Compressor: ratio must be >= 1.0");
		threshold_db_ = threshold_db;
		ratio_ = ratio;
		knee_db_ = knee_db;
		makeup_gain_ = static_cast<T>(std::pow(10.0, makeup_db / 20.0));
		envelope_.setup(sample_rate, attack_ms, release_ms);
	}

	T process(T input) {
		// Detect level
		T env = envelope_.process(input);
		double env_db = 20.0 * std::log10(std::max(static_cast<double>(env), 1e-20));

		// Compute gain reduction with optional soft knee
		double gain_db = 0.0;
		double x = env_db - threshold_db_;

		if (knee_db_ > 0.0 && std::abs(x) < knee_db_ * 0.5) {
			// Soft knee: quadratic interpolation in the knee region
			double knee_half = knee_db_ * 0.5;
			double t = (x + knee_half) / knee_db_;  // 0..1 across knee
			double slope = 1.0 - 1.0 / ratio_;
			gain_db = slope * t * t * knee_half;
		} else if (x > 0.0) {
			// Hard knee (or above soft knee region)
			gain_db = x - x / ratio_;
		}

		T gain = static_cast<T>(std::pow(10.0, -gain_db / 20.0)) * makeup_gain_;
		return input * gain;
	}

	void process_block(mtl::vec::dense_vector<T>& signal) {
		for (std::size_t i = 0; i < signal.size(); ++i) {
			signal[i] = process(signal[i]);
		}
	}

	void reset() { envelope_.reset(); }

private:
	double threshold_db_{-20.0};
	double ratio_{4.0};
	double knee_db_{0.0};
	T makeup_gain_{1};
	PeakEnvelope<T> envelope_;
};

} // namespace sw::dsp
