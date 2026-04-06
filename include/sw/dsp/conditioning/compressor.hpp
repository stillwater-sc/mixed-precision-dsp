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
	           double attack_ms, double release_ms, double makeup_db = 0.0) {
		threshold_db_ = threshold_db;
		ratio_ = ratio;
		makeup_gain_ = static_cast<T>(std::pow(10.0, makeup_db / 20.0));
		envelope_.setup(sample_rate, attack_ms, release_ms);
	}

	T process(T input) {
		// Detect level
		T env = envelope_.process(input);
		double env_db = 20.0 * std::log10(std::max(static_cast<double>(env), 1e-20));

		// Compute gain reduction
		double gain_db = 0.0;
		if (env_db > threshold_db_) {
			double excess = env_db - threshold_db_;
			gain_db = excess - excess / ratio_;  // how much to reduce
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
	T makeup_gain_{1};
	PeakEnvelope<T> envelope_;
};

} // namespace sw::dsp
