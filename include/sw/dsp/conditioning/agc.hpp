#pragma once
// agc.hpp: automatic gain control
//
// Adjusts gain to maintain a target output level.
// Uses an RMS envelope follower for level estimation.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/conditioning/envelope.hpp>

namespace sw::dsp {

// Automatic Gain Control.
//
// Measures the RMS level and adjusts gain so the output
// approaches target_level. The gain changes are smoothed
// by the RMS window to avoid audible pumping.
//
// Parameters:
//   target_level: desired RMS output level (linear, e.g., 0.5)
//   window_ms:    RMS averaging window in milliseconds
//   max_gain:     maximum gain to apply (prevents boosting silence)
template <DspOrderedField T>
class AGC {
public:
	AGC() = default;

	void setup(double sample_rate, double target_level,
	           double window_ms = 100.0, double max_gain = 100.0) {
		target_ = static_cast<T>(target_level);
		max_gain_ = static_cast<T>(max_gain);
		rms_.setup(sample_rate, window_ms);
	}

	T process(T input) {
		T level = rms_.process(input);

		// Compute gain (compare in T to avoid requiring double conversion)
		T gain;
		T silence = static_cast<T>(1e-10);
		if (level > silence) {
			gain = target_ / level;
			if (gain > max_gain_) gain = max_gain_;
		} else {
			gain = T{1};  // don't amplify silence
		}

		return input * gain;
	}

	void process_block(mtl::vec::dense_vector<T>& signal) {
		for (std::size_t i = 0; i < signal.size(); ++i) {
			signal[i] = process(signal[i]);
		}
	}

	void reset() { rms_.reset(); }

private:
	T target_{0.5};
	T max_gain_{100};
	RMSEnvelope<T> rms_;
};

} // namespace sw::dsp
