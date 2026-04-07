#pragma once
// envelope.hpp: peak and RMS envelope followers
//
// Track the amplitude envelope of a signal with configurable
// attack and release times.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

// Peak envelope follower.
//
// Tracks the peak amplitude with exponential attack and release.
// attack_coeff and release_coeff are smoothing coefficients in [0, 1]:
//   coeff = exp(-1 / (time_seconds * sample_rate))
template <DspField T>
class PeakEnvelope {
public:
	PeakEnvelope() = default;

	PeakEnvelope(double sample_rate, double attack_ms, double release_ms) {
		setup(sample_rate, attack_ms, release_ms);
	}

	void setup(double sample_rate, double attack_ms, double release_ms) {
		attack_coeff_ = static_cast<T>(std::exp(-1000.0 / (attack_ms * sample_rate)));
		release_coeff_ = static_cast<T>(std::exp(-1000.0 / (release_ms * sample_rate)));
	}

	T process(T input) {
		using std::abs;
		T level = static_cast<T>(abs(input));
		if (level > envelope_) {
			envelope_ = attack_coeff_ * envelope_ + (T{1} - attack_coeff_) * level;
		} else {
			envelope_ = release_coeff_ * envelope_ + (T{1} - release_coeff_) * level;
		}
		return envelope_;
	}

	void process_block(const mtl::vec::dense_vector<T>& input,
	                   mtl::vec::dense_vector<T>& output) {
		for (std::size_t i = 0; i < input.size(); ++i) {
			output[i] = process(input[i]);
		}
	}

	T value() const { return envelope_; }
	void reset() { envelope_ = T{}; }

private:
	T attack_coeff_{};
	T release_coeff_{};
	T envelope_{};
};

// RMS envelope follower.
//
// Tracks the RMS level using a first-order lowpass on the squared signal.
template <DspField T>
class RMSEnvelope {
public:
	RMSEnvelope() = default;

	RMSEnvelope(double sample_rate, double window_ms) {
		setup(sample_rate, window_ms);
	}

	void setup(double sample_rate, double window_ms) {
		coeff_ = static_cast<T>(std::exp(-1000.0 / (window_ms * sample_rate)));
	}

	T process(T input) {
		using std::sqrt;
		T sq = input * input;
		mean_sq_ = coeff_ * mean_sq_ + (T{1} - coeff_) * sq;
		return sqrt(mean_sq_);
	}

	T value() const { using std::sqrt; return sqrt(mean_sq_); }
	void reset() { mean_sq_ = T{}; }

private:
	T coeff_{};
	T mean_sq_{};
};

} // namespace sw::dsp
