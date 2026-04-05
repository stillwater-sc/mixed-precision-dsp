#pragma once
// cascade.hpp: cascade of second-order sections
//
// A Cascade holds an array of BiquadCoefficients and processes
// samples by running them through each stage in series. The
// cascade is constructed from a PoleZeroLayout by converting
// each pole/zero pair to biquad coefficients and normalizing
// the overall gain.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <array>
#include <cassert>
#include <complex>
#include <cmath>
#include <sw/dsp/types/biquad_coefficients.hpp>
#include <sw/dsp/filter/layout/layout.hpp>
#include <sw/dsp/math/constants.hpp>

namespace sw::dsp {

template <DspField CoeffScalar, int MaxStages>
class Cascade {
public:
	constexpr Cascade() = default;

	// Number of active biquad stages
	constexpr int num_stages() const { return num_stages_; }

	// Set number of active stages (for direct coefficient injection
	// without going through set_layout)
	constexpr void set_num_stages(int n) {
		assert(n >= 0 && n <= MaxStages);
		num_stages_ = n;
	}

	// Access a stage's coefficients
	constexpr const BiquadCoefficients<CoeffScalar>& stage(int index) const {
		assert(index >= 0 && index < num_stages_);
		return stages_[index];
	}

	constexpr BiquadCoefficients<CoeffScalar>& stage(int index) {
		assert(index >= 0 && index < num_stages_);
		return stages_[index];
	}

	// Access underlying array (for state arrays that need the same size)
	constexpr const std::array<BiquadCoefficients<CoeffScalar>, MaxStages>& stages() const {
		return stages_;
	}

	// Build cascade from a pole/zero layout.
	// Converts each PoleZeroPair to biquad coefficients and
	// normalizes gain so the response matches the layout's
	// specified normal_gain at normal_w.
	template <int MaxPoles>
	void set_layout(const PoleZeroLayout<CoeffScalar, MaxPoles>& layout) {
		num_stages_ = layout.num_pairs();
		assert(num_stages_ <= MaxStages);

		// Convert each pole/zero pair to biquad coefficients
		for (int i = 0; i < num_stages_; ++i) {
			stages_[i].set_from_pole_zero_pair(layout[i]);
		}

		// Initialize unused stages to identity
		for (int i = num_stages_; i < MaxStages; ++i) {
			stages_[i].set_identity();
		}

		// Normalize gain: evaluate cascade response at normal_w
		// and scale first stage so overall response matches normal_gain
		apply_scale(calc_normalization_scale(layout.normal_w(),
		                                     layout.normal_gain()));
	}

	// Process a single sample through all stages
	template <typename StateForm, DspScalar SampleScalar>
	SampleScalar process(SampleScalar in,
	                     std::array<StateForm, MaxStages>& state) const {
		SampleScalar out = in;
		for (int i = 0; i < num_stages_; ++i) {
			out = state[i].process(out, stages_[i]);
		}
		return out;
	}

	// Evaluate frequency response at normalized frequency f in [0, 0.5]
	std::complex<CoeffScalar> response(double normalized_freq) const {
		std::complex<CoeffScalar> result(CoeffScalar{1});
		for (int i = 0; i < num_stages_; ++i) {
			result = result * stages_[i].response(normalized_freq);
		}
		return result;
	}

private:
	// Apply gain scale to the first stage's numerator
	void apply_scale(CoeffScalar scale) {
		if (num_stages_ > 0) {
			stages_[0].apply_scale(scale);
		}
	}

	// Compute the scale factor needed to achieve desired gain at the
	// normalization frequency
	CoeffScalar calc_normalization_scale(CoeffScalar normal_w,
	                                     CoeffScalar normal_gain) const {
		// Evaluate the current response magnitude at normal_w
		double w = static_cast<double>(normal_w);
		double f = w / (2.0 * pi);  // convert angular freq to normalized freq

		auto current_response = response(f);
		using std::abs;  // ADL for Universal complex types
		double mag = static_cast<double>(abs(current_response));

		if (mag < 1e-30) {
			return CoeffScalar{1};  // avoid division by zero
		}

		return static_cast<CoeffScalar>(static_cast<double>(normal_gain) / mag);
	}

	std::array<BiquadCoefficients<CoeffScalar>, MaxStages> stages_{};
	int num_stages_{0};
};

} // namespace sw::dsp
