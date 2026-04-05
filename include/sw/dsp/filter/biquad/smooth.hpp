#pragma once
// smooth.hpp: smoothed cascade for glitch-free parameter modulation
//
// Interpolates biquad coefficients over a transition window when
// parameters change, preventing audible clicks and discontinuities
// in real-time applications.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <span>
#include <sw/dsp/filter/biquad/cascade.hpp>
#include <sw/dsp/filter/biquad/state.hpp>

namespace sw::dsp {

// SmoothedCascade wraps a Cascade and interpolates coefficient
// changes over a configurable number of transition samples.
//
// Usage:
//   SmoothedCascade<double, 4> sc(1024);  // 1024-sample transition
//   sc.set_cascade(new_cascade);          // triggers smooth transition
//   sc.process_block(samples, state);     // interpolates during block
template <DspField CoeffScalar, int MaxStages>
class SmoothedCascade {
public:
	explicit SmoothedCascade(int transition_samples)
		: transition_samples_(transition_samples) {}

	// Set new target cascade. If this is the first call, applies
	// immediately. Otherwise starts a smooth transition.
	void set_cascade(const Cascade<CoeffScalar, MaxStages>& target) {
		if (!initialized_) {
			current_ = target;
			target_ = target;
			initialized_ = true;
			remaining_ = 0;
		} else {
			current_ = target_;  // snapshot where we are now
			target_ = target;
			remaining_ = transition_samples_;
		}
	}

	// Process a block of samples with smooth coefficient interpolation
	template <typename StateForm, DspScalar SampleScalar>
	void process_block(std::span<SampleScalar> samples,
	                   std::array<StateForm, MaxStages>& state) {
		if (remaining_ <= 0) {
			// No transition: use target directly
			for (auto& s : samples) {
				s = target_.process(s, state);
			}
			return;
		}

		int num_stages = target_.num_stages();

		for (std::size_t n = 0; n < samples.size(); ++n) {
			if (remaining_ > 0) {
				// Interpolate coefficients for this sample
				CoeffScalar t = static_cast<CoeffScalar>(1) /
				                static_cast<CoeffScalar>(remaining_);

				BiquadCoefficients<CoeffScalar> interp;
				SampleScalar out = samples[n];

				for (int i = 0; i < num_stages; ++i) {
					const auto& from = current_.stage(i);
					const auto& to = target_.stage(i);

					interp.b0 = from.b0 + (to.b0 - from.b0) * t;
					interp.b1 = from.b1 + (to.b1 - from.b1) * t;
					interp.b2 = from.b2 + (to.b2 - from.b2) * t;
					interp.a1 = from.a1 + (to.a1 - from.a1) * t;
					interp.a2 = from.a2 + (to.a2 - from.a2) * t;

					out = state[i].process(out, interp);
				}
				samples[n] = out;

				// Advance interpolation state
				--remaining_;
				if (remaining_ == 0) {
					current_ = target_;
				} else {
					// Update current_ toward target_ by one step
					for (int i = 0; i < num_stages; ++i) {
						auto& from = current_.stage(i);
						const auto& to = target_.stage(i);
						from.b0 = from.b0 + (to.b0 - from.b0) * t;
						from.b1 = from.b1 + (to.b1 - from.b1) * t;
						from.b2 = from.b2 + (to.b2 - from.b2) * t;
						from.a1 = from.a1 + (to.a1 - from.a1) * t;
						from.a2 = from.a2 + (to.a2 - from.a2) * t;
					}
				}
			} else {
				samples[n] = target_.process(samples[n], state);
			}
		}
	}

	const Cascade<CoeffScalar, MaxStages>& target() const { return target_; }
	bool in_transition() const { return remaining_ > 0; }

private:
	Cascade<CoeffScalar, MaxStages> current_{};
	Cascade<CoeffScalar, MaxStages> target_{};
	int transition_samples_;
	int remaining_{0};
	bool initialized_{false};
};

} // namespace sw::dsp
