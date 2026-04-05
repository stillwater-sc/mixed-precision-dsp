#pragma once
// filter.hpp: convenience wrappers for filter design + state
//
// SimpleFilter combines a filter design (which provides a Cascade)
// with state arrays, giving a self-contained object that can
// process samples without external state management.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <array>
#include <span>
#include <sw/dsp/filter/biquad/cascade.hpp>
#include <sw/dsp/filter/biquad/state.hpp>

namespace sw::dsp {

// SimpleFilter bundles a filter design with processing state.
//
// FilterDesign must provide:
//   - static constexpr int max_stages
//   - using coeff_scalar = ...
//   - using state_scalar = ...
//   - using sample_scalar = ...
//   - const Cascade<coeff_scalar, max_stages>& cascade() const
//
// StateForm defaults to DirectFormII<state_scalar>.
//
// Usage:
//   SimpleFilter<iir::ButterworthLowPass<4>> f;
//   f.setup(4, 44100.0, 1000.0);
//   auto y = f.process(x);
template <typename FilterDesign,
          typename StateForm = DirectFormII<typename FilterDesign::state_scalar>>
class SimpleFilter : public FilterDesign {
public:
	using coeff_scalar  = typename FilterDesign::coeff_scalar;
	using state_scalar  = typename FilterDesign::state_scalar;
	using sample_scalar = typename FilterDesign::sample_scalar;

	static constexpr int max_stages = FilterDesign::max_stages;

	SimpleFilter() = default;

	// Process a single sample
	sample_scalar process(sample_scalar in) {
		return this->cascade().process(in, state_);
	}

	// Process a block of samples in-place
	void process_block(std::span<sample_scalar> samples) {
		for (auto& s : samples) {
			s = this->cascade().process(s, state_);
		}
	}

	// Reset all state to zero
	void reset() {
		for (auto& s : state_) {
			s.reset();
		}
	}

	// Access state (for inspection or external management)
	std::array<StateForm, max_stages>& state() { return state_; }
	const std::array<StateForm, max_stages>& state() const { return state_; }

private:
	std::array<StateForm, max_stages> state_{};
};

// ChannelFilter processes multiple channels with independent state.
// Channels is a compile-time channel count.
template <typename FilterDesign,
          int Channels,
          typename StateForm = DirectFormII<typename FilterDesign::state_scalar>>
class ChannelFilter : public FilterDesign {
public:
	using coeff_scalar  = typename FilterDesign::coeff_scalar;
	using state_scalar  = typename FilterDesign::state_scalar;
	using sample_scalar = typename FilterDesign::sample_scalar;

	static constexpr int max_stages = FilterDesign::max_stages;
	static constexpr int num_channels = Channels;

	// Process a single sample on a specific channel
	sample_scalar process(int channel, sample_scalar in) {
		return this->cascade().process(in, state_[channel]);
	}

	// Process a block on a specific channel
	void process_block(int channel, std::span<sample_scalar> samples) {
		for (auto& s : samples) {
			s = this->cascade().process(s, state_[channel]);
		}
	}

	// Reset all channels
	void reset() {
		for (auto& channel_state : state_) {
			for (auto& s : channel_state) {
				s.reset();
			}
		}
	}

private:
	std::array<std::array<StateForm, max_stages>, Channels> state_{};
};

} // namespace sw::dsp
