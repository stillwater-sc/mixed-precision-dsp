#pragma once
// channel_aligner.hpp: Multi-channel time alignment for instrument-style
// data acquisition.
//
// Aligns N channels that share a sample rate but have fixed inter-channel
// time skews. Channel 0 is the reference (skew = 0); each other channel's
// skew is the offset that would make IT line up with channel 0 — a
// positive skew means the channel is sampled LATER than channel 0.
//
// Implementation: one FractionalDelay per non-reference channel, with the
// integer part of the skew handled by the caller (via ring-buffer offset)
// and the fractional part [0, 1) routed through the FractionalDelay's
// windowed-sinc FIR.
//
// Real instruments need this for things like:
//   - Multi-ADC scopes where each ADC has a slightly different clock
//     latency (factory-calibrated skew)
//   - Differential probe pairs with cable-length mismatches
//   - Multiple independent capture cards on a shared trigger
//
// This issue handles the STATIC case (skews known at construction).
// Dynamic skew estimation (e.g., cross-correlation-based) is a follow-up.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/instrument/fractional_delay.hpp>

namespace sw::dsp::instrument {

template <DspField CoeffScalar  = double,
          DspField StateScalar  = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class ChannelAligner {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;

	// skews: per-channel fractional skew in samples, in [0, 1). Channel 0
	//   is the reference (skews[0] should be 0; nonzero is rejected to
	//   prevent the user accidentally double-correcting).
	// num_taps: FIR length passed through to each FractionalDelay.
	explicit ChannelAligner(std::span<const double> skews,
	                        std::size_t num_taps = 31) {
		if (skews.empty())
			throw std::invalid_argument(
				"ChannelAligner: at least one channel required");
		if (skews[0] != 0.0)
			throw std::invalid_argument(
				"ChannelAligner: channel 0 is the reference; "
				"skews[0] must be 0.0");
		// Channel 0: no delay needed; we keep a no-op stand-in via a
		// FractionalDelay(0.0, num_taps) so all channels go through the
		// same length of pipeline (matters for group-delay matching).
		// Each channel gets its own FractionalDelay — the per-channel
		// skew tells the FractionalDelay what fractional offset to
		// compensate.
		delays_.reserve(skews.size());
		for (double s : skews) {
			delays_.emplace_back(s, num_taps);
		}
	}

	// Push one sample per channel; returns the aligned samples in the
	// same channel order. The output has the same length as the input.
	//
	// Note on group delay: every channel's output is delayed by the FIR's
	// (N-1)/2 samples PLUS its individual fractional skew compensation.
	// Channel 0's compensation is 0 so its group delay is exactly
	// (N-1)/2; other channels' group delay is (N-1)/2 + their_skew. The
	// design ensures all channels have THE SAME group delay AT THE
	// reference time grid — the whole point of alignment.
	std::vector<SampleScalar>
	process(std::span<const SampleScalar> channel_samples) {
		if (channel_samples.size() != delays_.size())
			throw std::invalid_argument(
				"ChannelAligner::process: input length must equal "
				"number of channels");
		std::vector<SampleScalar> out(channel_samples.size());
		for (std::size_t c = 0; c < delays_.size(); ++c) {
			out[c] = delays_[c].process(channel_samples[c]);
		}
		return out;
	}

	std::size_t num_channels() const { return delays_.size(); }

private:
	std::vector<FractionalDelay<CoeffScalar, StateScalar, SampleScalar>> delays_;
};

} // namespace sw::dsp::instrument
