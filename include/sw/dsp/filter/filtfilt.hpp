#pragma once
// filtfilt.hpp: zero-phase forward-backward filtering
//
// Applies a biquad cascade forward then backward, cancelling phase
// distortion. The magnitude response is squared relative to a
// single pass. Signal edges are reflected to reduce transients.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <array>
#include <cstddef>
#include <algorithm>
#include <vector>
#include <sw/dsp/filter/biquad/cascade.hpp>
#include <sw/dsp/filter/biquad/state.hpp>

namespace sw::dsp {

// filtfilt: zero-phase filtering via forward-backward cascade processing.
//
// Template parameters:
//   StateForm   — biquad state realization (DirectFormI, DirectFormII, etc.)
//   CoeffScalar — coefficient precision
//   MaxStages   — maximum biquad stages in the cascade
//   SampleScalar — sample I/O precision
//
// The algorithm:
//   1. Reflect the signal at both ends to reduce edge transients
//   2. Filter forward through the cascade
//   3. Reverse the result and filter forward again (= backward pass)
//   4. Reverse and extract the central N samples
template <typename StateForm,
          DspField CoeffScalar, int MaxStages,
          DspScalar SampleScalar>
std::vector<SampleScalar> filtfilt(
    const Cascade<CoeffScalar, MaxStages>& cascade,
    const std::vector<SampleScalar>& input) {

	const std::size_t N = input.size();
	if (N == 0) return {};

	const int ns = cascade.num_stages();
	if (ns == 0) return std::vector<SampleScalar>(input.begin(), input.end());

	// Reflection length: 3 * (2 * num_stages + 1) - 1
	// Clamped to N-1 if input is shorter than the reflection length
	std::size_t nrefl = static_cast<std::size_t>(3 * (2 * ns + 1) - 1);
	if (nrefl >= N) nrefl = N - 1;

	if (nrefl == 0) {
		// Signal too short to reflect — just do forward-backward without extension
		std::vector<SampleScalar> out(input.begin(), input.end());

		std::array<StateForm, MaxStages> state{};
		for (std::size_t i = 0; i < N; ++i)
			out[i] = cascade.process(out[i], state);

		std::reverse(out.begin(), out.end());
		for (auto& s : state) s.reset();
		for (std::size_t i = 0; i < N; ++i)
			out[i] = cascade.process(out[i], state);

		std::reverse(out.begin(), out.end());
		return out;
	}

	// Build extended signal: [reflected_front | original | reflected_back]
	// Front reflection: 2*x[0] - x[nrefl], ..., 2*x[0] - x[1]
	// Back reflection:  2*x[N-1] - x[N-2], ..., 2*x[N-1] - x[N-1-nrefl]
	const std::size_t ext_len = nrefl + N + nrefl;
	std::vector<SampleScalar> ext(ext_len);

	SampleScalar x0 = input[0];
	SampleScalar xN = input[N - 1];
	SampleScalar two{2};

	for (std::size_t i = 0; i < nrefl; ++i)
		ext[i] = two * x0 - input[nrefl - i];

	for (std::size_t i = 0; i < N; ++i)
		ext[nrefl + i] = input[i];

	for (std::size_t i = 0; i < nrefl; ++i)
		ext[nrefl + N + i] = two * xN - input[N - 2 - i];

	// Forward pass
	std::array<StateForm, MaxStages> state{};
	for (std::size_t i = 0; i < ext_len; ++i)
		ext[i] = cascade.process(ext[i], state);

	// Reverse
	std::reverse(ext.begin(), ext.end());

	// Backward pass (forward on reversed signal)
	for (auto& s : state) s.reset();
	for (std::size_t i = 0; i < ext_len; ++i)
		ext[i] = cascade.process(ext[i], state);

	// Reverse back and extract central N samples
	std::reverse(ext.begin(), ext.end());

	return std::vector<SampleScalar>(ext.begin() + static_cast<std::ptrdiff_t>(nrefl),
	                                 ext.begin() + static_cast<std::ptrdiff_t>(nrefl + N));
}

// Convenience overload defaulting to DirectFormII
template <DspField CoeffScalar, int MaxStages,
          DspScalar SampleScalar>
std::vector<SampleScalar> filtfilt(
    const Cascade<CoeffScalar, MaxStages>& cascade,
    const std::vector<SampleScalar>& input) {
	return filtfilt<DirectFormII<CoeffScalar>, CoeffScalar, MaxStages, SampleScalar>(
	    cascade, input);
}

} // namespace sw::dsp
