#pragma once
// decimator_step.hpp: uniform streaming dispatch across decimator APIs
//
// The library's decimators (CICDecimator, HalfBandFilter, PolyphaseDecimator)
// expose three different streaming interfaces:
//
//   - std::pair<bool, T> process(T)              (PolyphaseDecimator)
//   - std::pair<bool, T> process_decimate(T)     (HalfBandFilter)
//   - bool push(T); T output() const             (CICDecimator)
//
// Compositional classes (DDC, DecimationChain) need to step any of them
// without caring which API is used. step_decimator detects the available
// method via if-constexpr requirements and returns a uniform
// {ready, output} pair.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <concepts>
#include <utility>

namespace sw::dsp::detail {

template <class Decimator, class Sample>
std::pair<bool, Sample> step_decimator(Decimator& d, Sample in) {
	if constexpr (requires { { d.process(in) } -> std::convertible_to<std::pair<bool, Sample>>; }) {
		return d.process(in);
	} else if constexpr (requires { { d.process_decimate(in) } -> std::convertible_to<std::pair<bool, Sample>>; }) {
		return d.process_decimate(in);
	} else if constexpr (requires { { d.push(in) } -> std::convertible_to<bool>; d.output(); }) {
		bool ready = d.push(in);
		if (ready) return {true, d.output()};
		return {false, Sample{}};
	} else {
		static_assert(sizeof(Decimator) == 0,
			"step_decimator: Decimator must have process(), process_decimate(), or push()/output()");
	}
}

} // namespace sw::dsp::detail
