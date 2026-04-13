#pragma once
// filter.hpp: concepts for filter types
//
// These concepts formalize the requirements that SimpleFilter and
// ChannelFilter rely on, enabling generic algorithms that work
// with any conforming filter design.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <concepts>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

// A filter design that provides a biquad cascade.
// This is what SimpleFilter wraps: it has scalar type aliases,
// a max_stages constant, and a cascade() accessor.
template <typename F>
concept FilterDesign = requires(const F& f) {
	typename F::coeff_scalar;
	typename F::state_scalar;
	typename F::sample_scalar;
	{ F::max_stages } -> std::convertible_to<int>;
	{ f.cascade() };
};

// A filter that can be set up with order, sample rate, and cutoff frequency.
// Covers standard lowpass/highpass IIR designs (Butterworth, Chebyshev, etc.)
template <typename F>
concept DesignableLowPass = FilterDesign<F> && requires(F& f) {
	{ f.setup(1, 44100.0, 1000.0) };
};

// A filter that can be set up with order, sample rate, center frequency, and bandwidth.
// Covers bandpass/bandstop IIR designs.
template <typename F>
concept DesignableBandPass = FilterDesign<F> && requires(F& f) {
	{ f.setup(2, 44100.0, 1000.0, 500.0) };
};

// A processable filter: has process() for single samples and reset().
// This is what SimpleFilter provides after wrapping a FilterDesign.
template <typename F>
concept Processable = requires(F& f, typename F::sample_scalar s) {
	{ f.process(s) } -> std::convertible_to<typename F::sample_scalar>;
	{ f.reset() };
};

} // namespace sw::dsp
