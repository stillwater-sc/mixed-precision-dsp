#pragma once
// complex_pair.hpp: a pair of complex values (conjugate or real pair)
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <complex>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

// A pair of complex numbers, typically representing a conjugate pair
// or a pair of real values. This is the building block for pole/zero
// representations that map directly to second-order sections.
template <DspField T>
struct ComplexPair {
	using complex_type = complex_for_t<T>;

	complex_type first{};
	complex_type second{};

	constexpr ComplexPair() = default;

	// Single real or complex value (second defaults to zero)
	constexpr explicit ComplexPair(const complex_type& c1)
		: first(c1), second(T{}) {}

	constexpr ComplexPair(const complex_type& c1, const complex_type& c2)
		: first(c1), second(c2) {}

	constexpr bool is_conjugate() const {
		return second == std::conj(first);
	}

	constexpr bool is_real() const {
		return first.imag() == T{} && second.imag() == T{};
	}

	// Returns true if this is either a conjugate pair,
	// or a pair of reals where neither is zero.
	constexpr bool is_matched_pair() const {
		if (first.imag() != T{}) {
			return second == std::conj(first);
		}
		return second.imag() == T{} &&
		       second.real() != T{} &&
		       first.real() != T{};
	}

	constexpr bool is_nan() const {
		return std::isnan(first.real()) || std::isnan(first.imag()) ||
		       std::isnan(second.real()) || std::isnan(second.imag());
	}
};

} // namespace sw::dsp
