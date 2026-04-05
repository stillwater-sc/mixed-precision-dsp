#pragma once
// pole_zero_pair.hpp: a pair of poles and zeros for a second-order section
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/types/complex_pair.hpp>

namespace sw::dsp {

// A pair of pole/zeros that maps to a single biquad section.
// For a second-order section, poles and zeros are either:
//   - A conjugate pair (complex poles/zeros)
//   - A pair of real values
//   - A single real value (first-order section, second is zero)
template <DspField T>
struct PoleZeroPair {
	using complex_type = std::complex<T>;

	ComplexPair<T> poles;
	ComplexPair<T> zeros;

	constexpr PoleZeroPair() = default;

	// Single pole/zero (first-order section)
	constexpr PoleZeroPair(const complex_type& p, const complex_type& z)
		: poles(p), zeros(z) {}

	// Conjugate pair of poles/zeros (second-order section)
	constexpr PoleZeroPair(const complex_type& p1, const complex_type& z1,
	                       const complex_type& p2, const complex_type& z2)
		: poles(p1, p2), zeros(z1, z2) {}

	constexpr bool is_single_pole() const {
		return poles.second == complex_type{} && zeros.second == complex_type{};
	}

	constexpr bool is_nan() const {
		return poles.is_nan() || zeros.is_nan();
	}
};

} // namespace sw::dsp
