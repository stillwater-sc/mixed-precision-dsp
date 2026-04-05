#pragma once
// denormal.hpp: traits-aware denormal prevention
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <limits>
#include <type_traits>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

// Traits to detect whether a number type has denormals.
// IEEE 754 float/double have denormals; posits and fixed-point do not.
template <typename T>
struct has_denormals : std::bool_constant<std::numeric_limits<T>::has_denorm != std::denorm_absent> {};

template <typename T>
inline constexpr bool has_denormals_v = has_denormals<T>::value;

// Denormal prevention policy.
//
// For IEEE 754 types: adds a small alternating signal to flush denormals.
// For posit/fixed-point types: no-op (these types have no denormals).
template <DspField T>
class DenormalPrevention {
public:
	// Returns a small value to add to accumulations to prevent denormals.
	// Alternates sign on each call (AC signal).
	T ac() {
		if constexpr (has_denormals_v<T>) {
			m_v = T{} - m_v;
			return m_v;
		} else {
			return T{};
		}
	}

	// Returns a small constant (DC signal) for denormal prevention.
	static constexpr T dc() {
		if constexpr (has_denormals_v<T>) {
			return T(1e-8);
		} else {
			return T{};
		}
	}

private:
	T m_v = has_denormals_v<T> ? T(1e-8) : T{};
};

} // namespace sw::dsp
