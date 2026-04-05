#pragma once
// scalar.hpp: DSP scalar type concepts bridging to MTL5
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <concepts>
#include <type_traits>
#include <complex>

namespace sw::dsp {

// A type that supports basic arithmetic and value initialization.
// This is the minimal requirement for sample data types.
template <typename T>
concept DspScalar = requires(T a, T b) {
	{ a + b } -> std::convertible_to<T>;
	{ a - b } -> std::convertible_to<T>;
	{ a * b } -> std::convertible_to<T>;
	{ -a }    -> std::convertible_to<T>;
	{ T{} };      // default constructible (zero)
};

// A field type: DspScalar with division.
// Required for filter coefficients and state variables.
template <typename T>
concept DspField = DspScalar<T> && requires(T a, T b) {
	{ a / b } -> std::convertible_to<T>;
};

// An ordered field: DspField with total ordering.
// Required for comparison-based algorithms (thresholding, clamping).
template <typename T>
concept DspOrderedField = DspField<T> && std::totally_ordered<T>;

// Detects complex number types (std::complex<T> or sw::universal::complex<T>)
template <typename T>
struct is_complex : std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};

template <typename T>
constexpr bool is_complex_v = is_complex<T>::value;

template <typename T>
concept ComplexType = requires(T z) {
	{ z.real() };
	{ z.imag() };
	{ std::conj(z) } -> std::convertible_to<T>;
};

// Trait to extract the real scalar type from a potentially complex type
template <typename T>
struct real_type { using type = T; };

template <typename T>
struct real_type<std::complex<T>> { using type = T; };

template <typename T>
using real_type_t = typename real_type<T>::type;

// Concept for types that can be converted to double (needed for design-time calculations)
template <typename T>
concept ConvertibleToDouble = requires(T a) {
	{ static_cast<double>(a) } -> std::convertible_to<double>;
};

} // namespace sw::dsp
