#pragma once
// scalar.hpp: DSP scalar type concepts bridging to MTL5
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <concepts>
#include <type_traits>
#include <complex>

// Conditionally include Universal's complex when available.
// This enables complex_for_t<T> to dispatch to sw::universal::complex<T>
// for non-native types, since std::complex<T> is only defined for
// float, double, and long double per the C++ standard.
#if __has_include(<universal/math/complex.hpp>)
#include <universal/math/complex.hpp>
#define SW_DSP_HAS_UNIVERSAL_COMPLEX 1
#else
#define SW_DSP_HAS_UNIVERSAL_COMPLEX 0
#endif

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

// ============================================================================
// complex_for<T>: maps a scalar type to the correct complex type.
//
// For native IEEE types (float, double, long double): std::complex<T>
// For all other types (Universal posit, cfloat, fixpnt, etc.):
//   sw::universal::complex<T> if available, otherwise std::complex<T> (best effort)
//
// Usage: complex_for_t<double>        -> std::complex<double>
//        complex_for_t<posit<32,2>>   -> sw::universal::complex<posit<32,2>>
// ============================================================================

template <typename T, typename Enable = void>
struct complex_for {
	// Default: use std::complex<T> for native floating-point types
	using type = std::complex<T>;
};

#if SW_DSP_HAS_UNIVERSAL_COMPLEX
// For non-native types: use sw::universal::complex<T>
template <typename T>
struct complex_for<T, std::enable_if_t<!std::is_floating_point_v<T>>> {
	using type = sw::universal::complex<T>;
};
#endif

template <typename T>
using complex_for_t = typename complex_for<T>::type;

} // namespace sw::dsp
