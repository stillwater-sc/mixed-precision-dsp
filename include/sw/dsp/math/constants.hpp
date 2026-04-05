#pragma once
// constants.hpp: mathematical constants as constexpr templates
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

namespace sw::dsp {

// Mathematical constants templated on scalar type for precision-correct values.

template <typename T>
inline constexpr T pi_v = T(3.14159265358979323846264338327950288419716939937510L);

template <typename T>
inline constexpr T two_pi_v = T(6.28318530717958647692528676655900576839433879875021L);

template <typename T>
inline constexpr T half_pi_v = T(1.57079632679489661923132169163975144209858469968755L);

template <typename T>
inline constexpr T ln2_v = T(0.69314718055994530941723212145817656807550013436026L);

template <typename T>
inline constexpr T ln10_v = T(2.30258509299404568401799145468436420760110148862877L);

template <typename T>
inline constexpr T sqrt2_v = T(1.41421356237309504880168872420969807856967187537694L);

template <typename T>
inline constexpr T inv_sqrt2_v = T(0.70710678118654752440084436210484903928483593768847L);

// Convenience aliases for double
inline constexpr double pi       = pi_v<double>;
inline constexpr double two_pi   = two_pi_v<double>;
inline constexpr double half_pi  = half_pi_v<double>;
inline constexpr double ln2      = ln2_v<double>;
inline constexpr double ln10     = ln10_v<double>;
inline constexpr double sqrt2    = sqrt2_v<double>;
inline constexpr double inv_sqrt2 = inv_sqrt2_v<double>;

} // namespace sw::dsp
