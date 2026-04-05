#pragma once
// quadratic.hpp: quadratic equation solver for complex roots
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <complex>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

// Solve ax^2 + bx + c = 0, returning the root with the positive discriminant sign.
template <DspField T>
complex_for_t<T> solve_quadratic_1(T a, T b, T c) {
	using complex_t = complex_for_t<T>;
	using std::sqrt;
	return (complex_t(-b) + sqrt(complex_t(b * b - T{4} * a * c))) / (T{2} * a);
}

// Solve ax^2 + bx + c = 0, returning the root with the negative discriminant sign.
template <DspField T>
complex_for_t<T> solve_quadratic_2(T a, T b, T c) {
	using complex_t = complex_for_t<T>;
	using std::sqrt;
	return (complex_t(-b) - sqrt(complex_t(b * b - T{4} * a * c))) / (T{2} * a);
}

// Return both roots of ax^2 + bx + c = 0
template <DspField T>
std::pair<complex_for_t<T>, complex_for_t<T>> solve_quadratic(T a, T b, T c) {
	return {solve_quadratic_1(a, b, c), solve_quadratic_2(a, b, c)};
}

} // namespace sw::dsp
