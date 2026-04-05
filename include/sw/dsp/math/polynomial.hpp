#pragma once
// polynomial.hpp: polynomial evaluation and manipulation
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <complex>
#include <vector>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

// Evaluate polynomial using Horner's method.
//
// p(x) = coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ... + coeffs[n]*x^n
//
// Coefficients are in ascending order of power.
template <DspField T>
T evaluate_polynomial(const std::vector<T>& coeffs, T x) {
	if (coeffs.empty()) return T{};

	T result = coeffs.back();
	for (int i = static_cast<int>(coeffs.size()) - 2; i >= 0; --i) {
		result = result * x + coeffs[static_cast<std::size_t>(i)];
	}
	return result;
}

// Evaluate polynomial at a complex point using Horner's method.
template <DspField T>
std::complex<T> evaluate_polynomial(const std::vector<T>& coeffs, std::complex<T> x) {
	if (coeffs.empty()) return std::complex<T>{};

	std::complex<T> result(coeffs.back());
	for (int i = static_cast<int>(coeffs.size()) - 2; i >= 0; --i) {
		result = result * x + std::complex<T>(coeffs[static_cast<std::size_t>(i)]);
	}
	return result;
}

// Evaluate a complex-coefficient polynomial at a complex point.
template <DspField T>
std::complex<T> evaluate_polynomial(const std::vector<std::complex<T>>& coeffs,
                                    std::complex<T> x) {
	if (coeffs.empty()) return std::complex<T>{};

	std::complex<T> result = coeffs.back();
	for (int i = static_cast<int>(coeffs.size()) - 2; i >= 0; --i) {
		result = result * x + coeffs[static_cast<std::size_t>(i)];
	}
	return result;
}

// Multiply two polynomials (convolution of coefficient vectors).
// Result has degree = deg(a) + deg(b).
template <DspField T>
std::vector<T> multiply_polynomials(const std::vector<T>& a, const std::vector<T>& b) {
	if (a.empty() || b.empty()) return {};

	std::vector<T> result(a.size() + b.size() - 1, T{});
	for (std::size_t i = 0; i < a.size(); ++i) {
		for (std::size_t j = 0; j < b.size(); ++j) {
			result[i + j] = result[i + j] + a[i] * b[j];
		}
	}
	return result;
}

} // namespace sw::dsp
