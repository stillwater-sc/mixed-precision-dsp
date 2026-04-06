#pragma once
// laplace.hpp: Laplace transform evaluation
//
// Evaluate continuous-time transfer functions H(s) at arbitrary
// s-plane points. Useful for analyzing analog prototype filters.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>

namespace sw::dsp::spectral {

// Continuous-time transfer function H(s) = N(s) / D(s)
// Numerator: n0 + n1*s + n2*s^2 + ... (ascending powers)
// Denominator: d0 + d1*s + d2*s^2 + ... (ascending powers)
template <DspField T>
struct ContinuousTransferFunction {
	mtl::vec::dense_vector<T> numerator;   // ascending powers of s
	mtl::vec::dense_vector<T> denominator; // ascending powers of s

	// Evaluate H(s) at a complex s-plane point
	complex_for_t<T> evaluate(complex_for_t<T> s) const {
		using complex_t = complex_for_t<T>;

		complex_t s_pow(T{1});
		complex_t num_val{};
		for (std::size_t i = 0; i < numerator.size(); ++i) {
			num_val = num_val + complex_t(numerator[i]) * s_pow;
			s_pow = s_pow * s;
		}

		s_pow = complex_t(T{1});
		complex_t den_val{};
		for (std::size_t i = 0; i < denominator.size(); ++i) {
			den_val = den_val + complex_t(denominator[i]) * s_pow;
			s_pow = s_pow * s;
		}

		return num_val / den_val;
	}

	// Evaluate frequency response H(jw)
	complex_for_t<T> frequency_response(double omega) const {
		using complex_t = complex_for_t<T>;
		complex_t jw(T{}, static_cast<T>(omega));
		return evaluate(jw);
	}
};

// Evaluate H(jw) at N uniformly spaced angular frequencies [0, omega_max).
template <DspField T>
mtl::vec::dense_vector<complex_for_t<T>> freqs(
		const ContinuousTransferFunction<T>& tf,
		double omega_max, std::size_t num_points = 512) {
	mtl::vec::dense_vector<complex_for_t<T>> H(num_points);
	for (std::size_t k = 0; k < num_points; ++k) {
		double omega = omega_max * static_cast<double>(k) / static_cast<double>(num_points);
		H[k] = tf.frequency_response(omega);
	}
	return H;
}

} // namespace sw::dsp::spectral
