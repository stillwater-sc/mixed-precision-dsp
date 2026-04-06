#pragma once
// ztransform.hpp: Z-transform evaluation
//
// Evaluate a TransferFunction<T> at arbitrary z-plane points.
// Useful for plotting frequency response at non-uniform frequencies,
// computing group delay, and analyzing filter behavior.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/types/transfer_function.hpp>

namespace sw::dsp::spectral {

// Evaluate a transfer function at N uniformly spaced frequencies [0, 0.5).
// Returns complex frequency response.
template <DspField T>
mtl::vec::dense_vector<complex_for_t<T>> freqz(
		const TransferFunction<T>& tf, std::size_t num_points = 512) {
	mtl::vec::dense_vector<complex_for_t<T>> H(num_points);
	for (std::size_t k = 0; k < num_points; ++k) {
		double f = static_cast<double>(k) / static_cast<double>(num_points) * 0.5;
		H[k] = tf.frequency_response(f);
	}
	return H;
}

// Evaluate a transfer function at specified z-plane points.
template <DspField T>
mtl::vec::dense_vector<complex_for_t<T>> evaluate_at(
		const TransferFunction<T>& tf,
		const mtl::vec::dense_vector<complex_for_t<T>>& z_points) {
	mtl::vec::dense_vector<complex_for_t<T>> H(z_points.size());
	for (std::size_t i = 0; i < z_points.size(); ++i) {
		H[i] = tf.evaluate(z_points[i]);
	}
	return H;
}

// Compute group delay at N uniformly spaced frequencies.
// Group delay = -d(phase)/d(omega), estimated by finite differences.
template <DspField T>
mtl::vec::dense_vector<double> group_delay(
		const TransferFunction<T>& tf, std::size_t num_points = 512) {
	mtl::vec::dense_vector<double> gd(num_points);
	double df = 0.5 / static_cast<double>(num_points);

	for (std::size_t k = 0; k < num_points; ++k) {
		double f = static_cast<double>(k) / static_cast<double>(num_points) * 0.5;
		double f_next = f + df * 0.01;  // small step for derivative
		double f_prev = f - df * 0.01;
		if (f_prev < 0) f_prev = 0;

		double phase_next = std::arg(tf.frequency_response(f_next));
		double phase_prev = std::arg(tf.frequency_response(f_prev));

		// Unwrap phase difference
		double dp = phase_next - phase_prev;
		while (dp > pi) dp -= two_pi;
		while (dp < -pi) dp += two_pi;

		double dw = two_pi * (f_next - f_prev);
		gd[k] = (dw > 0) ? -dp / dw : 0.0;
	}
	return gd;
}

} // namespace sw::dsp::spectral
