#pragma once
// rls.hpp: Recursive Least Squares adaptive filter
//
// Faster convergence than LMS at the cost of higher computational
// complexity (O(N^2) vs O(N) per sample). Uses an exponentially
// weighted forgetting factor to track non-stationary signals.
//
// Update rule (matrix form):
//   k = (P * x) / (lambda + x^T * P * x)    (gain vector)
//   error = desired - w^T * x
//   w = w + k * error
//   P = (P - k * x^T * P) / lambda
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <stdexcept>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

// RLS adaptive filter.
//
// Template parameters:
//   T: scalar type
//
// Constructor parameters:
//   num_taps:  filter length
//   lambda:    forgetting factor in (0, 1], typically 0.99
//   delta:     initial value for P diagonal (large for fast convergence)
template <DspField T>
class RLSFilter {
public:
	using matrix_t = mtl::mat::dense2D<T>;
	using vector_t = mtl::vec::dense_vector<T>;

	RLSFilter(std::size_t num_taps, T lambda = T{0.99}, T delta = T{1000})
		: weights_(num_taps, T{}),
		  delay_(num_taps, T{}),
		  P_(num_taps, num_taps),
		  num_taps_(num_taps),
		  lambda_(lambda)
	{
		if (num_taps == 0)
			throw std::invalid_argument("RLSFilter: num_taps must be > 0");
		if (!(lambda > T{} && lambda <= T{1}))
			throw std::invalid_argument("RLSFilter: lambda must be in (0, 1]");

		// P = delta * I
		for (std::size_t i = 0; i < num_taps; ++i) {
			for (std::size_t j = 0; j < num_taps; ++j) {
				P_(i, j) = (i == j) ? delta : T{};
			}
		}
	}

	// Process one sample with adaptation.
	T process(T input, T desired) {
		// Shift delay line: x[k] = x[k-1] for k = num_taps-1 .. 1; x[0] = input
		for (std::size_t k = num_taps_ - 1; k > 0; --k) {
			delay_[k] = delay_[k - 1];
		}
		delay_[0] = input;

		// Compute output: y = w^T * x
		T output{};
		for (std::size_t k = 0; k < num_taps_; ++k) {
			output = output + weights_[k] * delay_[k];
		}

		// Error
		T error = desired - output;
		last_error_ = error;

		// pi = P * x
		vector_t pi(num_taps_, T{});
		for (std::size_t i = 0; i < num_taps_; ++i) {
			T sum{};
			for (std::size_t j = 0; j < num_taps_; ++j) {
				sum = sum + P_(i, j) * delay_[j];
			}
			pi[i] = sum;
		}

		// gamma = lambda + x^T * pi
		T gamma = lambda_;
		for (std::size_t i = 0; i < num_taps_; ++i) {
			gamma = gamma + delay_[i] * pi[i];
		}

		// gain k = pi / gamma
		vector_t gain(num_taps_, T{});
		for (std::size_t i = 0; i < num_taps_; ++i) {
			gain[i] = pi[i] / gamma;
		}

		// Update weights: w = w + k * error
		for (std::size_t i = 0; i < num_taps_; ++i) {
			weights_[i] = weights_[i] + gain[i] * error;
		}

		// Update P: P = (P - k * pi^T) / lambda
		// Note: pi = P * x, so k * pi^T = (P*x*x^T*P) / (lambda + x^T*P*x)
		T inv_lambda = T{1} / lambda_;
		for (std::size_t i = 0; i < num_taps_; ++i) {
			for (std::size_t j = 0; j < num_taps_; ++j) {
				P_(i, j) = (P_(i, j) - gain[i] * pi[j]) * inv_lambda;
			}
		}

		return output;
	}

	// Reset weights and P
	void reset(T delta = T{1000}) {
		for (std::size_t i = 0; i < num_taps_; ++i) {
			weights_[i] = T{};
			delay_[i] = T{};
			for (std::size_t j = 0; j < num_taps_; ++j) {
				P_(i, j) = (i == j) ? delta : T{};
			}
		}
		last_error_ = T{};
	}

	// Access
	const vector_t& weights() const { return weights_; }
	std::size_t num_taps() const { return num_taps_; }
	T last_error() const { return last_error_; }
	T lambda() const { return lambda_; }

private:
	vector_t weights_;
	vector_t delay_;
	matrix_t P_;
	std::size_t num_taps_;
	T lambda_;
	T last_error_{};
};

} // namespace sw::dsp
