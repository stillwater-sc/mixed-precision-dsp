#pragma once
// lms.hpp: Least Mean Squares adaptive filter
//
// Online adaptation of FIR tap weights to minimize the mean
// squared error between a desired signal and the filter output.
//
// Update rule:
//   error = desired - output
//   w[k] = w[k] + mu * error * x[k]    for each tap k
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <stdexcept>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

// LMS adaptive filter.
//
// Template parameters:
//   T:  scalar type
//
// Usage:
//   LMSFilter<double> lms(num_taps, step_size);
//   double y = lms.process(input, desired);
template <DspField T>
class LMSFilter {
public:
	LMSFilter(std::size_t num_taps, T step_size)
		: weights_(num_taps, T{}),
		  delay_(num_taps, T{}),
		  step_size_(step_size),
		  num_taps_(num_taps),
		  write_pos_(0)
	{
		if (num_taps == 0)
			throw std::invalid_argument("LMSFilter: num_taps must be > 0");
	}

	// Process one sample with adaptation.
	// input:   the reference input signal x[n]
	// desired: the desired signal d[n]
	// Returns: the filter output y[n]
	// After processing, the error e[n] = d[n] - y[n] is available.
	T process(T input, T desired) {
		// Insert input into circular delay line
		delay_[write_pos_] = input;

		// Compute output: y = sum(w[k] * x[n-k])
		T output{};
		std::size_t idx = write_pos_;
		for (std::size_t k = 0; k < num_taps_; ++k) {
			output = output + weights_[k] * delay_[idx];
			idx = (idx == 0) ? num_taps_ - 1 : idx - 1;
		}

		// Compute error
		T error = desired - output;
		last_error_ = error;

		// Update weights: w[k] += mu * e * x[n-k]
		idx = write_pos_;
		for (std::size_t k = 0; k < num_taps_; ++k) {
			weights_[k] = weights_[k] + step_size_ * error * delay_[idx];
			idx = (idx == 0) ? num_taps_ - 1 : idx - 1;
		}

		// Advance write position
		write_pos_ = (write_pos_ + 1) % num_taps_;

		return output;
	}

	// Process without adapting (use frozen weights)
	T predict(T input) {
		delay_[write_pos_] = input;
		T output{};
		std::size_t idx = write_pos_;
		for (std::size_t k = 0; k < num_taps_; ++k) {
			output = output + weights_[k] * delay_[idx];
			idx = (idx == 0) ? num_taps_ - 1 : idx - 1;
		}
		write_pos_ = (write_pos_ + 1) % num_taps_;
		return output;
	}

	// Reset weights and delay line to zero
	void reset() {
		for (std::size_t i = 0; i < num_taps_; ++i) {
			weights_[i] = T{};
			delay_[i] = T{};
		}
		write_pos_ = 0;
		last_error_ = T{};
	}

	// Access
	const mtl::vec::dense_vector<T>& weights() const { return weights_; }
	mtl::vec::dense_vector<T>& weights() { return weights_; }
	std::size_t num_taps() const { return num_taps_; }
	T step_size() const { return step_size_; }
	T last_error() const { return last_error_; }

	void set_step_size(T mu) { step_size_ = mu; }

private:
	mtl::vec::dense_vector<T> weights_;
	mtl::vec::dense_vector<T> delay_;
	T step_size_;
	std::size_t num_taps_;
	std::size_t write_pos_;
	T last_error_{};
};

// Normalized LMS variant: scales step size by input power.
// Provides better stability across varying input levels.
template <DspField T>
class NLMSFilter {
public:
	NLMSFilter(std::size_t num_taps, T step_size, T epsilon = T{1e-6})
		: weights_(num_taps, T{}),
		  delay_(num_taps, T{}),
		  step_size_(step_size),
		  epsilon_(epsilon),
		  num_taps_(num_taps),
		  write_pos_(0)
	{
		if (num_taps == 0)
			throw std::invalid_argument("NLMSFilter: num_taps must be > 0");
	}

	T process(T input, T desired) {
		delay_[write_pos_] = input;

		T output{};
		T power{};
		std::size_t idx = write_pos_;
		for (std::size_t k = 0; k < num_taps_; ++k) {
			output = output + weights_[k] * delay_[idx];
			power = power + delay_[idx] * delay_[idx];
			idx = (idx == 0) ? num_taps_ - 1 : idx - 1;
		}

		T error = desired - output;
		last_error_ = error;

		// Normalized step: mu / (epsilon + power)
		T norm_mu = step_size_ / (epsilon_ + power);

		idx = write_pos_;
		for (std::size_t k = 0; k < num_taps_; ++k) {
			weights_[k] = weights_[k] + norm_mu * error * delay_[idx];
			idx = (idx == 0) ? num_taps_ - 1 : idx - 1;
		}

		write_pos_ = (write_pos_ + 1) % num_taps_;
		return output;
	}

	void reset() {
		for (std::size_t i = 0; i < num_taps_; ++i) { weights_[i] = T{}; delay_[i] = T{}; }
		write_pos_ = 0;
		last_error_ = T{};
	}

	const mtl::vec::dense_vector<T>& weights() const { return weights_; }
	std::size_t num_taps() const { return num_taps_; }
	T last_error() const { return last_error_; }

private:
	mtl::vec::dense_vector<T> weights_;
	mtl::vec::dense_vector<T> delay_;
	T step_size_;
	T epsilon_;
	std::size_t num_taps_;
	std::size_t write_pos_;
	T last_error_{};
};

} // namespace sw::dsp
