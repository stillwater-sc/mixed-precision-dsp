#pragma once
// signal.hpp: Signal<T> wrapper with sample rate metadata
//
// Wraps mtl::vec::dense_vector<T> with sample rate, providing
// a self-describing signal container for DSP workflows.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <span>
#include <stdexcept>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

template <DspField T>
class Signal {
public:
	Signal() = default;

	Signal(std::size_t length, double sample_rate)
		: data_(length), sample_rate_(sample_rate) {
		if (!(sample_rate > 0.0)) throw std::invalid_argument("Signal: sample_rate must be > 0");
	}

	Signal(mtl::vec::dense_vector<T> data, double sample_rate)
		: data_(std::move(data)), sample_rate_(sample_rate) {
		if (!(sample_rate > 0.0)) throw std::invalid_argument("Signal: sample_rate must be > 0");
	}

	// Access
	T& operator[](std::size_t i) { return data_[i]; }
	const T& operator[](std::size_t i) const { return data_[i]; }

	std::size_t size() const { return data_.size(); }
	double sample_rate() const { return sample_rate_; }
	double duration() const { return static_cast<double>(size()) / sample_rate_; }

	// Underlying data
	mtl::vec::dense_vector<T>& data() { return data_; }
	const mtl::vec::dense_vector<T>& data() const { return data_; }

	// Span access for interop
	std::span<T> span() { return {data_.data(), data_.size()}; }
	std::span<const T> span() const { return {data_.data(), data_.size()}; }

	// Iterators (delegate to dense_vector)
	auto begin() { return data_.begin(); }
	auto end() { return data_.end(); }
	auto begin() const { return data_.begin(); }
	auto end() const { return data_.end(); }

private:
	mtl::vec::dense_vector<T> data_;
	double sample_rate_{44100.0};
};

} // namespace sw::dsp
