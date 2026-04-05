#pragma once
// layout.hpp: pole/zero layout for filter design
//
// A fixed-capacity container of PoleZeroPair that describes a filter
// in either the s-plane (analog prototype) or z-plane (digital).
// Replaces the pointer-based LayoutBase from DSPFilters with a
// value-semantic std::array.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <array>
#include <cassert>
#include <complex>
#include <sw/dsp/types/pole_zero_pair.hpp>

namespace sw::dsp {

// PoleZeroLayout stores pole/zero pairs for a filter of up to MaxPoles poles.
// Conjugate pairs count as 2 poles but occupy 1 slot in the pairs array.
// A single (odd) real pole also occupies 1 slot.
//
// Usage:
//   PoleZeroLayout<double, 8> layout;   // up to 8th order
//   layout.add_conjugate_pairs(pole, zero);
//   layout.add(real_pole, real_zero);    // single pole (must come last)
template <DspField T, int MaxPoles>
class PoleZeroLayout {
public:
	static constexpr int max_pairs = (MaxPoles + 1) / 2;

	constexpr PoleZeroLayout() = default;

	// Reset to empty
	constexpr void reset() { num_poles_ = 0; }

	// Number of poles currently stored
	constexpr int num_poles() const { return num_poles_; }
	constexpr int max_poles() const { return MaxPoles; }

	// Number of PoleZeroPair slots used
	constexpr int num_pairs() const { return (num_poles_ + 1) / 2; }

	// Add a single real pole/zero (first-order section).
	// Must be the last addition (odd-order term).
	constexpr void add(const std::complex<T>& pole, const std::complex<T>& zero) {
		assert(!(num_poles_ & 1));  // single must come last
		int idx = num_poles_ / 2;
		assert(idx < max_pairs);
		pairs_[idx] = PoleZeroPair<T>(pole, zero);
		++num_poles_;
	}

	// Add a conjugate pair of poles/zeros (second-order section).
	constexpr void add_conjugate_pairs(const std::complex<T>& pole,
	                                   const std::complex<T>& zero) {
		assert(!(num_poles_ & 1));  // single must come last
		int idx = num_poles_ / 2;
		assert(idx < max_pairs);
		pairs_[idx] = PoleZeroPair<T>(pole, zero, std::conj(pole), std::conj(zero));
		num_poles_ += 2;
	}

	// Add an explicit pair of poles and zeros
	constexpr void add(const ComplexPair<T>& poles, const ComplexPair<T>& zeros) {
		assert(!(num_poles_ & 1));
		assert(poles.is_matched_pair());
		assert(zeros.is_matched_pair());
		int idx = num_poles_ / 2;
		assert(idx < max_pairs);
		pairs_[idx] = PoleZeroPair<T>(poles.first, zeros.first,
		                              poles.second, zeros.second);
		num_poles_ += 2;
	}

	// Access pair by index
	constexpr const PoleZeroPair<T>& operator[](int pair_index) const {
		assert(pair_index >= 0 && pair_index < num_pairs());
		return pairs_[pair_index];
	}

	constexpr PoleZeroPair<T>& operator[](int pair_index) {
		assert(pair_index >= 0 && pair_index < num_pairs());
		return pairs_[pair_index];
	}

	// Normalization: frequency and gain at which the filter response
	// is normalized (e.g., unity gain at DC for lowpass).
	constexpr T normal_w() const { return normal_w_; }
	constexpr T normal_gain() const { return normal_gain_; }

	constexpr void set_normal(T w, T gain) {
		normal_w_ = w;
		normal_gain_ = gain;
	}

private:
	std::array<PoleZeroPair<T>, max_pairs> pairs_{};
	int num_poles_{0};
	T normal_w_{};
	T normal_gain_{1};
};

} // namespace sw::dsp
