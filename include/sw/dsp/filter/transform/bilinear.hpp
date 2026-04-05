#pragma once
// bilinear.hpp: bilinear s-plane to z-plane transforms (lowpass, highpass)
//
// Transforms an analog prototype (s-plane pole/zero layout) to a digital
// filter (z-plane pole/zero layout) using the bilinear transform with
// frequency prewarping.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <complex>
#include <limits>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/filter/layout/layout.hpp>

namespace sw::dsp {

// Low-pass bilinear transform.
//
// Maps an analog lowpass prototype to a digital lowpass filter
// with cutoff at the specified normalized frequency fc = freq/sample_rate.
//
// The bilinear transform: z = (1 + s/f) / (1 - s/f)
// where f = tan(pi * fc) is the prewarped frequency.
template <DspField T>
class LowPassTransform {
public:
	// Transform analog layout into digital layout for lowpass at fc
	template <int MaxAnalog, int MaxDigital>
	LowPassTransform(T fc,
	                  PoleZeroLayout<T, MaxDigital>& digital,
	                  const PoleZeroLayout<T, MaxAnalog>& analog) {
		digital.reset();

		// Prewarp
		f_ = std::tan(pi_v<T> * fc);

		const int num_poles = analog.num_poles();
		const int pairs = num_poles / 2;

		for (int i = 0; i < pairs; ++i) {
			const auto& pair = analog[i];
			digital.add_conjugate_pairs(
				transform(pair.poles.first),
				transform(pair.zeros.first));
		}

		if (num_poles & 1) {
			const auto& pair = analog[pairs];
			digital.add(
				transform(pair.poles.first),
				transform(pair.zeros.first));
		}

		digital.set_normal(analog.normal_w(), analog.normal_gain());
	}

private:
	std::complex<T> transform(std::complex<T> c) const {
		if (c.real() == std::numeric_limits<T>::infinity()) {
			return std::complex<T>(T{-1}, T{});
		}
		c = f_ * c;
		return (T{1} + c) / (T{1} - c);
	}

	T f_;
};

// High-pass bilinear transform.
//
// Maps an analog lowpass prototype to a digital highpass filter.
// The sign flip in the bilinear mapping mirrors the frequency response.
template <DspField T>
class HighPassTransform {
public:
	template <int MaxAnalog, int MaxDigital>
	HighPassTransform(T fc,
	                  PoleZeroLayout<T, MaxDigital>& digital,
	                  const PoleZeroLayout<T, MaxAnalog>& analog) {
		digital.reset();

		// Prewarp (reciprocal for highpass)
		f_ = T{1} / std::tan(pi_v<T> * fc);

		const int num_poles = analog.num_poles();
		const int pairs = num_poles / 2;

		for (int i = 0; i < pairs; ++i) {
			const auto& pair = analog[i];
			digital.add_conjugate_pairs(
				transform(pair.poles.first),
				transform(pair.zeros.first));
		}

		if (num_poles & 1) {
			const auto& pair = analog[pairs];
			digital.add(
				transform(pair.poles.first),
				transform(pair.zeros.first));
		}

		digital.set_normal(pi_v<T> - analog.normal_w(), analog.normal_gain());
	}

private:
	std::complex<T> transform(std::complex<T> c) const {
		if (c.real() == std::numeric_limits<T>::infinity()) {
			return std::complex<T>(T{1}, T{});
		}
		c = f_ * c;
		return T{-1} * (T{1} + c) / (T{1} - c);
	}

	T f_;
};

} // namespace sw::dsp
