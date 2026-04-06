#pragma once
// dither.hpp: dithering for quantization noise shaping
//
// TPDF (triangular probability density function) and RPDF (rectangular)
// dithering add small noise before quantization to decorrelate
// quantization error from the signal, converting distortion into
// flat noise floor.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <random>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

// RPDF dither: uniform noise in [-amplitude, +amplitude]
template <DspField T>
class RPDFDither {
public:
	explicit RPDFDither(T amplitude, unsigned seed = 0)
		: amplitude_(amplitude)
		, gen_(seed == 0 ? std::random_device{}() : seed)
		, dist_(-1.0, 1.0) {}

	T operator()() {
		return amplitude_ * static_cast<T>(dist_(gen_));
	}

	// Apply dither to a vector in-place
	void apply(mtl::vec::dense_vector<T>& signal) {
		for (std::size_t i = 0; i < signal.size(); ++i) {
			signal[i] = signal[i] + (*this)();
		}
	}

private:
	T amplitude_;
	std::mt19937 gen_;
	std::uniform_real_distribution<double> dist_;
};

// TPDF dither: triangular noise (sum of two uniform)
// TPDF eliminates the noise modulation artifact that RPDF leaves.
template <DspField T>
class TPDFDither {
public:
	explicit TPDFDither(T amplitude, unsigned seed = 0)
		: amplitude_(amplitude)
		, gen_(seed == 0 ? std::random_device{}() : seed)
		, dist_(-1.0, 1.0) {}

	T operator()() {
		// Sum of two uniform gives triangular distribution
		double u1 = dist_(gen_);
		double u2 = dist_(gen_);
		return amplitude_ * static_cast<T>((u1 + u2) * 0.5);
	}

	void apply(mtl::vec::dense_vector<T>& signal) {
		for (std::size_t i = 0; i < signal.size(); ++i) {
			signal[i] = signal[i] + (*this)();
		}
	}

private:
	T amplitude_;
	std::mt19937 gen_;
	std::uniform_real_distribution<double> dist_;
};

} // namespace sw::dsp
