#pragma once
// sqnr.hpp: signal-to-quantization-noise ratio analysis
//
// Compares a reference signal against a quantized version to
// measure the signal-to-quantization-noise ratio in dB.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

// Compute SQNR in dB between a reference signal and a quantized version.
// SQNR = 10 * log10(signal_power / noise_power)
// where noise = reference - quantized.
//
// Both vectors must be the same length.
// Returns SQNR in dB. Higher = better.
template <DspField RefT, DspField QuantT>
double sqnr_db(const mtl::vec::dense_vector<RefT>& reference,
               const mtl::vec::dense_vector<QuantT>& quantized) {
	if (reference.size() != quantized.size())
		throw std::invalid_argument("sqnr_db: vectors must have equal length");
	if (reference.size() == 0)
		throw std::invalid_argument("sqnr_db: vectors must not be empty");

	double signal_power = 0.0;
	double noise_power = 0.0;

	for (std::size_t i = 0; i < reference.size(); ++i) {
		double ref = static_cast<double>(reference[i]);
		double quant = static_cast<double>(quantized[i]);
		double noise = ref - quant;
		signal_power += ref * ref;
		noise_power += noise * noise;
	}

	if (noise_power == 0.0) return std::numeric_limits<double>::infinity();
	if (signal_power == 0.0) return 0.0;

	return 10.0 * std::log10(signal_power / noise_power);
}

// Convenience: quantize a reference signal to type QuantT and compute SQNR.
template <DspField RefT, DspField QuantT>
double measure_sqnr_db(const mtl::vec::dense_vector<RefT>& reference) {
	mtl::vec::dense_vector<QuantT> quantized(reference.size());
	for (std::size_t i = 0; i < reference.size(); ++i) {
		quantized[i] = static_cast<QuantT>(reference[i]);
	}
	return sqnr_db(reference, quantized);
}

// Compute max absolute error between reference and quantized signals.
template <DspField RefT, DspField QuantT>
double max_absolute_error(const mtl::vec::dense_vector<RefT>& reference,
                           const mtl::vec::dense_vector<QuantT>& quantized) {
	if (reference.size() != quantized.size())
		throw std::invalid_argument("max_absolute_error: vectors must have equal length");

	double max_err = 0.0;
	for (std::size_t i = 0; i < reference.size(); ++i) {
		double err = std::abs(static_cast<double>(reference[i]) - static_cast<double>(quantized[i]));
		if (err > max_err) max_err = err;
	}
	return max_err;
}

// Compute max relative error between reference and quantized signals.
template <DspField RefT, DspField QuantT>
double max_relative_error(const mtl::vec::dense_vector<RefT>& reference,
                           const mtl::vec::dense_vector<QuantT>& quantized) {
	if (reference.size() != quantized.size())
		throw std::invalid_argument("max_relative_error: vectors must have equal length");

	double max_ref = 0.0;
	double max_err = 0.0;
	for (std::size_t i = 0; i < reference.size(); ++i) {
		double ref = std::abs(static_cast<double>(reference[i]));
		double err = std::abs(static_cast<double>(reference[i]) - static_cast<double>(quantized[i]));
		if (ref > max_ref) max_ref = ref;
		if (err > max_err) max_err = err;
	}
	return (max_ref > 0.0) ? max_err / max_ref : 0.0;
}

} // namespace sw::dsp
