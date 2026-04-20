#pragma once
// halfband.hpp: Half-band FIR filter for efficient 2x decimation/interpolation
//
// Half-band filters satisfy H(w) + H(pi-w) = 1, which means nearly half the
// coefficients are zero.  Combined with 2x decimation this yields ~4x
// computational savings over naive filter-and-decimate.
//
// Three-scalar parameterization:
//   CoeffScalar  - tap coefficients (design precision)
//   StateScalar  - delay line and accumulation (processing precision)
//   SampleScalar - input/output samples (streaming precision)
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/filter/fir/remez.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/math/denormal.hpp>

namespace sw::dsp {

// Design an equiripple half-band lowpass filter via Remez exchange.
//
// Parameters:
//   num_taps         - filter length, must be of the form 4K+3 (e.g., 7, 11, 15, 19, ...)
//   transition_width - transition bandwidth in normalized frequency [0, 0.5];
//                      passband edge = 0.25 - tw/2, stopband edge = 0.25 + tw/2
//
// Returns: dense_vector of half-band filter taps with enforced structure:
//   h[center] = 0.5, h[center +/- 2k] = 0 for k >= 1
template <DspField T>
mtl::vec::dense_vector<T> design_halfband(
    std::size_t num_taps,
    T transition_width) {

	if (num_taps < 3)
		throw std::invalid_argument("design_halfband: num_taps must be >= 3");
	if ((num_taps & 1) == 0)
		throw std::invalid_argument("design_halfband: num_taps must be odd");
	if ((num_taps % 4) != 3)
		throw std::invalid_argument(
			"design_halfband: num_taps must be of the form 4K+3 "
			"(e.g., 3, 7, 11, 15, 19, ...)");

	double tw = static_cast<double>(transition_width);
	if (tw <= 0.0 || tw >= 0.5)
		throw std::invalid_argument(
			"design_halfband: transition_width must be in (0, 0.5)");

	double pass_edge = 0.25 - tw / 2.0;
	double stop_edge = 0.25 + tw / 2.0;

	if (pass_edge <= 0.0 || stop_edge >= 0.5)
		throw std::invalid_argument(
			"design_halfband: transition_width too large");

	std::vector<T> bands   = {T{0}, static_cast<T>(pass_edge),
	                          static_cast<T>(stop_edge), T(0.5)};
	std::vector<T> desired = {T{1}, T{1}, T{0}, T{0}};
	std::vector<T> weights = {T{1}, T{1}};

	auto taps = remez<T>(num_taps, bands, desired, weights);

	// Enforce exact half-band constraints
	std::size_t center = (num_taps - 1) / 2;
	taps[center] = static_cast<T>(0.5);
	for (std::size_t k = 2; k <= center; k += 2) {
		taps[center - k] = T{0};
		taps[center + k] = T{0};
	}

	// Enforce perfect symmetry on the non-zero taps
	for (std::size_t k = 1; k <= center; k += 2) {
		T avg = static_cast<T>(
			(static_cast<double>(taps[center - k]) +
			 static_cast<double>(taps[center + k])) / 2.0);
		taps[center - k] = avg;
		taps[center + k] = avg;
	}

	// Normalize: center contributes 0.5, odd-offset taps must sum to 0.5
	double sum_odd = 0.0;
	for (std::size_t k = 1; k <= center; k += 2) {
		sum_odd += 2.0 * static_cast<double>(taps[center + k]);
	}
	if (std::abs(sum_odd) > 1e-15) {
		double scale = 0.5 / sum_odd;
		for (std::size_t k = 1; k <= center; k += 2) {
			T val = static_cast<T>(static_cast<double>(taps[center + k]) * scale);
			taps[center - k] = val;
			taps[center + k] = val;
		}
	}

	return taps;
}

// Half-band FIR filter with optimized computation that skips zero-valued taps.
//
// For 2x decimation, use process_decimate() or process_block_decimate() which
// compute only the output samples that survive downsampling, yielding ~4x
// savings over naive filter-then-decimate.
template <DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class HalfBandFilter {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;

	explicit HalfBandFilter(const mtl::vec::dense_vector<CoeffScalar>& taps)
		: taps_(taps),
		  delay_(taps.size(), StateScalar{}),
		  write_pos_(0),
		  center_((taps.size() - 1) / 2),
		  decimate_phase_(0) {

		if (taps.size() < 3)
			throw std::invalid_argument("HalfBandFilter: need at least 3 taps");
		if ((taps.size() & 1) == 0)
			throw std::invalid_argument("HalfBandFilter: num_taps must be odd");

		center_coeff_ = taps[center_];

		for (std::size_t k = 1; k <= center_; k += 2) {
			sym_offsets_.push_back(k);
			sym_coeffs_.push_back(taps[center_ + k]);
		}
	}

	// Full-rate: process one sample, produce one output
	SampleScalar process(SampleScalar in) {
		std::size_t N = taps_.size();
		delay_[write_pos_] = static_cast<StateScalar>(in);

		std::size_t ci = (write_pos_ + N - center_) % N;
		StateScalar acc = static_cast<StateScalar>(center_coeff_) * delay_[ci]
		                + denormal_.ac();

		for (std::size_t i = 0; i < sym_offsets_.size(); ++i) {
			std::size_t k = sym_offsets_[i];
			std::size_t left  = (ci + N - k) % N;
			std::size_t right = (ci + k) % N;
			acc = acc + static_cast<StateScalar>(sym_coeffs_[i]) *
			            (delay_[left] + delay_[right])
			    + denormal_.ac();
		}

		write_pos_ = (write_pos_ + 1) % N;
		return static_cast<SampleScalar>(acc);
	}

	// Block processing: full-rate
	void process_block(std::span<const SampleScalar> input,
	                   std::span<SampleScalar> output) {
		std::size_t n = std::min(input.size(), output.size());
		for (std::size_t i = 0; i < n; ++i) {
			output[i] = process(input[i]);
		}
	}

	// Dense-vector overload
	mtl::vec::dense_vector<SampleScalar> process_block(
			const mtl::vec::dense_vector<SampleScalar>& input) {
		mtl::vec::dense_vector<SampleScalar> output(input.size());
		for (std::size_t i = 0; i < input.size(); ++i) {
			output[i] = process(input[i]);
		}
		return output;
	}

	// Integrated 2x decimation: push one sample, output emitted every other call.
	// Returns {true, y} when output is ready, {false, 0} otherwise.
	std::pair<bool, SampleScalar> process_decimate(SampleScalar in) {
		std::size_t N = taps_.size();
		delay_[write_pos_] = static_cast<StateScalar>(in);

		bool emit = (decimate_phase_ == 1);

		if (emit) {
			std::size_t ci = (write_pos_ + N - center_) % N;
			StateScalar acc =
				static_cast<StateScalar>(center_coeff_) * delay_[ci]
				+ denormal_.ac();

			for (std::size_t i = 0; i < sym_offsets_.size(); ++i) {
				std::size_t k = sym_offsets_[i];
				std::size_t left  = (ci + N - k) % N;
				std::size_t right = (ci + k) % N;
				acc = acc + static_cast<StateScalar>(sym_coeffs_[i]) *
				            (delay_[left] + delay_[right])
				    + denormal_.ac();
			}

			write_pos_ = (write_pos_ + 1) % N;
			decimate_phase_ = 0;
			return {true, static_cast<SampleScalar>(acc)};
		}

		write_pos_ = (write_pos_ + 1) % N;
		decimate_phase_ = 1;
		return {false, SampleScalar{}};
	}

	// Block decimation: process input, append decimated outputs to vector
	void process_block_decimate(std::span<const SampleScalar> input,
	                            std::vector<SampleScalar>& output) {
		for (auto s : input) {
			auto [ready, y] = process_decimate(s);
			if (ready) output.push_back(y);
		}
	}

	// Dense-vector overload for decimation
	mtl::vec::dense_vector<SampleScalar> process_block_decimate(
			const mtl::vec::dense_vector<SampleScalar>& input) {
		std::vector<SampleScalar> tmp;
		tmp.reserve(input.size() / 2 + 1);
		for (std::size_t i = 0; i < input.size(); ++i) {
			auto [ready, y] = process_decimate(input[i]);
			if (ready) tmp.push_back(y);
		}
		mtl::vec::dense_vector<SampleScalar> result(tmp.size());
		for (std::size_t i = 0; i < tmp.size(); ++i) result[i] = tmp[i];
		return result;
	}

	void reset() {
		for (std::size_t i = 0; i < delay_.size(); ++i)
			delay_[i] = StateScalar{};
		write_pos_ = 0;
		decimate_phase_ = 0;
	}

	std::size_t order()           const { return taps_.size() > 0 ? taps_.size() - 1 : 0; }
	std::size_t num_taps()        const { return taps_.size(); }
	std::size_t num_nonzero_taps() const { return 1 + 2 * sym_coeffs_.size(); }
	const mtl::vec::dense_vector<CoeffScalar>& taps() const { return taps_; }

private:
	mtl::vec::dense_vector<CoeffScalar> taps_;
	mtl::vec::dense_vector<StateScalar> delay_;
	std::size_t write_pos_;
	std::size_t center_;
	int decimate_phase_;

	CoeffScalar center_coeff_;
	std::vector<std::size_t> sym_offsets_;
	std::vector<CoeffScalar> sym_coeffs_;
	DenormalPrevention<StateScalar> denormal_;
};

} // namespace sw::dsp
