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

	T tw = transition_width;
	T half = T(0.5);
	T quarter = T(0.25);
	T two = T{1} + T{1};

	if (!(tw > T{0}) || !(tw < half))
		throw std::invalid_argument(
			"design_halfband: transition_width must be in (0, 0.5)");

	T pass_edge = quarter - tw / two;
	T stop_edge = quarter + tw / two;

	if (!(pass_edge > T{0}) || !(stop_edge < half))
		throw std::invalid_argument(
			"design_halfband: transition_width too large");

	std::vector<T> bands   = {T{0}, pass_edge, stop_edge, half};
	std::vector<T> desired = {T{1}, T{1}, T{0}, T{0}};
	std::vector<T> weights = {T{1}, T{1}};

	auto taps = remez<T>(num_taps, bands, desired, weights);

	// Enforce exact half-band constraints
	std::size_t center = (num_taps - 1) / 2;
	taps[center] = half;
	for (std::size_t k = 2; k <= center; k += 2) {
		taps[center - k] = T{0};
		taps[center + k] = T{0};
	}

	// Enforce perfect symmetry on the non-zero taps
	for (std::size_t k = 1; k <= center; k += 2) {
		T avg = (taps[center - k] + taps[center + k]) / two;
		taps[center - k] = avg;
		taps[center + k] = avg;
	}

	// Normalize: center contributes 0.5, odd-offset taps must sum to 0.5
	T sum_odd{};
	for (std::size_t k = 1; k <= center; k += 2) {
		sum_odd = sum_odd + two * taps[center + k];
	}
	if (!(sum_odd == T{0})) {
		T scale = half / sum_odd;
		for (std::size_t k = 1; k <= center; k += 2) {
			T val = taps[center + k] * scale;
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

		// Validate half-band structure: even offsets from center must be zero,
		// and taps must be symmetric
		CoeffScalar zero{};
		for (std::size_t k = 2; k <= center_; k += 2) {
			if (!(taps[center_ - k] == zero) || !(taps[center_ + k] == zero))
				throw std::invalid_argument(
					"HalfBandFilter: even-offset taps must be zero "
					"(offset " + std::to_string(k) + " is non-zero)");
		}
		for (std::size_t k = 1; k <= center_; ++k) {
			if (!(taps[center_ - k] == taps[center_ + k]))
				throw std::invalid_argument(
					"HalfBandFilter: taps must be symmetric "
					"(mismatch at offset " + std::to_string(k) + ")");
		}

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

	// Block processing: full-rate (output span must be >= input span)
	void process_block(std::span<const SampleScalar> input,
	                   std::span<SampleScalar> output) {
		if (output.size() < input.size())
			throw std::invalid_argument(
				"HalfBandFilter::process_block: output span too small");
		for (std::size_t i = 0; i < input.size(); ++i) {
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

	// Block decimation from span, returns dense_vector of decimated outputs
	mtl::vec::dense_vector<SampleScalar> process_block_decimate(
			std::span<const SampleScalar> input) {
		// Pre-compute output count
		std::size_t count = 0;
		int phase = decimate_phase_;
		for (std::size_t i = 0; i < input.size(); ++i) {
			if (phase == 1) ++count;
			phase = (phase == 0) ? 1 : 0;
		}
		mtl::vec::dense_vector<SampleScalar> output(count);
		std::size_t idx = 0;
		for (std::size_t i = 0; i < input.size(); ++i) {
			auto [ready, y] = process_decimate(input[i]);
			if (ready) output[idx++] = y;
		}
		return output;
	}

	// Dense-vector overload for decimation
	mtl::vec::dense_vector<SampleScalar> process_block_decimate(
			const mtl::vec::dense_vector<SampleScalar>& input) {
		std::size_t count = 0;
		int phase = decimate_phase_;
		for (std::size_t i = 0; i < input.size(); ++i) {
			if (phase == 1) ++count;
			phase = (phase == 0) ? 1 : 0;
		}
		mtl::vec::dense_vector<SampleScalar> output(count);
		std::size_t idx = 0;
		for (std::size_t i = 0; i < input.size(); ++i) {
			auto [ready, y] = process_decimate(input[i]);
			if (ready) output[idx++] = y;
		}
		return output;
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
