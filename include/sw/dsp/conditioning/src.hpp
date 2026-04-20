#pragma once
// src.hpp: rational sample-rate conversion (L/M resampling)
//
// Combines polyphase interpolation by L and decimation by M with an
// auto-designed Kaiser-windowed sinc anti-alias filter. The polyphase
// structure avoids computing samples that would be discarded.
//
// Algorithm (time-register approach):
//   For each input sample pushed into the delay line:
//     while (time_register < L):
//       output = sub_filter[time_register] · delay_line
//       emit output
//       time_register += M
//     time_register -= L   (consumed one input)
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/filter/fir/fir_design.hpp>
#include <sw/dsp/windows/kaiser.hpp>

namespace sw::dsp {

template <DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class RationalResampler {
public:
	// L: interpolation factor, M: decimation factor
	// filter_half_length: half-length in periods of the slower rate
	// beta: Kaiser window shape parameter
	RationalResampler(std::size_t L, std::size_t M,
	                  std::size_t filter_half_length = 10,
	                  CoeffScalar beta = CoeffScalar{5})
		: L_(L), M_(M), time_reg_(0) {
		if (L == 0 || M == 0)
			throw std::invalid_argument("RationalResampler: L and M must be > 0");

		std::size_t g = std::gcd(L, M);
		L_ = L / g;
		M_ = M / g;

		std::size_t max_factor = std::max(L_, M_);
		std::size_t num_taps = 2 * filter_half_length * max_factor + 1;

		auto win = kaiser_window<CoeffScalar>(num_taps, static_cast<double>(beta));

		double cutoff = 0.5 / static_cast<double>(max_factor);
		auto taps = design_fir_lowpass<CoeffScalar>(num_taps, CoeffScalar(cutoff), win);

		// Scale by L for unity passband gain after interpolation
		for (std::size_t i = 0; i < num_taps; ++i) {
			taps[i] = taps[i] * CoeffScalar(static_cast<double>(L_));
		}

		// Decompose into L sub-filters
		std::size_t sub_len = (num_taps + L_ - 1) / L_;
		sub_taps_.resize(L_);
		for (std::size_t q = 0; q < L_; ++q) {
			sub_taps_[q].resize(sub_len, CoeffScalar{});
			for (std::size_t p = 0; p < sub_len; ++p) {
				std::size_t idx = p * L_ + q;
				if (idx < num_taps) sub_taps_[q][p] = taps[idx];
			}
		}

		delay_.resize(sub_len, SampleScalar{});
		write_pos_ = 0;
	}

	mtl::vec::dense_vector<SampleScalar> process(
	        const mtl::vec::dense_vector<SampleScalar>& input) {
		std::size_t out_cap = static_cast<std::size_t>(
			std::ceil(static_cast<double>(input.size()) *
			          static_cast<double>(L_) / static_cast<double>(M_))) + L_;
		std::vector<SampleScalar> out_buf;
		out_buf.reserve(out_cap);

		for (std::size_t i = 0; i < input.size(); ++i) {
			delay_[write_pos_] = input[i];
			write_pos_ = (write_pos_ + 1) % delay_.size();

			while (time_reg_ < L_) {
				out_buf.push_back(compute_sub_filter(time_reg_));
				time_reg_ += M_;
			}
			time_reg_ -= L_;
		}

		mtl::vec::dense_vector<SampleScalar> result(out_buf.size());
		for (std::size_t i = 0; i < out_buf.size(); ++i) {
			result[i] = out_buf[i];
		}
		return result;
	}

	void reset() {
		for (auto& d : delay_) d = SampleScalar{};
		write_pos_ = 0;
		time_reg_ = 0;
	}

	double ratio() const {
		return static_cast<double>(L_) / static_cast<double>(M_);
	}

	std::size_t interp_factor() const { return L_; }
	std::size_t decim_factor() const { return M_; }

private:
	SampleScalar compute_sub_filter(std::size_t phase) {
		const auto& taps = sub_taps_[phase];
		StateScalar acc{};
		std::size_t pos = (write_pos_ == 0) ? delay_.size() - 1 : write_pos_ - 1;
		for (std::size_t p = 0; p < taps.size(); ++p) {
			acc = acc + static_cast<StateScalar>(taps[p])
			          * static_cast<StateScalar>(delay_[pos]);
			pos = (pos == 0) ? delay_.size() - 1 : pos - 1;
		}
		return static_cast<SampleScalar>(acc);
	}

	std::size_t L_;
	std::size_t M_;
	std::size_t time_reg_;
	std::vector<std::vector<CoeffScalar>> sub_taps_;
	std::vector<SampleScalar> delay_;
	std::size_t write_pos_;
};

} // namespace sw::dsp
