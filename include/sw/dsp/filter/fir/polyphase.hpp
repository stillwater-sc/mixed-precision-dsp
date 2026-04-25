#pragma once
// polyphase.hpp: polyphase decomposition for efficient multirate FIR filtering
//
// Polyphase decomposition splits an N-tap FIR filter into L (or M) sub-filters
// of length ceil(N/L) whose outputs, appropriately interleaved or summed, are
// equivalent to upsample-then-filter (interpolation) or filter-then-downsample
// (decimation) with the original filter. The savings is the rate-change factor
// because sub-filters run at the slower rate instead of the faster rate.
//
// Interpolation by L:
//   y[nL + k] = sum_p h[pL + k] * x[n - p]  for k = 0..L-1
//   Each input sample x[n] produces L output samples; each sub-filter advances
//   once per input sample. Compute: L * (N/L) = N mults per input vs. L*N for
//   the naive upsample-then-filter approach.
//
// Decimation by M:
//   y[n] = sum_q h_q convolved with stream_q  at time n,
//   where stream_q[m] = x[mM - q] and h_q[p] = h[pM + q].
//   Each output sample consumes M input samples. Each sub-filter advances once
//   per output sample. Compute: M * (N/M) = N mults per output vs. N*M for the
//   naive filter-then-downsample approach.
//
// Three-scalar parameterization:
//   CoeffScalar  - tap coefficients (designed in high precision)
//   StateScalar  - delay-line accumulation precision
//   SampleScalar - input/output samples
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/filter/fir/fir_filter.hpp>

namespace sw::dsp {

namespace detail {

// Decompose taps h[0..N-1] into `factor` sub-tap arrays:
//   sub_taps[q][p] = h[p * factor + q]    for q in [0, factor),
//                                         p in [0, ceil(N/factor)).
// Each sub-array has identical length (zero-padded at the tail) so the
// sub-filters run in lock step. Zero padding does not change the convolution
// result; it only costs a few multiplies on the last sample.
template <typename T>
std::vector<mtl::vec::dense_vector<T>>
decompose_polyphase(const mtl::vec::dense_vector<T>& taps, std::size_t factor) {
	if (factor == 0)
		throw std::invalid_argument("polyphase: factor must be > 0");
	std::size_t N = taps.size();
	if (N == 0)
		throw std::invalid_argument("polyphase: taps must not be empty");

	std::size_t sub_len = (N + factor - 1) / factor;
	std::vector<mtl::vec::dense_vector<T>> sub_taps;
	sub_taps.reserve(factor);
	for (std::size_t q = 0; q < factor; ++q) {
		mtl::vec::dense_vector<T> sub(sub_len, T{});
		for (std::size_t p = 0; p < sub_len; ++p) {
			std::size_t idx = p * factor + q;
			if (idx < N) sub[p] = taps[idx];
		}
		sub_taps.push_back(std::move(sub));
	}
	return sub_taps;
}

} // namespace detail

// Public design helper: decompose an FIR prototype into the M sub-filter
// branches used by polyphase decimation/interpolation.
//
// Arguments:
//   taps   - prototype FIR impulse response, length N
//   factor - rate-change factor M (decimation or interpolation), > 0
//
// Returns:
//   M sub-filters of length ceil(N / M). sub[q][p] = taps[p*M + q] with
//   zero-padding at the tail so all branches have identical length.
//
// Useful when callers want to inspect the polyphase matrix (e.g., for
// pre-quantization analysis or hardware coefficient placement) without
// instantiating a runtime PolyphaseDecimator/Interpolator.
template <typename T>
std::vector<mtl::vec::dense_vector<T>>
polyphase_decompose(const mtl::vec::dense_vector<T>& taps, std::size_t factor) {
	return detail::decompose_polyphase(taps, factor);
}

// -----------------------------------------------------------------------------
// PolyphaseInterpolator: integer-factor upsampling with embedded FIR smoothing.
// -----------------------------------------------------------------------------
template <DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class PolyphaseInterpolator {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	using sub_filter_t  = FIRFilter<CoeffScalar, StateScalar, SampleScalar>;

	// taps: full-rate lowpass impulse response designed for a sample rate of
	//       factor * input_rate. Length need not be a multiple of factor
	//       (the decomposition zero-pads the tail).
	// factor: integer upsampling ratio L > 0.
	PolyphaseInterpolator(const mtl::vec::dense_vector<CoeffScalar>& taps,
	                      std::size_t factor)
		: factor_(factor) {
		auto sub_taps = detail::decompose_polyphase(taps, factor);
		sub_filters_.reserve(factor);
		for (auto& t : sub_taps) sub_filters_.emplace_back(t);
	}

	// Process one input sample; return L output samples (in their natural
	// output-time order: y[nL], y[nL+1], ..., y[nL+L-1]).
	mtl::vec::dense_vector<SampleScalar> process(SampleScalar in) {
		mtl::vec::dense_vector<SampleScalar> out(factor_);
		for (std::size_t k = 0; k < factor_; ++k) {
			out[k] = sub_filters_[k].process(in);
		}
		return out;
	}

	// Process a block of input samples; return factor * input.size() outputs.
	mtl::vec::dense_vector<SampleScalar>
	process_block(std::span<const SampleScalar> input) {
		std::size_t in_len = input.size();
		mtl::vec::dense_vector<SampleScalar> out(in_len * factor_);
		std::size_t idx = 0;
		for (std::size_t n = 0; n < in_len; ++n) {
			for (std::size_t k = 0; k < factor_; ++k) {
				out[idx++] = sub_filters_[k].process(input[n]);
			}
		}
		return out;
	}

	void reset() {
		for (auto& f : sub_filters_) f.reset();
	}

	std::size_t factor() const { return factor_; }
	std::size_t num_sub_filters() const { return sub_filters_.size(); }
	std::size_t sub_filter_length() const {
		return sub_filters_.empty() ? 0 : sub_filters_[0].num_taps();
	}

private:
	std::size_t factor_;
	std::vector<sub_filter_t> sub_filters_;
};

// -----------------------------------------------------------------------------
// PolyphaseDecimator: integer-factor downsampling with embedded FIR smoothing.
// -----------------------------------------------------------------------------
template <DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class PolyphaseDecimator {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	using sub_filter_t  = FIRFilter<CoeffScalar, StateScalar, SampleScalar>;

	// taps: full-rate anti-aliasing lowpass designed at the input rate.
	// factor: integer decimation ratio M > 0.
	//
	// Streaming contract: output is emitted whenever the input index is a
	// multiple of M (counting from the constructor). So the first emitted
	// output corresponds to input index 0, the second to index M, and so on.
	PolyphaseDecimator(const mtl::vec::dense_vector<CoeffScalar>& taps,
	                   std::size_t factor)
		: factor_(factor),
		  latest_(factor, SampleScalar{}),
		  input_phase_(0) {
		auto sub_taps = detail::decompose_polyphase(taps, factor);
		sub_filters_.reserve(factor);
		for (auto& t : sub_taps) sub_filters_.emplace_back(t);
	}

	// Process one input sample. Returns {true, y} on the emit cycle, else
	// {false, 0}. The emit cycle occurs when the running input index is a
	// multiple of M (phase 0).
	std::pair<bool, SampleScalar> process(SampleScalar in) {
		// Stream dispatch: input at phase r feeds sub-filter (M - r) mod M.
		std::size_t r = input_phase_;
		std::size_t q = (r == 0) ? 0 : factor_ - r;
		latest_[q] = sub_filters_[q].process(in);

		input_phase_ = (input_phase_ + 1) % factor_;

		if (r == 0) {
			// All M sub-filter outputs for this output step are now current.
			StateScalar acc{};
			for (std::size_t i = 0; i < factor_; ++i) {
				acc = acc + static_cast<StateScalar>(latest_[i]);
			}
			return {true, static_cast<SampleScalar>(acc)};
		}
		return {false, SampleScalar{}};
	}

	// Process a block of input samples; return exactly
	// count(k in [phase_at_entry, phase_at_entry + input.size()) : k mod M == 0)
	// output samples.
	mtl::vec::dense_vector<SampleScalar>
	process_block(std::span<const SampleScalar> input) {
		// Count upcoming emit events so we can size the output vector.
		std::size_t emits = 0;
		{
			std::size_t phase = input_phase_;
			for (std::size_t i = 0; i < input.size(); ++i) {
				if (phase == 0) ++emits;
				phase = (phase + 1) % factor_;
			}
		}
		mtl::vec::dense_vector<SampleScalar> out(emits);
		std::size_t idx = 0;
		for (std::size_t i = 0; i < input.size(); ++i) {
			auto [has_out, y] = process(input[i]);
			if (has_out) out[idx++] = y;
		}
		return out;
	}

	void reset() {
		for (auto& f : sub_filters_) f.reset();
		for (std::size_t i = 0; i < latest_.size(); ++i) latest_[i] = SampleScalar{};
		input_phase_ = 0;
	}

	std::size_t factor() const { return factor_; }
	std::size_t num_sub_filters() const { return sub_filters_.size(); }
	std::size_t sub_filter_length() const {
		return sub_filters_.empty() ? 0 : sub_filters_[0].num_taps();
	}

private:
	std::size_t factor_;
	std::vector<sub_filter_t> sub_filters_;
	mtl::vec::dense_vector<SampleScalar> latest_;
	std::size_t input_phase_;
};

} // namespace sw::dsp
