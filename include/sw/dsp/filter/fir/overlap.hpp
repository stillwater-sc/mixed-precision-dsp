#pragma once
// overlap.hpp: fast convolution via FFT using overlap-add and overlap-save
//
// For a signal x of length N and filter h of length M, direct convolution
// costs O(N*M) multiplies. When M is large (say > ~30) the block-FFT methods
// here win: cost O((N/L) * K log K) with K = next_pow2(L + M - 1).
//
// Overlap-add: each input block is FFT-convolved; the resulting (L + M - 1)
// sample output is split into an L-sample "body" that emits immediately and
// an (M - 1) sample "tail" that is added to the next block's head.
//
// Overlap-save: each input block is (M - 1) samples of prior history plus L
// new samples. The circular convolution wraps the first M-1 output samples
// (they alias the linear tail); these are discarded, and the remaining L
// samples are emitted directly.
//
// Both produce identical output (to FFT precision) and are mathematically
// equivalent to direct linear convolution x * h.
//
// Three-scalar parameterization:
//   CoeffScalar  - filter taps
//   StateScalar  - FFT arithmetic / accumulation
//   SampleScalar - input/output samples
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <span>
#include <stdexcept>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/spectral/fft.hpp>

namespace sw::dsp {

namespace detail {

inline std::size_t next_power_of_2(std::size_t n) {
	std::size_t p = 1;
	while (p < n) p <<= 1;
	return p;
}

// Compute the FFT of a real tap array, zero-padded to size fft_size.
template <typename StateScalar, typename CoeffScalar>
mtl::vec::dense_vector<complex_for_t<StateScalar>>
fft_of_taps(const mtl::vec::dense_vector<CoeffScalar>& taps, std::size_t fft_size) {
	using complex_t = complex_for_t<StateScalar>;
	mtl::vec::dense_vector<complex_t> H(fft_size, complex_t{});
	for (std::size_t i = 0; i < taps.size(); ++i) {
		H[i] = complex_t(static_cast<StateScalar>(taps[i]));
	}
	sw::dsp::spectral::fft_forward<StateScalar>(H);
	return H;
}

} // namespace detail

// -----------------------------------------------------------------------------
// OverlapAddConvolver: block-streaming fast convolution.
// -----------------------------------------------------------------------------
template <DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class OverlapAddConvolver {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	using complex_t     = complex_for_t<StateScalar>;

	// taps: filter impulse response, length M >= 1.
	// block_size: the number of samples per process_block() call, L >= 1.
	//             FFT size is internally set to next_pow2(L + M - 1).
	OverlapAddConvolver(const mtl::vec::dense_vector<CoeffScalar>& taps,
	                    std::size_t block_size)
		: block_size_(block_size),
		  filter_length_(taps.size()),
		  fft_size_(detail::next_power_of_2(block_size + taps.size() - 1)),
		  tail_(taps.size() > 0 ? taps.size() - 1 : 0, SampleScalar{}) {
		if (taps.size() == 0)
			throw std::invalid_argument("OverlapAddConvolver: taps must not be empty");
		if (block_size == 0)
			throw std::invalid_argument("OverlapAddConvolver: block_size must be > 0");
		H_ = detail::fft_of_taps<StateScalar, CoeffScalar>(taps, fft_size_);
	}

	// Process exactly block_size samples of input; return exactly block_size
	// samples of output. The first M-1 samples of the first call are the
	// start-up transient (correctly handled: initial tail is zero).
	mtl::vec::dense_vector<SampleScalar>
	process_block(std::span<const SampleScalar> input) {
		if (input.size() != block_size_)
			throw std::invalid_argument(
				"OverlapAddConvolver::process_block: input size must equal block_size");

		mtl::vec::dense_vector<complex_t> buf(fft_size_, complex_t{});
		for (std::size_t i = 0; i < block_size_; ++i) {
			buf[i] = complex_t(static_cast<StateScalar>(input[i]));
		}
		sw::dsp::spectral::fft_forward<StateScalar>(buf);
		for (std::size_t i = 0; i < fft_size_; ++i) {
			buf[i] = buf[i] * H_[i];
		}
		sw::dsp::spectral::fft_inverse<StateScalar>(buf);

		mtl::vec::dense_vector<SampleScalar> out(block_size_);
		std::size_t tail_len = tail_.size();  // M - 1
		// Add old tail to the head of this block's output.
		for (std::size_t i = 0; i < block_size_; ++i) {
			StateScalar v = buf[i].real();
			if (i < tail_len) {
				v = v + static_cast<StateScalar>(tail_[i]);
			}
			out[i] = static_cast<SampleScalar>(v);
		}
		// Build new tail: shift unconsumed old tail forward, then add this
		// block's convolution tail (y_block[block_size..block_size+M-2]).
		std::size_t carry = (tail_len > block_size_) ? tail_len - block_size_ : 0;
		for (std::size_t i = 0; i < carry; ++i) {
			tail_[i] = tail_[block_size_ + i];
		}
		for (std::size_t i = carry; i < tail_len; ++i) {
			tail_[i] = SampleScalar{};
		}
		for (std::size_t i = 0; i < tail_len; ++i) {
			tail_[i] = static_cast<SampleScalar>(
				static_cast<StateScalar>(tail_[i]) + buf[block_size_ + i].real());
		}
		return out;
	}

	// Emit the trailing M-1 samples that haven't been added to a subsequent
	// block yet. Call exactly once after the final process_block() to recover
	// the complete linear convolution.
	mtl::vec::dense_vector<SampleScalar> flush() {
		mtl::vec::dense_vector<SampleScalar> out(tail_.size());
		for (std::size_t i = 0; i < tail_.size(); ++i) {
			out[i] = tail_[i];
			tail_[i] = SampleScalar{};
		}
		return out;
	}

	void reset() {
		for (std::size_t i = 0; i < tail_.size(); ++i) tail_[i] = SampleScalar{};
	}

	std::size_t block_size() const { return block_size_; }
	std::size_t fft_size() const { return fft_size_; }
	std::size_t filter_length() const { return filter_length_; }

private:
	std::size_t block_size_;
	std::size_t filter_length_;
	std::size_t fft_size_;
	mtl::vec::dense_vector<complex_t> H_;
	mtl::vec::dense_vector<SampleScalar> tail_;
};

// -----------------------------------------------------------------------------
// OverlapSaveConvolver: block-streaming fast convolution, save variant.
// -----------------------------------------------------------------------------
template <DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class OverlapSaveConvolver {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	using complex_t     = complex_for_t<StateScalar>;

	OverlapSaveConvolver(const mtl::vec::dense_vector<CoeffScalar>& taps,
	                     std::size_t block_size)
		: block_size_(block_size),
		  filter_length_(taps.size()),
		  fft_size_(detail::next_power_of_2(block_size + taps.size() - 1)),
		  history_(taps.size() > 0 ? taps.size() - 1 : 0, SampleScalar{}) {
		if (taps.size() == 0)
			throw std::invalid_argument("OverlapSaveConvolver: taps must not be empty");
		if (block_size == 0)
			throw std::invalid_argument("OverlapSaveConvolver: block_size must be > 0");
		H_ = detail::fft_of_taps<StateScalar, CoeffScalar>(taps, fft_size_);
	}

	// Process exactly block_size input samples; return exactly block_size
	// output samples. The first M-1 samples of the first call are the
	// start-up transient (correctly handled via zero-initial history).
	mtl::vec::dense_vector<SampleScalar>
	process_block(std::span<const SampleScalar> input) {
		if (input.size() != block_size_)
			throw std::invalid_argument(
				"OverlapSaveConvolver::process_block: input size must equal block_size");

		std::size_t hist_len = history_.size();
		mtl::vec::dense_vector<complex_t> buf(fft_size_, complex_t{});
		// First hist_len samples: previous history.
		for (std::size_t i = 0; i < hist_len; ++i) {
			buf[i] = complex_t(static_cast<StateScalar>(history_[i]));
		}
		// Next block_size samples: new input.
		for (std::size_t i = 0; i < block_size_; ++i) {
			buf[hist_len + i] = complex_t(static_cast<StateScalar>(input[i]));
		}
		// Remaining positions are already zero-initialised.

		sw::dsp::spectral::fft_forward<StateScalar>(buf);
		for (std::size_t i = 0; i < fft_size_; ++i) {
			buf[i] = buf[i] * H_[i];
		}
		sw::dsp::spectral::fft_inverse<StateScalar>(buf);

		// Output: discard the first hist_len (aliased) samples, keep the next
		// block_size.
		mtl::vec::dense_vector<SampleScalar> out(block_size_);
		for (std::size_t i = 0; i < block_size_; ++i) {
			out[i] = static_cast<SampleScalar>(buf[hist_len + i].real());
		}

		// Update history: last hist_len samples of new input.
		if (hist_len > 0) {
			if (block_size_ >= hist_len) {
				for (std::size_t i = 0; i < hist_len; ++i) {
					history_[i] = input[block_size_ - hist_len + i];
				}
			} else {
				// block_size < hist_len: shift existing history, then append input.
				std::size_t shift = hist_len - block_size_;
				for (std::size_t i = 0; i < shift; ++i) {
					history_[i] = history_[i + block_size_];
				}
				for (std::size_t i = 0; i < block_size_; ++i) {
					history_[shift + i] = input[i];
				}
			}
		}
		return out;
	}

	void reset() {
		for (std::size_t i = 0; i < history_.size(); ++i) history_[i] = SampleScalar{};
	}

	std::size_t block_size() const { return block_size_; }
	std::size_t fft_size() const { return fft_size_; }
	std::size_t filter_length() const { return filter_length_; }

private:
	std::size_t block_size_;
	std::size_t filter_length_;
	std::size_t fft_size_;
	mtl::vec::dense_vector<complex_t> H_;
	mtl::vec::dense_vector<SampleScalar> history_;
};

// -----------------------------------------------------------------------------
// One-shot free-function convolutions.
// -----------------------------------------------------------------------------

// Overlap-add linear convolution: returns x * h of length x.size() + h.size() - 1.
// block_size: FFT block size (L). If 0, a reasonable default is chosen.
template <DspField T>
mtl::vec::dense_vector<T>
overlap_add_convolve(const mtl::vec::dense_vector<T>& x,
                     const mtl::vec::dense_vector<T>& h,
                     std::size_t block_size = 0) {
	if (x.size() == 0 || h.size() == 0) {
		return mtl::vec::dense_vector<T>(0);
	}
	if (block_size == 0) {
		// Choose L so FFT size is ~4*M (a reasonable default).
		std::size_t target = 4 * h.size();
		block_size = detail::next_power_of_2(target) - (h.size() - 1);
		if (block_size < 1) block_size = 1;
	}

	OverlapAddConvolver<T, T, T> oa(h, block_size);

	std::size_t total = x.size() + h.size() - 1;
	mtl::vec::dense_vector<T> y(total, T{});

	std::size_t written = 0;
	std::size_t read = 0;
	mtl::vec::dense_vector<T> chunk(block_size, T{});
	while (read < x.size()) {
		std::size_t avail = std::min(block_size, x.size() - read);
		for (std::size_t i = 0; i < avail; ++i) chunk[i] = x[read + i];
		for (std::size_t i = avail; i < block_size; ++i) chunk[i] = T{};
		auto out = oa.process_block(std::span<const T>(chunk.data(), block_size));
		for (std::size_t i = 0; i < block_size && written < total; ++i) {
			y[written++] = out[i];
		}
		read += block_size;
	}
	// Flush trailing M-1 samples.
	auto tail = oa.flush();
	for (std::size_t i = 0; i < tail.size() && written < total; ++i) {
		y[written++] = tail[i];
	}
	return y;
}

// Overlap-save linear convolution: returns x * h of length x.size() + h.size() - 1.
template <DspField T>
mtl::vec::dense_vector<T>
overlap_save_convolve(const mtl::vec::dense_vector<T>& x,
                      const mtl::vec::dense_vector<T>& h,
                      std::size_t block_size = 0) {
	if (x.size() == 0 || h.size() == 0) {
		return mtl::vec::dense_vector<T>(0);
	}
	if (block_size == 0) {
		std::size_t target = 4 * h.size();
		block_size = detail::next_power_of_2(target) - (h.size() - 1);
		if (block_size < 1) block_size = 1;
	}

	// Overlap-save processes x + (M-1) trailing zeros to flush the convolution tail.
	std::size_t tail_zeros = h.size() - 1;
	std::size_t total = x.size() + h.size() - 1;

	OverlapSaveConvolver<T, T, T> os(h, block_size);
	mtl::vec::dense_vector<T> y(total, T{});

	std::size_t written = 0;
	std::size_t total_input = x.size() + tail_zeros;
	std::size_t read = 0;
	mtl::vec::dense_vector<T> chunk(block_size, T{});
	while (read < total_input) {
		for (std::size_t i = 0; i < block_size; ++i) {
			std::size_t src = read + i;
			chunk[i] = (src < x.size()) ? x[src] : T{};
		}
		auto out = os.process_block(std::span<const T>(chunk.data(), block_size));
		for (std::size_t i = 0; i < block_size && written < total; ++i) {
			y[written++] = out[i];
		}
		read += block_size;
	}
	return y;
}

} // namespace sw::dsp
