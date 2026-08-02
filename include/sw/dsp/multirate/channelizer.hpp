#pragma once
// channelizer.hpp: Bellanger polyphase M-channel analysis channelizer.
//
// A single length-MK polyphase prototype filter followed by an M-point
// inverse FFT produces M parallel channel outputs at rate f_s / M -
// versus M parallel DDCs at O(M*K) cost per input sample for the naive
// implementation. The polyphase-plus-IFFT construction costs O(K + M
// log M) per input sample; for M = 8 or 16 and typical K in [8, 32]
// the IFFT is smaller than the FIR work and the total is about 1/M
// the cost of the naive design.
//
// Construction (Bellanger 1976):
//   1. Design a lowpass prototype h[n] of length M*K, cutoff 1/(2M), so
//      it isolates the baseband channel [-f_s/(2M), +f_s/(2M)] out of
//      the [-f_s/2, +f_s/2] input.
//   2. Polyphase decompose: E_k[p] = h[pM + k], for k = 0..M-1 and
//      p = 0..K-1.
//   3. Per input block of M samples x[mM+0], x[mM+1], ..., x[mM+M-1]:
//        a. Feed sample x[mM+k] to sub-filter k, advancing it one tap.
//           Collect sub-filter outputs y_k[m].
//        b. IFFT the M-vector y_k[m], k = 0..M-1.
//        c. IFFT output at index c is channel-c output at time m.
//
// The channel-c "filter" is h[n] * exp(j 2 pi c n / M) - a bandpass
// centered on c * f_s / M. Modulation-in-time = frequency shift, and
// the polyphase decomposition of the modulated filter turns out to be
// E_k[p] * exp(j 2 pi c k / M) - which is exactly what an M-point
// inverse DFT applies across the sub-filter outputs.
//
// Sign convention: channel c corresponds to input frequency band
// centered on +c * f_s / M (positive frequency, IFFT-based).
//
// Three-scalar parameterization:
//   CoeffScalar  - polyphase prototype taps (design precision)
//   StateScalar  - sub-filter accumulator + IFFT twiddles
//   SampleScalar - input samples; channel outputs are complex_for_t
//                  wrapping SampleScalar
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/spectral/fft.hpp>

namespace sw::dsp::multirate {

template <DspField CoeffScalar  = double,
          DspField StateScalar  = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class Channelizer {
public:
	using coeff_scalar   = CoeffScalar;
	using state_scalar   = StateScalar;
	using sample_scalar  = SampleScalar;
	using complex_sample = complex_for_t<SampleScalar>;
	using complex_state  = complex_for_t<StateScalar>;

	// M: number of channels. Must be a power of two (library FFT
	//    constraint). Output rate = input rate / M.
	// taps_per_phase: length of each polyphase sub-filter (K in the
	//    header comment). Total prototype length = M * K. Longer =
	//    sharper channel edges + deeper adjacent-channel rejection at
	//    the cost of more compute per input sample and (K-1)/2 samples
	//    of intrinsic group delay.
	// kaiser_beta: prototype-filter Kaiser window sharpness. beta=8
	//    gives ~-58 dB stopband; beta=12 gives ~-115 dB.
	Channelizer(std::size_t M,
	             std::size_t taps_per_phase = 16,
	             double kaiser_beta = 8.0)
		: M_(M),
		  K_(taps_per_phase),
		  sub_taps_(design_polyphase_prototype(M, taps_per_phase,
		                                        kaiser_beta)),
		  sub_delay_(M, mtl::vec::dense_vector<SampleScalar>(taps_per_phase,
		                                                        SampleScalar{})),
		  write_pos_(0) {
		if (M == 0 || (M & (M - 1)) != 0)
			throw std::invalid_argument(
				"Channelizer: M must be a nonzero power of two");
	}

	// Push exactly M input samples through the M sub-filters, then IFFT
	// the sub-filter output vector, and return M complex channel outputs.
	// Throws if `block.size() != M`.
	mtl::vec::dense_vector<complex_sample>
	process(std::span<const SampleScalar> block) {
		if (block.size() != M_)
			throw std::invalid_argument(
				"Channelizer::process: block size must equal M");

		// Advance each sub-filter by one tap. Commutator convention:
		// sub-filter k receives block[M-1-k], i.e., the OLDEST sample of
		// the block (position 0) goes to sub-filter M-1, and the NEWEST
		// sample (position M-1) goes to sub-filter 0. This is the
		// standard Bellanger/Harris ordering that makes the sub-filter
		// output vector's IFFT come out with channel c at index c and
		// positive frequency band centered on +c*f_s/M.
		//
		// Derivation: sub-filter k's newest input at output time q must
		// be x[qM - k]. If block position m within a length-M block
		// corresponds to input x[qM - (M - 1) + m] (with position M-1
		// being x[qM]), then position (M - 1 - k) corresponds to input
		// x[qM - k], which is exactly what sub-filter k needs.
		mtl::vec::dense_vector<complex_state> ifft_in(M_, complex_state{});
		for (std::size_t k = 0; k < M_; ++k) {
			auto& delay = sub_delay_[k];
			delay[write_pos_] = block[M_ - 1 - k];

			// Compute sub-filter output at current write position.
			// The "newest" sample is at delay[write_pos_], "oldest" at
			// delay[(write_pos_ + 1) % K_].
			StateScalar acc{};
			std::size_t idx = write_pos_;
			const auto& taps = sub_taps_[k];
			for (std::size_t p = 0; p < K_; ++p) {
				acc = acc + static_cast<StateScalar>(taps[p])
				          * static_cast<StateScalar>(delay[idx]);
				idx = (idx == 0) ? (K_ - 1) : (idx - 1);
			}
			ifft_in[k] = complex_state(acc, StateScalar{});
		}
		write_pos_ = (write_pos_ + 1) % K_;

		// M-point inverse FFT. Channel c is IFFT output index c.
		sw::dsp::spectral::fft_inverse<StateScalar>(ifft_in);

		// Convert to SampleScalar-complex for the caller.
		mtl::vec::dense_vector<complex_sample> channels(M_);
		for (std::size_t c = 0; c < M_; ++c) {
			channels[c] = complex_sample(
				static_cast<SampleScalar>(ifft_in[c].real()),
				static_cast<SampleScalar>(ifft_in[c].imag()));
		}
		return channels;
	}

	// Clear all sub-filter delay lines. Useful between independent test
	// segments so prior samples don't bleed into a fresh measurement.
	void reset() {
		for (auto& d : sub_delay_)
			for (std::size_t i = 0; i < K_; ++i) d[i] = SampleScalar{};
		write_pos_ = 0;
	}

	std::size_t M() const { return M_; }
	std::size_t taps_per_phase() const { return K_; }
	std::size_t num_taps() const { return M_ * K_; }

private:
	// Design the length-M*K Kaiser-windowed sinc prototype at cutoff
	// 1/(2M), decompose into M sub-filters E_k[p] = h[pM + k]. Prototype
	// is scaled by M so channel-c output amplitude matches input
	// amplitude for a full-scale tone at the channel center.
	//
	// PURE: called from the constructor's initializer list.
	static std::vector<mtl::vec::dense_vector<CoeffScalar>>
	design_polyphase_prototype(std::size_t M, std::size_t K,
	                            double kaiser_beta) {
		if (K == 0)
			throw std::invalid_argument(
				"Channelizer: taps_per_phase must be > 0");

		const std::size_t N   = M * K;
		const double pi       = std::numbers::pi_v<double>;
		const double center   = static_cast<double>(N - 1) / 2.0;
		const double cutoff   = 0.5 / static_cast<double>(M);

		// Modified Bessel I0 for Kaiser - power series adequate for
		// beta in [4, 18].
		auto bessel_i0 = [](double x) {
			double sum = 1.0;
			double term = 1.0;
			for (int i = 1; i < 40; ++i) {
				term *= (x / (2.0 * i)) * (x / (2.0 * i));
				sum += term;
				if (term < 1e-18 * sum) break;
			}
			return sum;
		};
		const double i0_beta = bessel_i0(kaiser_beta);

		mtl::vec::dense_vector<double> h(N);
		double sum = 0.0;
		for (std::size_t n = 0; n < N; ++n) {
			const double x = static_cast<double>(n) - center;
			double sinc;
			if (std::abs(x) < 1e-12) sinc = 1.0;
			else {
				const double arg = 2.0 * pi * cutoff * x;
				sinc = std::sin(arg) / arg;
			}
			const double r = 2.0 * static_cast<double>(n)
			                 / static_cast<double>(N - 1) - 1.0;
			const double w = bessel_i0(kaiser_beta * std::sqrt(1.0 - r * r))
			                 / i0_beta;
			h[n] = w * sinc * (2.0 * cutoff);  // sinc*2c normalizes DC gain
			sum += h[n];
		}
		if (std::abs(sum) < 1e-300)
			throw std::runtime_error(
				"Channelizer: prototype summed to zero");
		// Normalize to unity DC gain, then scale by M so channel-center
		// tones survive the IFFT-across-M-sub-filters at unity amplitude.
		const double scale = static_cast<double>(M) / sum;
		for (std::size_t n = 0; n < N; ++n) h[n] *= scale;

		// Polyphase decomposition E_k[p] = h[pM + k].
		std::vector<mtl::vec::dense_vector<CoeffScalar>> banks;
		banks.reserve(M);
		for (std::size_t k = 0; k < M; ++k) {
			mtl::vec::dense_vector<CoeffScalar> sub(K);
			for (std::size_t p = 0; p < K; ++p) {
				sub[p] = static_cast<CoeffScalar>(h[p * M + k]);
			}
			banks.push_back(std::move(sub));
		}
		return banks;
	}

	std::size_t M_;
	std::size_t K_;
	std::vector<mtl::vec::dense_vector<CoeffScalar>>   sub_taps_;   // [M][K]
	std::vector<mtl::vec::dense_vector<SampleScalar>>  sub_delay_;  // [M][K]
	std::size_t write_pos_;
};

} // namespace sw::dsp::multirate
