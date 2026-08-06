#pragma once
// fractional_delay.hpp: Polyphase fractional-delay line with runtime-
// variable delay.
//
// A length-N*L FIR filter designed as a lowpass at cutoff 1/(2L),
// decomposed into L phases each of length N, gives a filter bank where
// phase p represents a fractional delay of p/L samples. Selecting a
// phase at runtime lets a stream be delayed by any multiple of 1/L
// samples without re-designing any taps - the L phases are pre-computed
// at construction and the "which phase" decision is a single call
// to std::round.
//
// This is the runtime-variable counterpart to
// `sw::dsp::instrument::FractionalDelay` (which sets its delay at
// construction and redesigns its FIR whenever `set_delay` is called).
// Use this class when the delay is:
//   * Continuously variable (clock-recovery loops, phase trackers)
//   * Swept as part of an analysis (delay-vs-frequency sweeps)
//   * One of many values chosen from a small set (channel aligners
//     with per-channel skew)
//
// Reach for `instrument::FractionalDelay` instead when the delay is
// static and the polyphase memory overhead (L * K coefficients + a
// ring buffer) isn't worth the runtime-flexibility payoff.
//
// Three-scalar precision parameterization matches the rest of the
// library:
//   CoeffScalar  - the L*K polyphase-bank taps
//   StateScalar  - convolution accumulator
//   SampleScalar - input/output samples
//
// The prototype filter is designed in double and projected to
// CoeffScalar at the end - same pattern as `instrument::FractionalDelay`
// and the acquisition demos. Keeps cross-precision comparisons focused
// on streaming arithmetic, not filter-design variance.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp::multirate {

template <DspField CoeffScalar  = double,
          DspField StateScalar  = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class FractionalDelay {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;

	// L: number of polyphase phases. Delay resolution = 1/L samples.
	//    Higher L = finer resolution + more coefficient memory.
	// taps_per_phase: length of each phase filter, and MUST BE ODD
	//    (>= 3). Total prototype length = L * taps_per_phase. Longer =
	//    better in-band flatness and lower stopband. 11 is a reasonable
	//    default; 15-25 for stricter audio applications.
	//
	//    The oddness requirement is not incidental. With odd K the
	//    intrinsic group delay (K-1)/2 is an INTEGER, so phase 0 is
	//    sinc(k - center) sampled on the integers: exactly one non-zero
	//    tap, i.e. an unfiltered passthrough. An integer delay request
	//    therefore costs nothing at all — measured, the phase-0 impulse
	//    response reproduces its input to 3e-17. With even K the floor
	//    is a half-integer, so NO request lands on an unfiltered tap and
	//    every output, including a nominally integer delay, pays
	//    interpolation error and passband droop. Callers who want an
	//    even-length prototype want a different filter, not this one.
	// max_int_delay: maximum integer-part delay the caller will request.
	//    Ring buffer sized to taps_per_phase + max_int_delay so any
	//    delay in [group_delay, group_delay + max_int_delay + 1) can
	//    be served without wrap-around aliasing.
	// kaiser_beta: window sharpness. beta = 8 gives ~-58 dB stopband,
	//    beta = 12 gives ~-115 dB. Higher beta = wider transition band.
	FractionalDelay(std::size_t L,
	                 std::size_t taps_per_phase = 11,
	                 std::size_t max_int_delay = 32,
	                 double kaiser_beta = 8.0)
		: L_(L),
		  K_(taps_per_phase),
		  max_int_delay_(max_int_delay),
		  sub_taps_(design_bank(L, taps_per_phase, kaiser_beta)),
		  ring_(taps_per_phase + max_int_delay + 1, SampleScalar{}),
		  ring_size_(taps_per_phase + max_int_delay + 1),
		  write_pos_(0) {}

	// Push `in` into the delay line and return the interpolated output
	// at the specified offset. `offset_samples` is measured from the
	// just-written sample; the smallest expressible offset equals the
	// filter's intrinsic group delay (see `base_group_delay_samples`).
	//
	// Requests below the group-delay floor round up to the floor (they
	// cannot be served exactly - a filter cannot reconstruct samples
	// from the future). Requests above group_delay + max_int_delay
	// throw runtime_error because the ring buffer no longer holds the
	// required history.
	SampleScalar delay(SampleScalar in, double offset_samples) {
		// Write incoming sample.
		ring_[write_pos_] = in;
		write_pos_ = (write_pos_ + 1) % ring_size_;

		const double base = base_group_delay_samples();
		double d = offset_samples - base;
		if (d < 0.0) d = 0.0;

		// Round d to the nearest 1/L increment.
		const long total_units = static_cast<long>(
			std::llround(d * static_cast<double>(L_)));
		const std::size_t int_shift = static_cast<std::size_t>(total_units)
		                              / L_;
		const std::size_t phase     = static_cast<std::size_t>(total_units)
		                              % L_;
		if (int_shift > max_int_delay_) {
			throw std::runtime_error(
				"FractionalDelay::delay: offset exceeds max_int_delay");
		}

		// Read K samples from ring starting `int_shift` samples before
		// the just-written one; run the phase filter.
		const auto& taps = sub_taps_[phase];
		StateScalar acc{};
		// The most-recently-written sample lives at (write_pos_ - 1)
		// mod ring_size. Skip back `int_shift` more to reach the
		// "current" sample for this integer delay level.
		std::size_t idx = (write_pos_ + ring_size_ - 1 - int_shift)
		                  % ring_size_;
		for (std::size_t k = 0; k < K_; ++k) {
			acc = acc + static_cast<StateScalar>(taps[k])
			          * static_cast<StateScalar>(ring_[idx]);
			idx = (idx == 0) ? (ring_size_ - 1) : (idx - 1);
		}
		return static_cast<SampleScalar>(acc);
	}

	// Clear the ring buffer to zero and reset the write cursor.
	void reset() {
		for (std::size_t i = 0; i < ring_size_; ++i)
			ring_[i] = SampleScalar{};
		write_pos_ = 0;
	}

	// The polyphase filter has an intrinsic group delay of (K-1)/2
	// input samples (linear-phase FIR of length K). Requests to
	// `delay()` below this floor round up to it.
	double base_group_delay_samples() const {
		return static_cast<double>(K_ - 1) / 2.0;
	}

	std::size_t L() const { return L_; }
	std::size_t taps_per_phase() const { return K_; }
	std::size_t num_taps() const { return L_ * K_; }
	std::size_t max_int_delay() const { return max_int_delay_; }

private:
	// Design L phase filters directly. Phase p is a windowed-sinc that
	// implements fractional delay p/L samples on top of the intrinsic
	// (K-1)/2 group delay.
	//
	// h_p[k] = sinc(k - (K-1)/2 - p/L) * kaiser(k, beta),  k = 0..K-1
	// then normalize each phase so its DC gain is 1.
	//
	// PURE: no side effects. Called from the constructor's initializer
	// list.
	//
	// EVERY argument is validated here rather than in the constructor
	// body, because the body runs after the whole initializer list: a
	// check placed there cannot stop this function, or the ring-buffer
	// sizing beside it, from running on a bad value first.
	static std::vector<mtl::vec::dense_vector<CoeffScalar>>
	design_bank(std::size_t L, std::size_t K, double kaiser_beta) {
		if (L == 0)
			throw std::invalid_argument(
				"FractionalDelay: L must be > 0");
		// Odd K keeps the (K-1)/2 group delay an integer, which is what
		// makes phase 0 an exact passthrough. See the constructor's
		// parameter documentation.
		if (K < 3 || (K & 1U) == 0)
			throw std::invalid_argument(
				"FractionalDelay: taps_per_phase must be odd and >= 3, got " +
				std::to_string(K));

		const double pi     = std::numbers::pi_v<double>;
		const double center = static_cast<double>(K - 1) / 2.0;

		// Modified Bessel I0 for Kaiser window - power series good enough
		// for typical beta in [4, 18].
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
		auto kaiser_val = [&](std::size_t n, std::size_t N) {
			const double r = 2.0 * static_cast<double>(n)
			                 / static_cast<double>(N - 1) - 1.0;
			return bessel_i0(kaiser_beta * std::sqrt(1.0 - r * r)) / i0_beta;
		};

		std::vector<mtl::vec::dense_vector<CoeffScalar>> bank;
		bank.reserve(L);
		for (std::size_t p = 0; p < L; ++p) {
			const double frac = static_cast<double>(p)
			                    / static_cast<double>(L);
			mtl::vec::dense_vector<double> h(K);
			double sum = 0.0;
			for (std::size_t k = 0; k < K; ++k) {
				const double x = static_cast<double>(k) - center - frac;
				double s;
				if (std::abs(x) < 1e-12) s = 1.0;
				else {
					const double pix = pi * x;
					s = std::sin(pix) / pix;
				}
				const double w = kaiser_val(k, K);
				h[k] = w * s;
				sum += h[k];
			}
			if (std::abs(sum) < 1e-300)
				throw std::runtime_error(
					"FractionalDelay: phase filter summed to zero");
			mtl::vec::dense_vector<CoeffScalar> taps(K);
			for (std::size_t k = 0; k < K; ++k)
				taps[k] = static_cast<CoeffScalar>(h[k] / sum);
			bank.push_back(std::move(taps));
		}
		return bank;
	}

	std::size_t L_;
	std::size_t K_;
	std::size_t max_int_delay_;
	std::vector<mtl::vec::dense_vector<CoeffScalar>> sub_taps_;
	mtl::vec::dense_vector<SampleScalar> ring_;
	std::size_t ring_size_;
	std::size_t write_pos_;
};

} // namespace sw::dsp::multirate
