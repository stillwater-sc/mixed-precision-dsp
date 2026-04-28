#pragma once
// rbw_filter.hpp: spectrum-analyzer resolution-bandwidth (RBW) filter.
//
// In a swept-tuned analyzer the RBW filter sits between the mixer
// (driven by the swept LO) and the detector. It selects a narrow
// frequency window around the IF; the detector measures the energy
// in that window. The filter's shape factor (the 60 dB / 3 dB
// bandwidth ratio) sets the analyzer's resolution: how close two
// adjacent spectral lines can be while still being seen as separate.
//
// Implementation: an order-N synchronously-tuned cascade of N
// identical RBJ-style band-pass biquads. The synchronous-tuned
// architecture is the classic analog implementation, easy to derive
// analytically, and gives a clean monotonic shape factor that
// improves with N. A 5th-order sync-tuned design has shape factor
// ~10x — comparable to a Gaussian filter for swept-analyzer use,
// and far simpler to design + tune at runtime.
//
// Per-biquad Q calibration:
//   For N stages cascaded, the cascade's response is |H_one|^N,
//   so the cascade -3 dB bandwidth is narrower than a single
//   stage's. To make the cascade -3 dB land at the user's
//   `bandwidth_hz`, each biquad gets:
//
//     Q_per_biquad = Q_target * sqrt(2^(1/N) - 1)
//                  = (f0 / BW_target) * sqrt(2^(1/N) - 1)
//
// Shape factor closed form:
//   For N synchronously tuned stages near resonance,
//     |H_cascade(f)|^2 = 1 / (1 + delta^2)^N
//   where delta = (f - f0) / (BW_per_biquad/2). Solving for the
//   60 dB and 3 dB bandwidths gives:
//
//     shape_factor(N) = sqrt((10^(6/N) - 1) / (2^(1/N) - 1))
//
//   N=5: ~10. N=3: ~16. N=1: ~2010 (unusable).
//
// Mixed-precision contract:
//   - CoeffScalar precision drives shape factor. Narrowband filters
//     have biquad coefficients with small differences (1 - alpha vs.
//     1 + alpha for the denominator); coefficient quantization shifts
//     the poles and degrades the shape factor. Coefficient design
//     happens in double and casts to CoeffScalar at the end (same
//     convention as VBW filter, EqualizerFilter, and the analyzer
//     demos — narrow fixpnt types can't represent intermediate
//     constants like 2*pi without saturation).
//   - StateScalar holds the per-stage delay-line variables. The
//     library's TransposedDirectFormII state form is preferred for
//     narrowband filters: each state variable accumulates smaller
//     quantities than the canonical Direct-Form II, reducing
//     sensitivity to narrow-StateScalar precision loss.
//   - SampleScalar is the streaming I/O type.
//
// Retune (`retune()`) is bumpless: only coefficients change, biquad
// state is preserved across the redesign. The "samples re-energize
// the new filter" semantics matches what a real analyzer's RBW knob
// does.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/filter/biquad/cascade.hpp>
#include <sw/dsp/filter/biquad/state.hpp>
#include <sw/dsp/types/biquad_coefficients.hpp>

namespace sw::dsp::spectrum {

template <DspField CoeffScalar  = double,
          DspField StateScalar  = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class RBWFilter {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;

	// Compile-time max stages. Order parameter at construction picks
	// the runtime stage count up to this bound. A user wanting >8
	// would have to change this constant — pragmatic since orders
	// above 8 don't add useful shape factor in practice.
	static constexpr int kMaxOrder = 8;

	// center_freq_hz:  filter resonance frequency. Must be > 0 and
	//                  < sample_rate / 2.
	// bandwidth_hz:    cascade -3 dB bandwidth. Must be > 0 and
	//                  comfortably less than 2 * center_freq (very
	//                  wide bands stop being narrowband).
	// sample_rate_hz:  streaming rate. > 0 and finite.
	// order:           number of biquad stages, in [1, kMaxOrder].
	//                  Default 5 for ~10x shape factor.
	RBWFilter(double center_freq_hz, double bandwidth_hz,
	          double sample_rate_hz, std::size_t order = 5)
		: sample_rate_hz_(sample_rate_hz) {
		if (!std::isfinite(sample_rate_hz) || !(sample_rate_hz > 0.0))
			throw std::invalid_argument(
				"RBWFilter: sample_rate_hz must be positive and finite (got "
				+ std::to_string(sample_rate_hz) + ")");
		if (order < 1 || order > static_cast<std::size_t>(kMaxOrder))
			throw std::invalid_argument(
				"RBWFilter: order must be in [1, "
				+ std::to_string(kMaxOrder) + "] (got "
				+ std::to_string(order) + ")");
		order_ = order;
		retune(center_freq_hz, bandwidth_hz);   // designs all stages
	}

	// Streaming - single sample.
	SampleScalar process(SampleScalar x) {
		return cascade_.process(x, state_);
	}

	// In-place block process.
	void process_block(std::span<SampleScalar> samples) {
		for (auto& s : samples) s = cascade_.process(s, state_);
	}

	// Separate-span block process. Length mismatch throws.
	void process_block(std::span<const SampleScalar> input,
	                   std::span<SampleScalar> output) {
		if (input.size() != output.size())
			throw std::invalid_argument(
				"RBWFilter::process_block: input length "
				+ std::to_string(input.size())
				+ " does not match output length "
				+ std::to_string(output.size()));
		for (std::size_t i = 0; i < input.size(); ++i)
			output[i] = cascade_.process(input[i], state_);
	}

	// Retune the filter. Coefficients are redesigned and projected to
	// CoeffScalar, but the per-stage TransposedDirectFormII state is
	// preserved so the existing samples re-energize the new
	// filter — no discontinuity in the displayed trace.
	void retune(double center_freq_hz, double bandwidth_hz) {
		if (!std::isfinite(center_freq_hz) || !(center_freq_hz > 0.0))
			throw std::invalid_argument(
				"RBWFilter: center_freq_hz must be positive and finite (got "
				+ std::to_string(center_freq_hz) + ")");
		if (center_freq_hz >= 0.5 * sample_rate_hz_)
			throw std::invalid_argument(
				"RBWFilter: center_freq_hz (" + std::to_string(center_freq_hz)
				+ ") must be strictly less than Nyquist (sample_rate_hz / 2 = "
				+ std::to_string(0.5 * sample_rate_hz_) + ")");
		if (!std::isfinite(bandwidth_hz) || !(bandwidth_hz > 0.0))
			throw std::invalid_argument(
				"RBWFilter: bandwidth_hz must be positive and finite (got "
				+ std::to_string(bandwidth_hz) + ")");

		// Realizability of the symmetric -3 dB shoulders: the lower
		// shoulder (fc - bw/2) must be > 0 and the upper shoulder
		// (fc + bw/2) must be < Nyquist. Without these checks an
		// over-wide bandwidth_hz would design a filter whose
		// nominal -3 dB shoulders sit outside the sampled band —
		// the cascade still produces output, but its frequency
		// response is not the requested symmetric narrowband shape.
		const double lo_shoulder = center_freq_hz - 0.5 * bandwidth_hz;
		const double hi_shoulder = center_freq_hz + 0.5 * bandwidth_hz;
		if (lo_shoulder <= 0.0)
			throw std::invalid_argument(
				"RBWFilter: lower -3 dB shoulder (center_freq_hz - bandwidth_hz/2 = "
				+ std::to_string(lo_shoulder)
				+ ") must be > 0 — bandwidth_hz too wide for center_freq_hz="
				+ std::to_string(center_freq_hz));
		if (hi_shoulder >= 0.5 * sample_rate_hz_)
			throw std::invalid_argument(
				"RBWFilter: upper -3 dB shoulder (center_freq_hz + bandwidth_hz/2 = "
				+ std::to_string(hi_shoulder)
				+ ") must be < Nyquist (sample_rate_hz / 2 = "
				+ std::to_string(0.5 * sample_rate_hz_) + ")");

		center_freq_hz_ = center_freq_hz;
		bandwidth_hz_   = bandwidth_hz;

		// Per-biquad Q so that the cascade -3 dB bandwidth equals the
		// user's requested bandwidth_hz. See header prose for the
		// derivation.
		const double n_d            = static_cast<double>(order_);
		const double sync_correction = std::sqrt(std::pow(2.0, 1.0 / n_d) - 1.0);
		const double Q_target        = center_freq_hz / bandwidth_hz;
		const double Q_per_biquad    = Q_target * sync_correction;

		// RBJ band-pass biquad coefficients (a0-normalized form):
		//   w0 = 2*pi*f0/fs
		//   alpha = sin(w0) / (2*Q)
		//   b0 =  alpha,  b1 = 0,  b2 = -alpha
		//   a0 =  1 + alpha,  a1 = -2*cos(w0),  a2 = 1 - alpha
		// Then divide everything by a0 so a0' = 1.
		constexpr double pi = 3.14159265358979323846;
		const double w0    = 2.0 * pi * center_freq_hz / sample_rate_hz_;
		const double cs    = std::cos(w0);
		const double sn    = std::sin(w0);
		const double alpha = sn / (2.0 * Q_per_biquad);
		const double a0    = 1.0 + alpha;

		BiquadCoefficients<CoeffScalar> bq{};
		bq.b0 = static_cast<CoeffScalar>( alpha     / a0);
		bq.b1 = static_cast<CoeffScalar>( 0.0);
		bq.b2 = static_cast<CoeffScalar>(-alpha     / a0);
		bq.a1 = static_cast<CoeffScalar>(-2.0 * cs  / a0);
		bq.a2 = static_cast<CoeffScalar>((1.0 - alpha) / a0);

		// Synchronously tuned: every active stage gets identical
		// coefficients. State arrays untouched (bumpless).
		cascade_.set_num_stages(static_cast<int>(order_));
		for (std::size_t i = 0; i < order_; ++i) {
			cascade_.stage(static_cast<int>(i)) = bq;
		}
	}

	// Closed-form analytical shape factor of an N-stage synchronously-
	// tuned cascade. For N synchronously tuned stages near resonance,
	//
	//   |H_cascade(f)|^2 = 1 / (1 + delta^2)^N
	//
	// where delta is the normalized offset from f0. Solving for the
	// 60 dB and 3 dB half-bandwidths and ratioing them:
	//
	//   shape_factor(N) = sqrt((10^(6/N) - 1) / (2^(1/N) - 1))
	//
	// Returned as a runtime double for inspection (test code, demo).
	[[nodiscard]] double shape_factor() const {
		const double n_d = static_cast<double>(order_);
		const double num = std::pow(10.0, 6.0 / n_d) - 1.0;
		const double den = std::pow(2.0, 1.0 / n_d) - 1.0;
		return std::sqrt(num / den);
	}

	// Clear all biquad delay-line state to zero. Useful between
	// independent measurements / unrelated stream segments. Coefficients
	// (alpha, etc.) and order are preserved.
	void reset() {
		for (auto& s : state_) s.reset();
	}

	[[nodiscard]] double      center_freq_hz()   const { return center_freq_hz_; }
	[[nodiscard]] double      bandwidth_hz()     const { return bandwidth_hz_;   }
	[[nodiscard]] double      sample_rate_hz()   const { return sample_rate_hz_; }
	[[nodiscard]] std::size_t order()            const { return order_;          }

private:
	double      sample_rate_hz_;
	double      center_freq_hz_ = 0.0;
	double      bandwidth_hz_   = 0.0;
	std::size_t order_          = 0;
	Cascade<CoeffScalar, kMaxOrder>                       cascade_{};
	std::array<TransposedDirectFormII<StateScalar>, kMaxOrder> state_{};
};

} // namespace sw::dsp::spectrum
