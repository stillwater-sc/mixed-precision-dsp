#pragma once
// vbw_filter.hpp: spectrum-analyzer video-bandwidth (VBW) post-detector LPF.
//
// In a swept-tuned analyzer the VBW filter sits AFTER the detector and
// BEFORE the trace memory. Its job is to smooth the detector output:
// lower VBW = more averaging = lower noise floor at the cost of slower
// response to amplitude changes; higher VBW = faster response but
// noisier trace. It's the standard scope-analyzer noise-vs-speed knob.
//
// Implementation: a simple leaky-integrator single-pole IIR
//
//     y[n] = alpha * x[n] + (1 - alpha) * y[n-1]    + denormal AC
//
// where alpha = 1 - exp(-2*pi*fc / fs). This is the "matched-z" /
// impulse-invariant form. -3 dB at fc to within 5% for fc << fs/2,
// which is the typical VBW-vs-RBW use (VBW is usually >= 10x lower
// than the analyzer's bin rate). For fc closer to Nyquist the
// approximation drifts; the constructor rejects fc > fs/2 to keep the
// filter inside its design range.
//
// Mixed-precision contract:
//   - CoeffScalar holds alpha (and 1 - alpha). For very low cutoffs
//     the pole sits very close to z = 1; coefficient precision matters
//     for stability at fc ~ fs/1000.
//   - StateScalar holds y_prev_. Narrow types may quantize the
//     integrator's settling tail.
//   - SampleScalar is the streaming I/O type.
//   - DenormalPrevention<SampleScalar> AC injection on each update -
//     same convention as the IIR stages in nco.hpp / halfband.hpp /
//     spectrum/trace_averaging.hpp.
//
// Retune (set_cutoff()) is bumpless: y_prev_ is preserved across the
// coefficient change. Real analyzers behave this way - the user
// sliding the VBW knob doesn't get a discontinuity in the displayed
// trace.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/denormal.hpp>

namespace sw::dsp::spectrum {

template <DspField CoeffScalar  = double,
          DspField StateScalar  = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class VBWFilter {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;

	// cutoff_hz:      -3 dB cutoff frequency. Must be > 0 and <= fs/2.
	// sample_rate_hz: streaming rate at which process() will be called.
	//                 Must be > 0.
	VBWFilter(double cutoff_hz, double sample_rate_hz)
		: sample_rate_hz_(sample_rate_hz) {
		// isfinite + positive. The two checks together exclude NaN
		// (rejected by both), -inf (rejected by `> 0.0`), and +inf
		// (rejected by isfinite). +inf is the sneaky one: it would
		// pass the > 0.0 check and then make alpha = 1 - exp(-x/inf)
		// = 0, giving a useless "y stuck at y_prev" filter.
		if (!std::isfinite(sample_rate_hz) || !(sample_rate_hz > 0.0))
			throw std::invalid_argument(
				"VBWFilter: sample_rate_hz must be positive and finite (got "
				+ std::to_string(sample_rate_hz) + ")");
		set_cutoff(cutoff_hz);   // designs alpha; sets cutoff_hz_
	}

	// Streaming - single sample.
	SampleScalar process(SampleScalar x) {
		// y[n] = alpha * x[n] + (1 - alpha) * y[n-1]   + denormal AC
		// Computed in StateScalar precision then cast to SampleScalar
		// on output. The cast at the boundary mirrors the convention
		// used by nco.hpp / halfband.hpp.
		const StateScalar y =
			static_cast<StateScalar>(alpha_)
			* static_cast<StateScalar>(x)
			+ static_cast<StateScalar>(one_minus_alpha_) * y_prev_;
		const SampleScalar y_out =
			static_cast<SampleScalar>(y) + denormal_.ac();
		y_prev_ = static_cast<StateScalar>(y_out);
		return y_out;
	}

	// In-place block process.
	void process_block(std::span<SampleScalar> samples) {
		for (auto& s : samples) s = process(s);
	}

	// Separate-span block process. Length mismatch throws.
	void process_block(std::span<const SampleScalar> input,
	                   std::span<SampleScalar> output) {
		if (input.size() != output.size())
			throw std::invalid_argument(
				"VBWFilter::process_block: input length "
				+ std::to_string(input.size())
				+ " does not match output length "
				+ std::to_string(output.size()));
		for (std::size_t i = 0; i < input.size(); ++i)
			output[i] = process(input[i]);
	}

	// Bumpless retune: alpha_ and one_minus_alpha_ are recomputed from
	// the new cutoff, but y_prev_ is preserved so the next process()
	// call continues smoothly from the running state. The user sliding
	// the VBW knob doesn't see a discontinuity.
	void set_cutoff(double cutoff_hz) {
		// Same isfinite-and-positive rule as the constructor.
		// +inf would pass `> 0.0` and then short-circuit through
		// the Nyquist check (`inf > 0.5*fs` is true), but explicit
		// isfinite gives a clearer error message.
		if (!std::isfinite(cutoff_hz) || !(cutoff_hz > 0.0))
			throw std::invalid_argument(
				"VBWFilter: cutoff_hz must be positive and finite (got "
				+ std::to_string(cutoff_hz) + ")");
		if (cutoff_hz > 0.5 * sample_rate_hz_)
			throw std::invalid_argument(
				"VBWFilter: cutoff_hz (" + std::to_string(cutoff_hz)
				+ ") exceeds Nyquist (sample_rate_hz / 2 = "
				+ std::to_string(0.5 * sample_rate_hz_) + ")");
		cutoff_hz_ = cutoff_hz;
		// alpha = 1 - exp(-2*pi*fc/fs)
		// matched-z / impulse-invariant single-pole LPF; -3 dB at fc
		// within 5% for fc << fs/2.
		constexpr double pi = 3.14159265358979323846;
		const double alpha_d = 1.0 - std::exp(-2.0 * pi * cutoff_hz / sample_rate_hz_);
		alpha_           = static_cast<CoeffScalar>(alpha_d);
		one_minus_alpha_ = static_cast<CoeffScalar>(1.0 - alpha_d);
	}

	// Clear y_prev_ to zero. Useful between independent measurements
	// or unrelated stream segments where the previous state should not
	// leak into the new run. denormal_'s alternating-sign tracker is
	// also reseeded so a fresh-construction and a reset-then-rerun
	// produce identical output (same fix as TraceAverager::reset).
	void reset() {
		y_prev_ = StateScalar{};
		denormal_ = DenormalPrevention<SampleScalar>{};
	}

	[[nodiscard]] double cutoff_hz()      const { return cutoff_hz_; }
	[[nodiscard]] double sample_rate_hz() const { return sample_rate_hz_; }

private:
	double      sample_rate_hz_;
	double      cutoff_hz_       = 0.0;
	CoeffScalar alpha_           = CoeffScalar{};
	CoeffScalar one_minus_alpha_ = CoeffScalar{};
	StateScalar y_prev_          = StateScalar{};
	DenormalPrevention<SampleScalar> denormal_;
};

} // namespace sw::dsp::spectrum
