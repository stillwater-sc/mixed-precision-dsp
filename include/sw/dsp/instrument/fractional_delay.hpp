#pragma once
// fractional_delay.hpp: Static sub-sample fractional-delay FIR for
// instrument-style data acquisition.
//
// Real instruments often need to align channels that are sampled at the same
// rate but are offset by a fixed fractional sample period — e.g., due to ADC
// clock skew, cable-length differences, or differential probe routing. This
// primitive shifts a stream by a static fractional delay in [0, 1) samples.
//
// Implementation: a windowed-sinc FIR designed at construction time. The
// ideal fractional delay's impulse response is a shifted sinc, h[n] =
// sinc(n - center - delay) for n = 0..N-1 where center = (N-1)/2. We
// truncate to N taps and apply a Hamming window to suppress sidelobes,
// then normalize so the DC gain is exactly 1.
//
// The class can be re-tuned via `set_delay()` without resetting the FIR
// state, useful for trim adjustments. Larger integer delays should be
// handled by the caller (e.g., via a ring buffer); this primitive only
// covers the sub-sample fractional part.
//
// Three-scalar precision parameterization:
//   CoeffScalar  — sinc-windowed taps (sensitive to delay accuracy)
//   StateScalar  — FIR delay-line accumulator
//   SampleScalar — input/output stream
//
// Tap design runs in `double` and is cast to `CoeffScalar` at the end —
// the same pattern used by EqualizerFilter (calibration.hpp). This keeps
// cross-precision SNR comparisons meaningful by isolating streaming-
// arithmetic precision from filter-design variance.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <stdexcept>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/filter/fir/fir_filter.hpp>

namespace sw::dsp::instrument {

template <DspField CoeffScalar  = double,
          DspField StateScalar  = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class FractionalDelay {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;

	// delay_samples: a real number in [0, 1) for the desired sub-sample
	//   delay. Values outside [0, 1) are rejected — larger integer
	//   delays should be handled by the caller via a ring buffer offset.
	// num_taps:      FIR length. Longer = better in-band flatness and
	//   more accurate group delay, at the cost of more arithmetic.
	//   31 taps is typical for 0.5-sample resolution with > -60 dB
	//   stopband. Must be odd so the integer-delay center is an integer
	//   sample (simpler design + symmetric phase response when delay=0).
	FractionalDelay(double delay_samples, std::size_t num_taps = 31)
		: num_taps_(num_taps),
		  delay_samples_(delay_samples),
		  fir_(design_taps(delay_samples, num_taps)) {}

	SampleScalar process(SampleScalar in) { return fir_.process(in); }

	void process_block(std::span<SampleScalar> samples) {
		fir_.process_block(samples);
	}

	void process_block(std::span<const SampleScalar> input,
	                   std::span<SampleScalar> output) {
		fir_.process_block(input, output);
	}

	// Re-tune the delay without resetting the FIR delay-line state. The
	// new taps replace the old in place; a brief transient is expected
	// as the delay-line samples adapt to the new impulse response.
	//
	// Strong exception guarantee: if design_taps throws (e.g., out-of-
	// range delay), the existing taps and delay_samples_ are unchanged.
	void set_delay(double delay_samples) {
		auto new_taps = design_taps(delay_samples, num_taps_);
		// design_taps returned cleanly — now commit the state changes.
		delay_samples_ = delay_samples;
		fir_.update_taps(new_taps);
	}

	// Clear the FIR delay-line state. Useful between independent test
	// runs or stream segments where prior samples should not bleed into
	// the new measurement.
	void reset() { fir_.reset(); }

	// Group delay introduced by this filter (in samples). For a
	// linear-phase FIR of odd length N with fractional delay d, the
	// group delay is exactly (N-1)/2 + d.
	double group_delay_samples() const {
		return delay_samples_ + static_cast<double>(num_taps_ - 1) / 2.0;
	}

	std::size_t num_taps() const { return num_taps_; }
	double      delay()    const { return delay_samples_; }

private:
	// Design windowed-sinc taps for a fractional delay. PURE: no side
	// effects on object state. Callers (constructor + set_delay) own
	// the state mutations.
	//
	//   h[n] = window[n] * sinc(n - center - delay) for n = 0..N-1
	//   then normalize so sum(h[n]) == 1 (DC gain = 1)
	//
	// Static so the constructor's initializer list can call it before
	// the object is fully constructed.
	static mtl::vec::dense_vector<CoeffScalar>
	design_taps(double delay_samples, std::size_t num_taps) {
		if (num_taps < 3 || (num_taps & 1U) == 0)
			throw std::invalid_argument(
				"FractionalDelay: num_taps must be odd and >= 3");
		if (!(delay_samples >= 0.0 && delay_samples < 1.0))
			throw std::invalid_argument(
				"FractionalDelay: delay_samples must be in [0, 1) "
				"(use a ring buffer for integer delays)");

		const double center = static_cast<double>(num_taps - 1) / 2.0;
		const double pi     = std::numbers::pi_v<double>;

		mtl::vec::dense_vector<double> h(num_taps);
		double sum = 0.0;
		for (std::size_t n = 0; n < num_taps; ++n) {
			// sinc((n - center) - delay), with sinc(0) = 1
			const double x = static_cast<double>(n) - center - delay_samples;
			double s;
			if (std::abs(x) < 1e-12) {
				s = 1.0;
			} else {
				const double pix = pi * x;
				s = std::sin(pix) / pix;
			}
			// Hamming window
			const double w = 0.54 - 0.46 * std::cos(
				2.0 * pi * static_cast<double>(n) /
				static_cast<double>(num_taps - 1));
			h[n] = w * s;
			sum += h[n];
		}
		// Normalize for unity DC gain. With delay=0 and a symmetric window
		// the unnormalized taps already sum close to 1; for nonzero delay
		// the symmetry breaks and a small renorm is needed.
		if (std::abs(sum) < 1e-300)
			throw std::runtime_error(
				"FractionalDelay: window+sinc summed to zero (degenerate)");
		mtl::vec::dense_vector<CoeffScalar> taps(num_taps);
		for (std::size_t n = 0; n < num_taps; ++n) {
			taps[n] = static_cast<CoeffScalar>(h[n] / sum);
		}
		return taps;
	}

	std::size_t                                                 num_taps_;
	double                                                      delay_samples_ = 0.0;
	FIRFilter<CoeffScalar, StateScalar, SampleScalar>           fir_;
};

} // namespace sw::dsp::instrument
