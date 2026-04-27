#pragma once
// trace_averaging.hpp: spectrum-analyzer trace averaging modes.
//
// Where detector modes reduce *within* a bin (#177), trace averaging
// reduces *across sweeps*. The same five-mode set found on commercial
// analyzers:
//
//   Linear      - cumulative unweighted mean of all sweeps since reset.
//                 The classic noise-floor smoother. O(1) memory beyond
//                 the trace buffer; running sum accumulates in double
//                 to keep narrow SampleScalar types from drifting.
//   Exponential - single-pole IIR: y[n] = alpha*x[n] + (1-alpha)*y[n-1].
//                 The "live" smoother that keeps tracking changes.
//                 alpha in (0, 1]; lower alpha = more smoothing,
//                 longer settling time.
//   MaxHold     - element-wise max across all sweeps since reset.
//                 Comparison-only, precision-blind.
//   MinHold     - element-wise min across all sweeps since reset.
//   MaxHoldN    - max-hold over a rolling window of the last N sweeps.
//                 Stores N traces in a ring buffer; on each accept,
//                 the output is the element-wise max over the ring.
//                 Memory O(N * trace_length), time O(N * trace_length)
//                 per accept.
//
// Linear is intentionally cumulative (not windowed). A windowed linear
// average would need a ring buffer of N traces, identical in cost to
// MaxHoldN but for a different reduction. Callers wanting "Linear N"
// can approximate with Exponential at alpha = 1/N (matches the
// effective time constant of an N-sweep moving average) and accept the
// IIR boundary characteristics, or call accept_sweep() N times then
// reset() for a strict batch. The simpler cumulative form serves the
// noise-floor measurement use case directly.
//
// Mixed-precision contract:
//   - Linear: running sum accumulates in double; current_trace() is
//     filled by dividing the sum by sweeps_accumulated() and casting
//     to SampleScalar on read.
//   - Exponential: arithmetic in SampleScalar (single-pole IIR per
//     bin); narrow types may show drift toward the alpha-quantization
//     step, same dynamic as any leaky integrator. The IIR loop
//     applies DenormalPrevention<SampleScalar> (a tiny alternating
//     AC injection) on each update — no-op for posit / fixpnt, flushes
//     denormals on IEEE float / double. Same pattern as the IIR
//     stages in acquisition/nco.hpp and acquisition/halfband.hpp.
//   - MaxHold / MinHold / MaxHoldN: comparison-only, precision-blind.
//     The stored values are bit-exact copies of the input.
//
// Edge cases:
//   - Empty input span -> std::invalid_argument.
//   - Length mismatch (input.size() != trace_length) -> invalid_argument.
//   - Reading current_trace() before any accept_sweep() returns the
//     mode's initial state (zeros for all modes; the value isn't
//     meaningful until at least one sweep has been accepted, and
//     sweeps_accumulated() == 0 distinguishes that case).
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/denormal.hpp>

namespace sw::dsp::spectrum {

template <DspOrderedField SampleScalar>
	requires ConvertibleToDouble<SampleScalar>
class TraceAverager {
public:
	using sample_scalar = SampleScalar;

	enum class Mode {
		Linear,
		Exponential,
		MaxHold,
		MinHold,
		MaxHoldN
	};

	// trace_length: number of bins per sweep (must be > 0).
	// mode:         averaging mode (see enum above).
	// config:       mode-specific scalar parameter:
	//                 MaxHoldN     -> window N >= 1 (cast from double)
	//                 Exponential  -> alpha in (0, 1]
	//                 Linear / MaxHold / MinHold -> ignored
	//               Using `double` rather than std::size_t to carry
	//               either an integer window or a fractional alpha
	//               through one constructor argument.
	TraceAverager(std::size_t trace_length,
	              Mode mode,
	              double config = 0.0)
		: trace_length_(trace_length),
		  mode_(mode),
		  current_(trace_length) {
		if (trace_length == 0)
			throw std::invalid_argument(
				"TraceAverager: trace_length must be > 0");
		switch (mode_) {
			case Mode::Linear:
				// dense_vector doesn't support .resize() / .assign(),
				// so size is fixed at construction. Default-constructed
				// sum_ is zero-sized; assign a properly-sized instance
				// here. Other modes leave sum_ at size zero (it's
				// never read for them).
				sum_ = mtl::vec::dense_vector<double>(trace_length);
				break;
			case Mode::Exponential:
				if (!(config > 0.0 && config <= 1.0))
					throw std::invalid_argument(
						"TraceAverager: Exponential requires alpha in (0, 1] (got "
						+ std::to_string(config) + ")");
				alpha_ = config;
				break;
			case Mode::MaxHold:
			case Mode::MinHold:
				break;
			case Mode::MaxHoldN:
				// Two checks: N >= 1 catches negatives, zero, and NaN
				// (NaN >= 1 is false). Then config == floor(config)
				// catches non-integer values like 2.5 — without this
				// they'd silently truncate via the static_cast. NaN also
				// fails this second check (NaN != NaN).
				if (!(config >= 1.0))
					throw std::invalid_argument(
						"TraceAverager: MaxHoldN requires window N >= 1 (got "
						+ std::to_string(config) + ")");
				if (config != std::floor(config))
					throw std::invalid_argument(
						"TraceAverager: MaxHoldN requires integer-valued window N (got "
						+ std::to_string(config) + ")");
				window_n_ = static_cast<std::size_t>(config);
				ring_.resize(window_n_);
				for (auto& t : ring_)
					t = mtl::vec::dense_vector<SampleScalar>(trace_length);
				break;
		}
		reset_state();
	}

	// Push a new sweep into the averager. Throws on length mismatch.
	void accept_sweep(std::span<const SampleScalar> trace) {
		if (trace.size() != trace_length_)
			throw std::invalid_argument(
				"TraceAverager::accept_sweep: trace length "
				+ std::to_string(trace.size())
				+ " does not match averager's trace_length "
				+ std::to_string(trace_length_));

		switch (mode_) {
			case Mode::Linear:
				for (std::size_t i = 0; i < trace_length_; ++i)
					sum_[i] += static_cast<double>(trace[i]);
				++sweeps_;
				// current_ holds the running mean, recomputed on every
				// accept. For mixed precision this is the cleanest
				// place to do the divide-and-cast: the accumulator
				// stays in double, current_ stays in SampleScalar.
				for (std::size_t i = 0; i < trace_length_; ++i)
					current_[i] = static_cast<SampleScalar>(
						sum_[i] / static_cast<double>(sweeps_));
				break;

			case Mode::Exponential:
				if (sweeps_ == 0) {
					// First sweep seeds the IIR state directly. Blending
					// with zero would mistakenly drag the first output
					// toward zero by (1 - alpha).
					for (std::size_t i = 0; i < trace_length_; ++i)
						current_[i] = trace[i];
				} else {
					// y[i] = alpha*x[i] + (1-alpha)*y[i-1] + denormal AC
					// Computed in SampleScalar; alpha multiplied via
					// the SampleScalar(double) ctor for type uniformity.
					// `+ denormal_.ac()` injects a tiny alternating value
					// that flushes accumulator denormals on IEEE types
					// (no-op on posit / fixpnt). Same pattern as the IIR
					// stages in nco.hpp / halfband.hpp / src.hpp.
					const SampleScalar a = static_cast<SampleScalar>(alpha_);
					const SampleScalar one_minus_a =
						static_cast<SampleScalar>(1.0 - alpha_);
					for (std::size_t i = 0; i < trace_length_; ++i)
						current_[i] = a * trace[i]
						            + one_minus_a * current_[i]
						            + denormal_.ac();
				}
				++sweeps_;
				break;

			case Mode::MaxHold:
				if (sweeps_ == 0) {
					for (std::size_t i = 0; i < trace_length_; ++i)
						current_[i] = trace[i];
				} else {
					for (std::size_t i = 0; i < trace_length_; ++i)
						if (trace[i] > current_[i]) current_[i] = trace[i];
				}
				++sweeps_;
				break;

			case Mode::MinHold:
				if (sweeps_ == 0) {
					for (std::size_t i = 0; i < trace_length_; ++i)
						current_[i] = trace[i];
				} else {
					for (std::size_t i = 0; i < trace_length_; ++i)
						if (trace[i] < current_[i]) current_[i] = trace[i];
				}
				++sweeps_;
				break;

			case Mode::MaxHoldN: {
				// Push the new sweep into the ring at ring_pos_; the
				// oldest sweep there is overwritten. Then compute the
				// element-wise max over the (up to window_n_) valid
				// ring entries. Until window_n_ sweeps have been seen
				// the ring is partially populated; iterate over the
				// valid prefix only.
				for (std::size_t i = 0; i < trace_length_; ++i)
					ring_[ring_pos_][i] = trace[i];
				ring_pos_ = (ring_pos_ + 1) % window_n_;
				++sweeps_;
				const std::size_t valid =
					sweeps_ < window_n_ ? sweeps_ : window_n_;
				// Seed current_ from the first valid ring entry, then
				// max in the rest. Using `valid` as the bound prevents
				// uninitialized ring entries from corrupting the output
				// during ring fill-up.
				for (std::size_t i = 0; i < trace_length_; ++i)
					current_[i] = ring_[0][i];
				for (std::size_t k = 1; k < valid; ++k)
					for (std::size_t i = 0; i < trace_length_; ++i)
						if (ring_[k][i] > current_[i])
							current_[i] = ring_[k][i];
				break;
			}
		}
	}

	// Read the current accumulated trace. Length always equals
	// trace_length(); content is meaningless when sweeps_accumulated() == 0.
	[[nodiscard]] std::span<const SampleScalar> current_trace() const {
		return std::span<const SampleScalar>(current_.data(), trace_length_);
	}

	// Discard accumulated state and return to the construction-time
	// initial state. Mode and config (alpha / window_n) are preserved.
	void reset() { reset_state(); }

	[[nodiscard]] std::size_t sweeps_accumulated() const { return sweeps_; }
	[[nodiscard]] std::size_t trace_length()       const { return trace_length_; }
	[[nodiscard]] Mode        mode()               const { return mode_; }

private:
	void reset_state() {
		sweeps_ = 0;
		ring_pos_ = 0;
		// Zero current_ so a read-before-accept returns a defined value.
		for (std::size_t i = 0; i < trace_length_; ++i)
			current_[i] = SampleScalar{};
		if (mode_ == Mode::Linear) {
			for (std::size_t i = 0; i < trace_length_; ++i) sum_[i] = 0.0;
		}
		// Ring buffer entries don't need explicit zeroing because the
		// `valid` bound in MaxHoldN's accept_sweep keeps the read
		// confined to written entries.
		// denormal_'s alternating-sign tracker keeps state across
		// accept_sweep calls; reset() is supposed to return us to
		// fresh-construction state, so reseed denormal_ as well.
		// Otherwise a fresh and a reset averager would diverge by the
		// 1e-8 AC sign on the first Exponential update after reset.
		denormal_ = DenormalPrevention<SampleScalar>{};
	}

	std::size_t trace_length_;
	Mode        mode_;
	std::size_t window_n_ = 0;       // MaxHoldN
	double      alpha_    = 0.0;     // Exponential
	std::size_t sweeps_   = 0;
	std::size_t ring_pos_ = 0;       // MaxHoldN write head

	mtl::vec::dense_vector<SampleScalar>             current_;
	mtl::vec::dense_vector<double>                   sum_;       // Linear
	std::vector<mtl::vec::dense_vector<SampleScalar>> ring_;     // MaxHoldN
	DenormalPrevention<SampleScalar>                 denormal_;  // Exponential IIR
};

} // namespace sw::dsp::spectrum
