#pragma once
// signal_probe.hpp: Pipeline sample-tap primitives.
//
// The probe module lets a user attach a "tap" to any point in a
// pipeline to capture samples flowing through, so those samples can
// be dumped to CSV/JSON and analyzed / visualized externally
// (typically by the mp-dsp-python peer repo).
//
// The three primitives here are:
//
//   SignalProbe<T>  - fixed-capacity ring buffer that captures samples
//                     pushed to it. Wraps when full; oldest samples
//                     survive at the front of the returned span.
//   NoOpProbe<T>    - API-compatible drop-in whose push() does nothing.
//                     Templated pipelines can select at compile time
//                     between real probes and this no-op so production
//                     builds pay zero runtime cost.
//   ProbedStage<S>  - wraps any stage exposing process()/process_block()
//                     and taps each output sample into a SignalProbe.
//                     Constructed via make_probe(stage, ...).
//
// This is the internal-introspection cousin of #133's oscilloscope
// demo: probes look INTO a pipeline while an oscilloscope looks AT an
// external waveform. Probes stay templated on the pipeline sample
// type so mixed-precision configurations are captured faithfully.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp::probe {

// ============================================================================
// SignalProbe - capture up to `capacity` most-recent samples
// ============================================================================

template <DspScalar SampleScalar>
class SignalProbe {
public:
	using sample_scalar = SampleScalar;

	// label: human-readable name written into the sidecar JSON so a
	//   viewer can distinguish streams from multiple probes.
	// capacity: ring-buffer size. Push() past capacity overwrites the
	//   oldest sample; samples() returns oldest-first up to `capacity`
	//   samples in chronological order.
	// sample_rate_hz: metadata for downstream views/plotting; the probe
	//   itself does not assume anything about the sample rate.
	SignalProbe(std::string label,
	             std::size_t capacity,
	             double sample_rate_hz)
		: label_(std::move(label)),
		  capacity_(capacity),
		  sample_rate_hz_(sample_rate_hz),
		  buffer_(capacity, SampleScalar{}),
		  write_pos_(0),
		  count_(0) {
		if (capacity == 0)
			throw std::invalid_argument("SignalProbe: capacity must be > 0");
	}

	// Ingest a sample. Overwrites the oldest sample once capacity is
	// reached (standard ring-buffer behavior). O(1).
	void push(SampleScalar x) {
		buffer_[write_pos_] = x;
		write_pos_ = (write_pos_ + 1) % capacity_;
		if (count_ < capacity_) ++count_;
	}

	// Returns the captured samples in chronological order (oldest
	// first, newest last). Under-fill: shorter than `capacity_` while
	// count_ < capacity_. Wrap-around: unrolls the ring into a linear
	// span via a lazy contiguous copy on first access after wrap.
	//
	// The span is valid until the next call to push() or samples().
	std::span<const SampleScalar> samples() const {
		if (count_ < capacity_) {
			// Not yet wrapped - samples sit in [0, count_) directly.
			return std::span<const SampleScalar>(buffer_.data(), count_);
		}
		// Wrapped - reorder into `linear_` on demand.
		linear_.resize(capacity_);
		const std::size_t split = write_pos_;   // oldest is at write_pos_
		std::copy(buffer_.begin() + split, buffer_.end(), linear_.begin());
		std::copy(buffer_.begin(), buffer_.begin() + split,
		           linear_.begin() + (capacity_ - split));
		return std::span<const SampleScalar>(linear_.data(), capacity_);
	}

	const std::string& label()       const { return label_; }
	double             sample_rate() const { return sample_rate_hz_; }
	std::size_t        capacity()    const { return capacity_; }
	std::size_t        size()        const { return count_; }
	bool               is_full()     const { return count_ >= capacity_; }

	// Reset the probe to its initial (empty) state.
	void clear() {
		std::fill(buffer_.begin(), buffer_.end(), SampleScalar{});
		write_pos_ = 0;
		count_ = 0;
	}

	// Dump captured samples to a two-column CSV:
	//   sample_index,sample_value
	// Compatible with acquisition_demo.csv style. Also writes a
	// sidecar JSON file next to the CSV (path + ".json") holding
	// label, sample_rate, capture_size for the mp-dsp-python viewer.
	void dump_csv(const std::string& path) const {
		std::ofstream out(path);
		if (!out) throw std::runtime_error(
			"SignalProbe::dump_csv: cannot open " + path);
		out << "sample_index,sample_value\n";
		out << std::setprecision(17);
		auto s = samples();
		for (std::size_t i = 0; i < s.size(); ++i) {
			out << i << "," << static_cast<double>(s[i]) << "\n";
		}
		// Sidecar JSON.
		std::ofstream jout(path + ".json");
		if (!jout) throw std::runtime_error(
			"SignalProbe::dump_csv: cannot open " + path + ".json");
		jout << "{\n"
		     << "  \"label\": \"" << label_ << "\",\n"
		     << "  \"sample_rate_hz\": " << sample_rate_hz_ << ",\n"
		     << "  \"capacity\": " << capacity_ << ",\n"
		     << "  \"captured\": " << s.size() << "\n"
		     << "}\n";
	}

private:
	std::string                  label_;
	std::size_t                  capacity_;
	double                       sample_rate_hz_;
	std::vector<SampleScalar>    buffer_;
	std::size_t                  write_pos_;
	std::size_t                  count_;
	// Lazy linearization buffer, populated by samples() when the ring
	// has wrapped. Marked mutable so samples() stays const.
	mutable std::vector<SampleScalar> linear_;
};

// ============================================================================
// NoOpProbe - API-parity, zero-cost drop-in for production builds
// ============================================================================

template <DspScalar SampleScalar>
class NoOpProbe {
public:
	using sample_scalar = SampleScalar;

	NoOpProbe(std::string /*label*/,
	           std::size_t /*capacity*/,
	           double /*sample_rate_hz*/) noexcept {}

	void push(SampleScalar) noexcept {}

	std::span<const SampleScalar> samples() const noexcept { return {}; }

	const std::string& label() const noexcept {
		static const std::string k_empty;
		return k_empty;
	}
	double      sample_rate() const noexcept { return 0.0; }
	std::size_t capacity()    const noexcept { return 0; }
	std::size_t size()        const noexcept { return 0; }
	bool        is_full()     const noexcept { return false; }

	void clear()                                 noexcept {}
	void dump_csv(const std::string&) const      noexcept {}
};

// ============================================================================
// ProbedStage - wrap any pipeline stage + push each output to a probe
// ============================================================================
//
// The wrapped stage must expose:
//   * a `sample_scalar` typedef
//   * a `process(sample_scalar)` method that returns EITHER
//     `sample_scalar` (simple stages)
//     OR `std::pair<bool, sample_scalar>` (decimating stages, where
//     the bool flags "output ready this cycle")
//
// Optionally exposes process_block(span, span) or similar; the wrapper
// forwards those unchanged (they land in the underlying stage) but
// probes only sample-at-a-time cycles. Callers wanting probes on
// block-processed stages should push through process() one at a time.

namespace detail {

template <class T>
concept HasSampleScalar = requires { typename T::sample_scalar; };

} // namespace detail

template <class Stage,
          template <class> class Probe = SignalProbe>
class ProbedStage {
	static_assert(detail::HasSampleScalar<Stage>,
		"ProbedStage: Stage must expose a `sample_scalar` typedef");
public:
	using sample_scalar = typename Stage::sample_scalar;

	// Take ownership of an existing stage plus construct a probe with
	// the given metadata. The stage is moved into the wrapper.
	ProbedStage(Stage stage,
	             std::string label,
	             std::size_t capacity,
	             double sample_rate_hz)
		: stage_(std::move(stage)),
		  probe_(std::move(label), capacity, sample_rate_hz) {}

	// Delegate process() to the stage, then push the (possibly wrapped)
	// output to the probe. Requires the stage's process to return a
	// value convertible to sample_scalar OR a std::pair<bool, S> where
	// the bool controls whether the sample gets pushed.
	template <class... Args>
	auto process(Args&&... args) {
		auto out = stage_.process(std::forward<Args>(args)...);
		push_result(out);
		return out;
	}

	// Non-const passthrough accessors so callers can still reach
	// stage-specific reset()/tap-query methods.
	Stage&                  stage()       { return stage_; }
	const Stage&            stage() const { return stage_; }
	Probe<sample_scalar>&        probe()       { return probe_; }
	const Probe<sample_scalar>&  probe() const { return probe_; }

private:
	template <class S>
	void push_result(const S& s) {
		if constexpr (requires { s.first; s.second; }) {
			// process() returned a std::pair<bool, sample_scalar>:
			// decimating-stage convention. Only push when a real
			// output is emitted.
			if (s.first) probe_.push(s.second);
		} else {
			probe_.push(static_cast<sample_scalar>(s));
		}
	}

	Stage                       stage_;
	Probe<sample_scalar>        probe_;
};

// Convenience factory: `auto p = make_probe(stage, "after_mixer", 4096, fs);`
// Deduces Stage from the passed-in argument.
template <class Stage>
auto make_probe(Stage stage,
                std::string label,
                std::size_t capacity,
                double sample_rate_hz) {
	return ProbedStage<Stage>(std::move(stage), std::move(label),
	                           capacity, sample_rate_hz);
}

} // namespace sw::dsp::probe
