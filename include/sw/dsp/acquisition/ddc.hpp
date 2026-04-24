#pragma once
// ddc.hpp: Digital Down-Converter for high-rate data acquisition
//
// A DDC translates a band of interest from an IF or RF frequency down to
// baseband by mixing with a complex local oscillator (NCO) and passing the
// I and Q streams through a decimation filter.
//
// Signal flow:
//
//     real input --> [mixer] --> I + jQ --> [decim_i] --> I_out + jQ_out
//                        ^                  [decim_q]
//                        |
//                   NCO (conj)
//
// The mixer multiplies the real input by the conjugate of the NCO output:
//
//     y[n] = x[n] * conj(lo[n]) = x[n] * cos(w n) + j * (-x[n] * sin(w n))
//
// After decimation the output is a complex baseband signal representing the
// spectral band centered on the NCO frequency, aliased down to DC.
//
// Template policy: the Decimator parameter selects between CIC, half-band,
// polyphase FIR, or any type that exposes one of:
//   - std::pair<bool, T> process(T)       (PolyphaseDecimator)
//   - std::pair<bool, T> process_decimate(T)  (HalfBandFilter)
//   - bool push(T); T output() const;     (CICDecimator)
//
// Two independent decimator instances run in lockstep on the I and Q
// streams. They are initialized to identical state and fed identical
// sample counts, so they always emit on the same cycle.
//
// Three-scalar parameterization:
//   CoeffScalar  - decimator coefficient precision (design time)
//   StateScalar  - NCO phase accumulator + decimator accumulator
//   SampleScalar - input samples + mixer and decimator output samples
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/acquisition/detail/decimator_step.hpp>
#include <sw/dsp/acquisition/nco.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/filter/fir/polyphase.hpp>

namespace sw::dsp {

// Digital Down-Converter: NCO -> mixer -> decimation filter.
//
// Construct with a center frequency, an input sample rate, and a prototype
// decimator. The prototype is copied twice (once for the I stream, once for
// the Q stream); the caller's instance is not mutated.
template <DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspField SampleScalar = StateScalar,
          class Decimator = PolyphaseDecimator<CoeffScalar, StateScalar, SampleScalar>>
class DDC {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	using complex_t     = complex_for_t<SampleScalar>;
	using nco_t         = NCO<StateScalar, SampleScalar>;
	using decimator_t   = Decimator;

	DDC(StateScalar center_frequency,
	    StateScalar sample_rate,
	    const Decimator& decimator)
		: nco_(center_frequency, sample_rate),
		  decim_i_(decimator),
		  decim_q_(decimator),
		  center_frequency_(center_frequency),
		  sample_rate_(sample_rate) {}

	// Retune the local oscillator.
	void set_center_frequency(StateScalar frequency) {
		nco_.set_frequency(frequency, sample_rate_);
		center_frequency_ = frequency;
	}

	StateScalar center_frequency() const { return center_frequency_; }
	StateScalar sample_rate()      const { return sample_rate_; }

	const nco_t& nco() const { return nco_; }
	nco_t&       nco()       { return nco_; }

	// Streaming: consume one real input sample. Returns {true, I+jQ} on the
	// emit cycle of the decimator, {false, 0} otherwise.
	std::pair<bool, complex_t> process(SampleScalar input) {
		complex_t lo = nco_.generate_sample();
		// Mix down: x * conj(lo) = (x*cos) + j*(-x*sin)
		SampleScalar i_in = static_cast<SampleScalar>(input * lo.real());
		SampleScalar q_in = static_cast<SampleScalar>(SampleScalar{} - input * lo.imag());

		auto [ri, yi] = detail::step_decimator(decim_i_, i_in);
		auto [rq, yq] = detail::step_decimator(decim_q_, q_in);

		if (ri != rq)
			throw std::logic_error("DDC: I and Q decimators out of lockstep");

		if (ri) return {true, complex_t(yi, yq)};
		return {false, complex_t{}};
	}

	// Block processing: consume a span of real input and return all decimated
	// complex samples produced during the block.
	mtl::vec::dense_vector<complex_t> process_block(std::span<const SampleScalar> input) {
		std::vector<complex_t> tmp;
		tmp.reserve(input.size());
		for (std::size_t n = 0; n < input.size(); ++n) {
			auto [ready, z] = process(input[n]);
			if (ready) tmp.push_back(z);
		}
		mtl::vec::dense_vector<complex_t> out(tmp.size());
		for (std::size_t i = 0; i < tmp.size(); ++i) out[i] = tmp[i];
		return out;
	}

	// Dense-vector overload.
	mtl::vec::dense_vector<complex_t> process_block(
			const mtl::vec::dense_vector<SampleScalar>& input) {
		std::vector<complex_t> tmp;
		tmp.reserve(input.size());
		for (std::size_t n = 0; n < input.size(); ++n) {
			auto [ready, z] = process(input[n]);
			if (ready) tmp.push_back(z);
		}
		mtl::vec::dense_vector<complex_t> out(tmp.size());
		for (std::size_t i = 0; i < tmp.size(); ++i) out[i] = tmp[i];
		return out;
	}

	void reset() {
		nco_.reset();
		decim_i_.reset();
		decim_q_.reset();
	}

private:
	nco_t       nco_;
	Decimator   decim_i_;
	Decimator   decim_q_;
	StateScalar center_frequency_;
	StateScalar sample_rate_;
};

} // namespace sw::dsp
