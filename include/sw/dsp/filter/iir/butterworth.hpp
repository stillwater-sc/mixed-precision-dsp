#pragma once
// butterworth.hpp: Butterworth IIR filter design
//
// Maximally flat magnitude response in the passband. Poles are equally
// spaced on a circle in the s-plane. All zeros at infinity (all-pole).
//
// Provides LowPass, HighPass, BandPass, BandStop, and shelf variants.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <complex>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/types/biquad_coefficients.hpp>
#include <sw/dsp/filter/layout/layout.hpp>
#include <sw/dsp/filter/layout/analog_prototype.hpp>
#include <sw/dsp/filter/biquad/cascade.hpp>
#include <sw/dsp/filter/biquad/state.hpp>
#include <sw/dsp/filter/transform/bilinear.hpp>
#include <sw/dsp/filter/transform/constantinides.hpp>

namespace sw::dsp::iir {

// Butterworth analog lowpass prototype.
//
// Places num_poles poles equally spaced on the left half of the
// unit circle in the s-plane. All zeros at s = infinity.
// Normal gain = 1 at w = 0 (DC).
template <DspField T, int MaxOrder>
class ButterworthAnalogPrototype {
public:
	void design(int num_poles, PoleZeroLayout<T, MaxOrder>& layout) {
		layout.reset();
		layout.set_normal(T{}, T{1});

		const T n2 = T{2} * static_cast<T>(num_poles);
		const int pairs = num_poles / 2;

		for (int i = 0; i < pairs; ++i) {
			T theta = half_pi_v<T> + static_cast<T>(2 * i + 1) * pi_v<T> / n2;
			using std::polar;  // ADL for Universal types
			auto pole = polar(T{1}, theta);
			layout.add_conjugate_pairs(pole, s_infinity<T>());
		}

		if (num_poles & 1) {
			layout.add(complex_for_t<T>(T{-1}), s_infinity<T>());
		}
	}
};

// Butterworth analog low shelf prototype.
//
// Distributes poles and zeros to achieve a shelf response with
// the specified gain in dB. Based on Orfanidis' parametric EQ design.
template <DspField T, int MaxOrder>
class ButterworthAnalogLowShelf {
public:
	void design(int num_poles, T gain_db, PoleZeroLayout<T, MaxOrder>& layout) {
		layout.reset();
		layout.set_normal(pi_v<T>, T{1});

		const T n2 = static_cast<T>(num_poles) * T{2};
		const T g = std::pow(std::pow(T{10}, gain_db / T{20}), T{1} / n2);
		const T gp = T{-1} / g;
		const T gz = T{-1} * g;

		const int pairs = num_poles / 2;
		for (int i = 1; i <= pairs; ++i) {
			T theta = pi_v<T> * (T{0.5} - static_cast<T>(2 * i - 1) / n2);
			using std::polar;  // ADL for Universal types
			layout.add_conjugate_pairs(polar(gp, theta), polar(gz, theta));
		}

		if (num_poles & 1) {
			layout.add(complex_for_t<T>(gp), complex_for_t<T>(gz));
		}
	}
};

// ============================================================================
// Butterworth filter designs: the complete pipeline from analog prototype
// through frequency transformation and bilinear transform to cascade.
// ============================================================================

template <int MaxOrder,
          DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class ButterworthLowPass {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = (MaxOrder + 1) / 2;

	void setup(int order, double sample_rate, double cutoff_freq) {
		ButterworthAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, analog_);

		LowPassTransform<CoeffScalar>(
			static_cast<CoeffScalar>(cutoff_freq / sample_rate),
			digital_, analog_);

		cascade_.set_layout(digital_);
	}

	const Cascade<CoeffScalar, max_stages>& cascade() const { return cascade_; }

private:
	PoleZeroLayout<CoeffScalar, MaxOrder> analog_;
	PoleZeroLayout<CoeffScalar, MaxOrder> digital_;
	Cascade<CoeffScalar, max_stages> cascade_;
};

template <int MaxOrder,
          DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class ButterworthHighPass {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = (MaxOrder + 1) / 2;

	void setup(int order, double sample_rate, double cutoff_freq) {
		ButterworthAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, analog_);

		HighPassTransform<CoeffScalar>(
			static_cast<CoeffScalar>(cutoff_freq / sample_rate),
			digital_, analog_);

		cascade_.set_layout(digital_);
	}

	const Cascade<CoeffScalar, max_stages>& cascade() const { return cascade_; }

private:
	PoleZeroLayout<CoeffScalar, MaxOrder> analog_;
	PoleZeroLayout<CoeffScalar, MaxOrder> digital_;
	Cascade<CoeffScalar, max_stages> cascade_;
};

template <int MaxOrder,
          DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class ButterworthBandPass {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = MaxOrder;  // order doubles for bandpass

	void setup(int order, double sample_rate, double center_freq, double width_freq) {
		ButterworthAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, analog_);

		BandPassTransform<CoeffScalar>(
			static_cast<CoeffScalar>(center_freq / sample_rate),
			static_cast<CoeffScalar>(width_freq / sample_rate),
			digital_, analog_);

		cascade_.set_layout(digital_);
	}

	const Cascade<CoeffScalar, max_stages>& cascade() const { return cascade_; }

private:
	PoleZeroLayout<CoeffScalar, MaxOrder> analog_;
	PoleZeroLayout<CoeffScalar, MaxOrder * 2> digital_;
	Cascade<CoeffScalar, max_stages> cascade_;
};

template <int MaxOrder,
          DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class ButterworthBandStop {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = MaxOrder;

	void setup(int order, double sample_rate, double center_freq, double width_freq) {
		ButterworthAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, analog_);

		BandStopTransform<CoeffScalar>(
			static_cast<CoeffScalar>(center_freq / sample_rate),
			static_cast<CoeffScalar>(width_freq / sample_rate),
			digital_, analog_);

		cascade_.set_layout(digital_);
	}

	const Cascade<CoeffScalar, max_stages>& cascade() const { return cascade_; }

private:
	PoleZeroLayout<CoeffScalar, MaxOrder> analog_;
	PoleZeroLayout<CoeffScalar, MaxOrder * 2> digital_;
	Cascade<CoeffScalar, max_stages> cascade_;
};

template <int MaxOrder,
          DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class ButterworthLowShelf {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = (MaxOrder + 1) / 2;

	void setup(int order, double sample_rate, double cutoff_freq, double gain_db) {
		ButterworthAnalogLowShelf<CoeffScalar, MaxOrder> proto;
		proto.design(order, static_cast<CoeffScalar>(gain_db), analog_);

		LowPassTransform<CoeffScalar>(
			static_cast<CoeffScalar>(cutoff_freq / sample_rate),
			digital_, analog_);

		cascade_.set_layout(digital_);
	}

	const Cascade<CoeffScalar, max_stages>& cascade() const { return cascade_; }

private:
	PoleZeroLayout<CoeffScalar, MaxOrder> analog_;
	PoleZeroLayout<CoeffScalar, MaxOrder> digital_;
	Cascade<CoeffScalar, max_stages> cascade_;
};

template <int MaxOrder,
          DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class ButterworthHighShelf {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = (MaxOrder + 1) / 2;

	void setup(int order, double sample_rate, double cutoff_freq, double gain_db) {
		ButterworthAnalogLowShelf<CoeffScalar, MaxOrder> proto;
		proto.design(order, static_cast<CoeffScalar>(gain_db), analog_);

		HighPassTransform<CoeffScalar>(
			static_cast<CoeffScalar>(cutoff_freq / sample_rate),
			digital_, analog_);

		cascade_.set_layout(digital_);
	}

	const Cascade<CoeffScalar, max_stages>& cascade() const { return cascade_; }

private:
	PoleZeroLayout<CoeffScalar, MaxOrder> analog_;
	PoleZeroLayout<CoeffScalar, MaxOrder> digital_;
	Cascade<CoeffScalar, max_stages> cascade_;
};

} // namespace sw::dsp::iir
