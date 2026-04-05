#pragma once
// chebyshev1.hpp: Chebyshev Type I IIR filter design
//
// Equiripple passband, monotonic stopband. Specified by order and
// passband ripple in dB. Steeper rolloff than Butterworth at the
// cost of passband ripple.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <complex>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/filter/layout/layout.hpp>
#include <sw/dsp/filter/layout/analog_prototype.hpp>
#include <sw/dsp/filter/biquad/cascade.hpp>
#include <sw/dsp/filter/transform/bilinear.hpp>
#include <sw/dsp/filter/transform/constantinides.hpp>

namespace sw::dsp::iir {

// Chebyshev Type I analog lowpass prototype.
// Poles lie on an ellipse in the s-plane. All zeros at infinity.
template <DspField T, int MaxOrder>
class ChebyshevIAnalogPrototype {
public:
	void design(int num_poles, T ripple_db, PoleZeroLayout<T, MaxOrder>& layout) {
		layout.reset();

		using std::sqrt;
		using std::exp;
		const T eps = sqrt(T{1} / exp(T{-1} * ripple_db * T{0.1} * ln10_v<T>) - T{1});
		const T v0 = static_cast<T>(std::asinh(static_cast<double>(T{1} / eps))) / static_cast<T>(num_poles);
		const T sinh_v0 = T{-1} * static_cast<T>(std::sinh(static_cast<double>(v0)));
		const T cosh_v0 = static_cast<T>(std::cosh(static_cast<double>(v0)));

		const T n2 = T{2} * static_cast<T>(num_poles);
		const int pairs = num_poles / 2;

		for (int i = 0; i < pairs; ++i) {
			const int k = 2 * i + 1 - num_poles;
			T a = sinh_v0 * static_cast<T>(std::cos(static_cast<double>(static_cast<T>(k) * pi_v<T> / n2)));
			T b = cosh_v0 * static_cast<T>(std::sin(static_cast<double>(static_cast<T>(k) * pi_v<T> / n2)));
			layout.add_conjugate_pairs(complex_for_t<T>(a, b), s_infinity<T>());
		}

		if (num_poles & 1) {
			layout.add(complex_for_t<T>(sinh_v0), s_infinity<T>());
			layout.set_normal(T{}, T{1});
		} else {
			layout.set_normal(T{}, static_cast<T>(std::pow(10.0, static_cast<double>(T{-1} * ripple_db) / 20.0)));
		}
	}
};

// Chebyshev Type I filter designs: LP, HP, BP, BS
template <int MaxOrder,
          DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class ChebyshevILowPass {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = (MaxOrder + 1) / 2;

	void setup(int order, double sample_rate, double cutoff_freq, double ripple_db) {
		ChebyshevIAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, static_cast<CoeffScalar>(ripple_db), analog_);
		LowPassTransform<CoeffScalar>(
			static_cast<CoeffScalar>(cutoff_freq / sample_rate), digital_, analog_);
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
class ChebyshevIHighPass {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = (MaxOrder + 1) / 2;

	void setup(int order, double sample_rate, double cutoff_freq, double ripple_db) {
		ChebyshevIAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, static_cast<CoeffScalar>(ripple_db), analog_);
		HighPassTransform<CoeffScalar>(
			static_cast<CoeffScalar>(cutoff_freq / sample_rate), digital_, analog_);
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
class ChebyshevIBandPass {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = MaxOrder;

	void setup(int order, double sample_rate, double center_freq, double width_freq, double ripple_db) {
		ChebyshevIAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, static_cast<CoeffScalar>(ripple_db), analog_);
		BandPassTransform<CoeffScalar>(
			static_cast<CoeffScalar>(center_freq / sample_rate),
			static_cast<CoeffScalar>(width_freq / sample_rate), digital_, analog_);
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
class ChebyshevIBandStop {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = MaxOrder;

	void setup(int order, double sample_rate, double center_freq, double width_freq, double ripple_db) {
		ChebyshevIAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, static_cast<CoeffScalar>(ripple_db), analog_);
		BandStopTransform<CoeffScalar>(
			static_cast<CoeffScalar>(center_freq / sample_rate),
			static_cast<CoeffScalar>(width_freq / sample_rate), digital_, analog_);
		cascade_.set_layout(digital_);
	}

	const Cascade<CoeffScalar, max_stages>& cascade() const { return cascade_; }

private:
	PoleZeroLayout<CoeffScalar, MaxOrder> analog_;
	PoleZeroLayout<CoeffScalar, MaxOrder * 2> digital_;
	Cascade<CoeffScalar, max_stages> cascade_;
};

} // namespace sw::dsp::iir
