#pragma once
// chebyshev2.hpp: Chebyshev Type II (Inverse Chebyshev) IIR filter design
//
// Monotonic passband, equiripple stopband. Specified by order and
// stopband attenuation in dB. Has finite zeros (not all at infinity)
// which gives better stopband rejection.
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

// Chebyshev Type II analog lowpass prototype.
// Poles and zeros are derived from the stopband specification.
// Zeros lie on the imaginary axis (giving the notches in the stopband).
template <DspField T, int MaxOrder>
class ChebyshevIIAnalogPrototype {
public:
	void design(int num_poles, T stopband_db, PoleZeroLayout<T, MaxOrder>& layout) {
		layout.reset();
		layout.set_normal(T{}, T{1});

		const T eps = std::sqrt(T{1} / (std::exp(stopband_db * T{0.1} * ln10_v<T>) - T{1}));
		const T v0 = static_cast<T>(std::asinh(static_cast<double>(T{1} / eps))) / static_cast<T>(num_poles);
		const T sinh_v0 = T{-1} * static_cast<T>(std::sinh(static_cast<double>(v0)));
		const T cosh_v0 = static_cast<T>(std::cosh(static_cast<double>(v0)));
		const T fn = pi_v<T> / (T{2} * static_cast<T>(num_poles));

		int k = 1;
		for (int i = num_poles / 2; --i >= 0; k += 2) {
			T a = sinh_v0 * static_cast<T>(std::cos(static_cast<double>(static_cast<T>(k - num_poles) * fn)));
			T b = cosh_v0 * static_cast<T>(std::sin(static_cast<double>(static_cast<T>(k - num_poles) * fn)));
			T d2 = a * a + b * b;
			T im = T{1} / static_cast<T>(std::cos(static_cast<double>(static_cast<T>(k) * fn)));

			layout.add_conjugate_pairs(
				std::complex<T>(a / d2, b / d2),
				std::complex<T>(T{}, im));
		}

		if (num_poles & 1) {
			layout.add(std::complex<T>(T{1} / sinh_v0), s_infinity<T>());
		}
	}
};

// Chebyshev Type II filter designs: LP, HP, BP, BS
template <int MaxOrder,
          DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class ChebyshevIILowPass {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = (MaxOrder + 1) / 2;

	void setup(int order, double sample_rate, double cutoff_freq, double stopband_db) {
		ChebyshevIIAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, static_cast<CoeffScalar>(stopband_db), analog_);
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
class ChebyshevIIHighPass {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = (MaxOrder + 1) / 2;

	void setup(int order, double sample_rate, double cutoff_freq, double stopband_db) {
		ChebyshevIIAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, static_cast<CoeffScalar>(stopband_db), analog_);
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
class ChebyshevIIBandPass {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = MaxOrder;

	void setup(int order, double sample_rate, double center_freq, double width_freq, double stopband_db) {
		ChebyshevIIAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, static_cast<CoeffScalar>(stopband_db), analog_);
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
class ChebyshevIIBandStop {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = MaxOrder;

	void setup(int order, double sample_rate, double center_freq, double width_freq, double stopband_db) {
		ChebyshevIIAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, static_cast<CoeffScalar>(stopband_db), analog_);
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
