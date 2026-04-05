#pragma once
// bessel.hpp: Bessel (Thomson) IIR filter design
//
// Maximally flat group delay. Poles are found by computing the reverse
// Bessel polynomial coefficients and finding roots via Laguerre's method.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <complex>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/math/root_finder.hpp>
#include <sw/dsp/filter/layout/layout.hpp>
#include <sw/dsp/filter/layout/analog_prototype.hpp>
#include <sw/dsp/filter/biquad/cascade.hpp>
#include <sw/dsp/filter/transform/bilinear.hpp>
#include <sw/dsp/filter/transform/constantinides.hpp>

namespace sw::dsp::iir {

namespace detail {

// n! (factorial)
inline double factorial(int n) {
	double y = 1.0;
	for (int i = 2; i <= n; ++i) y *= i;
	return y;
}

// k-th coefficient of the reverse Bessel polynomial of degree n
// reversebessel(k, n) = (2n - k)! / (k! * (n-k)! * 2^(n-k))
inline double reverse_bessel_coef(int k, int n) {
	return factorial(2 * n - k) /
	       (factorial(n - k) * factorial(k) * std::pow(2.0, n - k));
}

} // namespace detail

// Bessel analog lowpass prototype.
// Uses Laguerre root finder on the reverse Bessel polynomial.
template <DspField T, int MaxOrder>
class BesselAnalogPrototype {
public:
	void design(int num_poles, PoleZeroLayout<T, MaxOrder>& layout) {
		layout.reset();
		layout.set_normal(T{}, T{1});

		// Set up reverse Bessel polynomial coefficients
		RootFinder<T, MaxOrder> solver;
		for (int i = 0; i <= num_poles; ++i) {
			solver.coef(i) = std::complex<T>(
				static_cast<T>(detail::reverse_bessel_coef(i, num_poles)));
		}
		solver.solve(num_poles);

		const int pairs = num_poles / 2;
		for (int i = 0; i < pairs; ++i) {
			layout.add_conjugate_pairs(solver.root(i), s_infinity<T>());
		}

		if (num_poles & 1) {
			layout.add(
				std::complex<T>(solver.root(pairs).real(), T{}),
				s_infinity<T>());
		}
	}
};

// Bessel filter designs: LP, HP, BP, BS
template <int MaxOrder,
          DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class BesselLowPass {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = (MaxOrder + 1) / 2;

	void setup(int order, double sample_rate, double cutoff_freq) {
		BesselAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, analog_);
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
class BesselHighPass {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = (MaxOrder + 1) / 2;

	void setup(int order, double sample_rate, double cutoff_freq) {
		BesselAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, analog_);
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
class BesselBandPass {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = MaxOrder;

	void setup(int order, double sample_rate, double center_freq, double width_freq) {
		BesselAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, analog_);
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
class BesselBandStop {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = MaxOrder;

	void setup(int order, double sample_rate, double center_freq, double width_freq) {
		BesselAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, analog_);
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
