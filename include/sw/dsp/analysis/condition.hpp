#pragma once
// condition.hpp: condition number estimation for biquad coefficients
//
// The condition number of a biquad section measures how sensitive
// the frequency response is to small perturbations in the coefficients.
// A high condition number means the filter is numerically fragile —
// small coefficient errors cause large response changes.
//
// We estimate the condition number as the ratio of the maximum to
// minimum frequency response magnitude perturbation per unit
// coefficient change, sampled across the frequency band.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/types/biquad_coefficients.hpp>
#include <sw/dsp/filter/biquad/cascade.hpp>

namespace sw::dsp {

namespace detail {

// Compute a perturbation delta that is guaranteed to survive the
// round-trip through type T. For narrow types where 1e-8 would be
// absorbed, we use a relative epsilon based on the coefficient
// magnitude. The perturbation is computed in double, applied to
// the double representation, then cast back to T.
template <DspField T>
	requires ConvertibleToDouble<T>
double representable_delta(T coeff) {
	double val = static_cast<double>(coeff);
	// Use a relative perturbation: max(|val| * 1e-7, 1e-10)
	// Then verify the round-trip actually changes the value.
	double delta = std::max(std::abs(val) * 1e-7, 1e-10);
	T perturbed = static_cast<T>(val + delta);
	double actual_delta = static_cast<double>(perturbed) - val;
	// If the perturbation was absorbed (narrow type), double delta
	while (actual_delta == 0.0 && delta < 1.0) {
		delta *= 2.0;
		perturbed = static_cast<T>(val + delta);
		actual_delta = static_cast<double>(perturbed) - val;
	}
	return (actual_delta != 0.0) ? actual_delta : delta;
}

} // namespace detail

// Condition number of a biquad section's frequency response.
//
// Measures max|dH/H| / |dc| where H is the frequency response
// and c are the coefficients. A large value means the response is
// numerically sensitive to coefficient perturbations.
//
// The perturbation for each coefficient is chosen to be representable
// in type T, avoiding silent no-ops for narrow arithmetic types.
//
// num_freqs: number of frequency points to sample in [0, 0.5]
template <DspField T>
	requires ConvertibleToDouble<T>
double biquad_condition_number(const BiquadCoefficients<T>& bq,
                               int num_freqs = 256) {
	if (num_freqs <= 0)
		throw std::invalid_argument("biquad_condition_number: num_freqs must be > 0");

	double max_rel_change = 0.0;

	// Perturb a coefficient and measure the maximum relative frequency
	// response change across the band.
	auto perturb_and_measure = [&](auto make_perturbed, double delta) {
		if (delta <= 0.0) return;
		for (int k = 0; k < num_freqs; ++k) {
			double f = 0.5 * k / num_freqs;
			auto H0 = bq.response(f);
			double mag0 = std::abs(std::complex<double>(
				static_cast<double>(H0.real()),
				static_cast<double>(H0.imag())));
			if (mag0 < 1e-20) continue;

			auto bq_pert = make_perturbed();
			auto H1 = bq_pert.response(f);
			double mag1 = std::abs(std::complex<double>(
				static_cast<double>(H1.real()),
				static_cast<double>(H1.imag())));

			double rel_change = std::abs(mag1 - mag0) / (mag0 * delta);
			if (rel_change > max_rel_change) max_rel_change = rel_change;
		}
	};

	// Perturb each coefficient with a representable delta
	{
		double da1 = detail::representable_delta(bq.a1);
		perturb_and_measure([&]() {
			BiquadCoefficients<T> p = bq;
			p.a1 = static_cast<T>(static_cast<double>(bq.a1) + da1);
			return p;
		}, da1);
	}
	{
		double da2 = detail::representable_delta(bq.a2);
		perturb_and_measure([&]() {
			BiquadCoefficients<T> p = bq;
			p.a2 = static_cast<T>(static_cast<double>(bq.a2) + da2);
			return p;
		}, da2);
	}
	{
		double db0 = detail::representable_delta(bq.b0);
		perturb_and_measure([&]() {
			BiquadCoefficients<T> p = bq;
			p.b0 = static_cast<T>(static_cast<double>(bq.b0) + db0);
			return p;
		}, db0);
	}
	{
		double db1 = detail::representable_delta(bq.b1);
		perturb_and_measure([&]() {
			BiquadCoefficients<T> p = bq;
			p.b1 = static_cast<T>(static_cast<double>(bq.b1) + db1);
			return p;
		}, db1);
	}
	{
		double db2 = detail::representable_delta(bq.b2);
		perturb_and_measure([&]() {
			BiquadCoefficients<T> p = bq;
			p.b2 = static_cast<T>(static_cast<double>(bq.b2) + db2);
			return p;
		}, db2);
	}

	return max_rel_change;
}

// Worst-case condition number across all stages of a cascade.
template <DspField T, int MaxStages>
	requires ConvertibleToDouble<T>
double cascade_condition_number(const Cascade<T, MaxStages>& cascade,
                                int num_freqs = 256) {
	double worst = 0.0;
	for (int i = 0; i < cascade.num_stages(); ++i) {
		double cn = biquad_condition_number(cascade.stage(i), num_freqs);
		if (cn > worst) worst = cn;
	}
	return worst;
}

} // namespace sw::dsp
