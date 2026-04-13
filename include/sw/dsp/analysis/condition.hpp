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
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/types/biquad_coefficients.hpp>
#include <sw/dsp/filter/biquad/cascade.hpp>

namespace sw::dsp {

// Condition number of a biquad section's frequency response.
//
// Measures max|dH/H| / max|dc/c| where H is the frequency response
// and c are the coefficients. A large value means the response is
// numerically sensitive to coefficient perturbations.
//
// num_freqs: number of frequency points to sample in [0, 0.5]
template <DspField T>
	requires ConvertibleToDouble<T>
double biquad_condition_number(const BiquadCoefficients<T>& bq,
                               int num_freqs = 256) {
	double max_rel_change = 0.0;
	double epsilon = 1e-8;

	// Coefficients to perturb: a1, a2, b0, b1, b2
	auto perturb_and_measure = [&](auto make_perturbed) {
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

			double rel_change = std::abs(mag1 - mag0) / (mag0 * epsilon);
			if (rel_change > max_rel_change) max_rel_change = rel_change;
		}
	};

	// Perturb each coefficient
	perturb_and_measure([&]() {
		BiquadCoefficients<T> p = bq;
		p.a1 = static_cast<T>(static_cast<double>(bq.a1) + epsilon);
		return p;
	});
	perturb_and_measure([&]() {
		BiquadCoefficients<T> p = bq;
		p.a2 = static_cast<T>(static_cast<double>(bq.a2) + epsilon);
		return p;
	});
	perturb_and_measure([&]() {
		BiquadCoefficients<T> p = bq;
		p.b0 = static_cast<T>(static_cast<double>(bq.b0) + epsilon);
		return p;
	});
	perturb_and_measure([&]() {
		BiquadCoefficients<T> p = bq;
		p.b1 = static_cast<T>(static_cast<double>(bq.b1) + epsilon);
		return p;
	});
	perturb_and_measure([&]() {
		BiquadCoefficients<T> p = bq;
		p.b2 = static_cast<T>(static_cast<double>(bq.b2) + epsilon);
		return p;
	});

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
