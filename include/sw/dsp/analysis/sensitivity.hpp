#pragma once
// sensitivity.hpp: coefficient sensitivity analysis
//
// Measures how pole positions shift when biquad coefficients are
// perturbed. This quantifies the impact of coefficient quantization
// on filter behavior — critical for mixed-precision design.
//
// For a biquad z^2 + a1*z + a2 = 0, the partial derivatives of
// the poles with respect to a1 and a2 are computed analytically:
//
//   dp/da1 = -1 / (2*p - (-a1))  [from implicit differentiation]
//          = -1 / (p1 - p2)       [for the first pole]
//
//   dp/da2 = -1 / (2*p + a1)
//          = p / (p1 - p2)        [since 2*p + a1 = +/- sqrt(disc)]
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <complex>
#include <cstddef>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/types/biquad_coefficients.hpp>
#include <sw/dsp/filter/biquad/cascade.hpp>
#include <sw/dsp/analysis/stability.hpp>

namespace sw::dsp {

// Coefficient sensitivity of a biquad section.
//
// Measures how much the pole radius changes per unit change in each
// coefficient. Computed by finite differences (perturb each coefficient
// by epsilon and measure the pole radius shift).
//
// Returns a struct with sensitivities for a1 and a2.
struct BiquadSensitivity {
	double dp_da1;  // d(max_pole_radius) / d(a1)
	double dp_da2;  // d(max_pole_radius) / d(a2)
};

template <DspField T>
	requires ConvertibleToDouble<T>
BiquadSensitivity coefficient_sensitivity(const BiquadCoefficients<T>& bq,
                                          double epsilon = 1e-8) {
	double r0 = max_pole_radius(bq);

	// Perturb a1
	BiquadCoefficients<T> bq_a1 = bq;
	bq_a1.a1 = static_cast<T>(static_cast<double>(bq.a1) + epsilon);
	double dp_da1 = (max_pole_radius(bq_a1) - r0) / epsilon;

	// Perturb a2
	BiquadCoefficients<T> bq_a2 = bq;
	bq_a2.a2 = static_cast<T>(static_cast<double>(bq.a2) + epsilon);
	double dp_da2 = (max_pole_radius(bq_a2) - r0) / epsilon;

	return { dp_da1, dp_da2 };
}

// Worst-case coefficient sensitivity across all stages of a cascade.
// Returns the maximum sensitivity magnitude found in any stage.
template <DspField T, int MaxStages>
	requires ConvertibleToDouble<T>
double worst_case_sensitivity(const Cascade<T, MaxStages>& cascade,
                              double epsilon = 1e-8) {
	double worst = 0.0;
	for (int i = 0; i < cascade.num_stages(); ++i) {
		auto sens = coefficient_sensitivity(cascade.stage(i), epsilon);
		double s = std::max(std::abs(sens.dp_da1), std::abs(sens.dp_da2));
		if (s > worst) worst = s;
	}
	return worst;
}

// Pole displacement: measure how much poles move when coefficients
// are quantized from type T to type Q. Returns the maximum pole
// position change (in complex magnitude) across all stages.
template <DspField T, DspField Q, int MaxStages>
	requires (ConvertibleToDouble<T> && ConvertibleToDouble<Q>)
double pole_displacement(const Cascade<T, MaxStages>& original,
                         const Cascade<Q, MaxStages>& quantized) {
	double max_disp = 0.0;
	int n = std::min(original.num_stages(), quantized.num_stages());
	for (int i = 0; i < n; ++i) {
		auto [p1_orig, p2_orig] = biquad_poles(original.stage(i));
		auto [p1_quant, p2_quant] = biquad_poles(quantized.stage(i));
		double d1 = std::abs(p1_orig - p1_quant);
		double d2 = std::abs(p2_orig - p2_quant);
		double d = std::max(d1, d2);
		if (d > max_disp) max_disp = d;
	}
	return max_disp;
}

} // namespace sw::dsp
