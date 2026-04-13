#pragma once
// stability.hpp: pole radius and stability margin analysis
//
// For a biquad section H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2),
// the poles are roots of z^2 + a1*z + a2 = 0.
//
// A discrete-time system is stable iff all poles lie strictly inside
// the unit circle (|p| < 1). The stability margin is 1 - max|p|.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <complex>
#include <cstddef>
#include <utility>
#include <vector>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/types/biquad_coefficients.hpp>
#include <sw/dsp/filter/biquad/cascade.hpp>

namespace sw::dsp {

// Compute the two poles of a biquad section.
// Denominator: z^2 + a1*z + a2 = 0
// Returns a pair of complex poles (may be conjugate pair or two real).
template <DspField T>
	requires ConvertibleToDouble<T>
std::pair<std::complex<double>, std::complex<double>>
biquad_poles(const BiquadCoefficients<T>& bq) {
	double a1 = static_cast<double>(bq.a1);
	double a2 = static_cast<double>(bq.a2);

	// z^2 + a1*z + a2 = 0  =>  z = (-a1 +/- sqrt(a1^2 - 4*a2)) / 2
	double disc = a1 * a1 - 4.0 * a2;
	if (disc >= 0.0) {
		double sq = std::sqrt(disc);
		return { std::complex<double>((-a1 + sq) / 2.0, 0.0),
		         std::complex<double>((-a1 - sq) / 2.0, 0.0) };
	} else {
		double sq = std::sqrt(-disc);
		return { std::complex<double>(-a1 / 2.0,  sq / 2.0),
		         std::complex<double>(-a1 / 2.0, -sq / 2.0) };
	}
}

// Maximum pole radius of a single biquad section.
template <DspField T>
	requires ConvertibleToDouble<T>
double max_pole_radius(const BiquadCoefficients<T>& bq) {
	auto [p1, p2] = biquad_poles(bq);
	return std::max(std::abs(p1), std::abs(p2));
}

// Check if a single biquad section is stable (all poles inside unit circle).
template <DspField T>
	requires ConvertibleToDouble<T>
bool is_stable(const BiquadCoefficients<T>& bq) {
	return max_pole_radius(bq) < 1.0;
}

// Maximum pole radius across all stages of a cascade.
template <DspField T, int MaxStages>
	requires ConvertibleToDouble<T>
double max_pole_radius(const Cascade<T, MaxStages>& cascade) {
	double max_r = 0.0;
	for (int i = 0; i < cascade.num_stages(); ++i) {
		double r = max_pole_radius(cascade.stage(i));
		if (r > max_r) max_r = r;
	}
	return max_r;
}

// Check if an entire cascade is stable.
template <DspField T, int MaxStages>
	requires ConvertibleToDouble<T>
bool is_stable(const Cascade<T, MaxStages>& cascade) {
	for (int i = 0; i < cascade.num_stages(); ++i) {
		if (!is_stable(cascade.stage(i))) return false;
	}
	return true;
}

// Stability margin: 1 - max_pole_radius.
// Positive means stable, zero means marginally stable, negative means unstable.
template <DspField T, int MaxStages>
	requires ConvertibleToDouble<T>
double stability_margin(const Cascade<T, MaxStages>& cascade) {
	return 1.0 - max_pole_radius(cascade);
}

// Collect all poles from a cascade into a vector of complex values.
template <DspField T, int MaxStages>
	requires ConvertibleToDouble<T>
std::vector<std::complex<double>> all_poles(const Cascade<T, MaxStages>& cascade) {
	std::vector<std::complex<double>> poles;
	poles.reserve(2 * static_cast<std::size_t>(cascade.num_stages()));
	for (int i = 0; i < cascade.num_stages(); ++i) {
		auto [p1, p2] = biquad_poles(cascade.stage(i));
		poles.push_back(p1);
		// Skip degenerate second pole for first-order sections (a2 == 0)
		if (static_cast<double>(cascade.stage(i).a2) != 0.0) {
			poles.push_back(p2);
		}
	}
	return poles;
}

} // namespace sw::dsp
