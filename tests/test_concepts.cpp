// test_concepts.cpp: verify that DSP concepts accept expected types
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/concepts/signal.hpp>
#include <sw/dsp/types/complex_pair.hpp>
#include <sw/dsp/types/pole_zero_pair.hpp>
#include <sw/dsp/types/biquad_coefficients.hpp>
#include <sw/dsp/types/filter_kind.hpp>
#include <sw/dsp/types/filter_spec.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/math/denormal.hpp>
#include <sw/dsp/math/quadratic.hpp>
#include <sw/dsp/math/polynomial.hpp>

#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

using namespace sw::dsp;

// Compile-time concept checks for built-in types
static_assert(DspScalar<float>);
static_assert(DspScalar<double>);
static_assert(DspScalar<long double>);
static_assert(DspField<float>);
static_assert(DspField<double>);
static_assert(DspOrderedField<float>);
static_assert(DspOrderedField<double>);

// std::complex should satisfy DspScalar and DspField but not DspOrderedField
static_assert(DspScalar<std::complex<double>>);
static_assert(DspField<std::complex<double>>);
static_assert(!DspOrderedField<std::complex<double>>);

// int should satisfy DspScalar but not DspField (integer division truncates)
static_assert(DspScalar<int>);

// Signal container checks
static_assert(SignalContainer<std::vector<double>>);
static_assert(MutableSignalContainer<std::vector<double>>);
static_assert(ContiguousSignalContainer<std::vector<double>>);

int main() {
	// Test ComplexPair
	{
		ComplexPair<double> cp(std::complex<double>(1.0, 2.0),
		                       std::complex<double>(1.0, -2.0));
		assert(cp.is_conjugate());
		assert(!cp.is_real());
		assert(cp.is_matched_pair());
		assert(!cp.is_nan());

		ComplexPair<double> real_pair(std::complex<double>(1.0, 0.0),
		                              std::complex<double>(2.0, 0.0));
		assert(real_pair.is_real());
		assert(real_pair.is_matched_pair());
	}

	// Test PoleZeroPair
	{
		PoleZeroPair<double> pz(std::complex<double>(-0.5, 0.0),
		                        std::complex<double>(-1.0, 0.0));
		assert(pz.is_single_pole());

		PoleZeroPair<double> pz2(std::complex<double>(-0.5, 0.5),
		                         std::complex<double>(-1.0, 0.0),
		                         std::complex<double>(-0.5, -0.5),
		                         std::complex<double>(-1.0, 0.0));
		assert(!pz2.is_single_pole());
	}

	// Test BiquadCoefficients
	{
		BiquadCoefficients<double> bq;
		bq.set_identity();
		assert(bq.b0 == 1.0);
		assert(bq.b1 == 0.0);
		assert(bq.a1 == 0.0);

		// Response of identity filter should be 1.0 at all frequencies
		auto r = bq.response(0.25);
		assert(std::abs(std::abs(r) - 1.0) < 1e-10);
	}

	// Test FilterKind string conversion
	{
		assert(to_string(FilterKind::low_pass) == "Low Pass");
		assert(to_string(FilterKind::band_stop) == "Band Stop");
	}

	// Test FilterSpec
	{
		FilterSpec spec;
		spec.sample_rate = 48000.0;
		spec.cutoff_frequency = 1000.0;
		assert(std::abs(spec.normalized_cutoff() - 1000.0 / 48000.0) < 1e-15);
	}

	// Test constants
	{
		assert(std::abs(pi - 3.14159265358979323846) < 1e-15);
		assert(std::abs(two_pi - 2.0 * pi) < 1e-15);
		assert(std::abs(pi_v<float> - 3.14159265f) < 1e-6f);
	}

	// Test DenormalPrevention
	{
		DenormalPrevention<double> dp;
		double v1 = dp.ac();
		double v2 = dp.ac();
		assert(v1 == -v2);  // alternating sign
		assert(v1 != 0.0);  // non-zero for IEEE types
	}

	// Test quadratic solver
	{
		// x^2 - 5x + 6 = 0 -> roots at 2 and 3
		auto [r1, r2] = solve_quadratic(1.0, -5.0, 6.0);
		assert(std::abs(r1.real() - 3.0) < 1e-10);
		assert(std::abs(r2.real() - 2.0) < 1e-10);
		assert(std::abs(r1.imag()) < 1e-10);
		assert(std::abs(r2.imag()) < 1e-10);

		// x^2 + 1 = 0 -> roots at +i and -i
		auto [c1, c2] = solve_quadratic(1.0, 0.0, 1.0);
		assert(std::abs(c1.real()) < 1e-10);
		assert(std::abs(std::abs(c1.imag()) - 1.0) < 1e-10);
	}

	// Test polynomial evaluation
	{
		// p(x) = 1 + 2x + 3x^2, p(2) = 1 + 4 + 12 = 17
		std::vector<double> coeffs = {1.0, 2.0, 3.0};
		assert(std::abs(evaluate_polynomial(coeffs, 2.0) - 17.0) < 1e-10);

		// Polynomial multiplication: (1 + x)(1 + 2x) = 1 + 3x + 2x^2
		std::vector<double> a = {1.0, 1.0};
		std::vector<double> b = {1.0, 2.0};
		auto product = multiply_polynomials(a, b);
		assert(product.size() == 3);
		assert(std::abs(product[0] - 1.0) < 1e-10);
		assert(std::abs(product[1] - 3.0) < 1e-10);
		assert(std::abs(product[2] - 2.0) < 1e-10);
	}

	std::cout << "All Phase 1 tests passed.\n";
	return 0;
}
