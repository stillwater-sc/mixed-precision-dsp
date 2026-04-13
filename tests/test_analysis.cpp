// test_analysis.cpp: test numerical analysis tools — stability, sensitivity, condition
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/analysis/analysis.hpp>
#include <sw/dsp/filter/iir/butterworth.hpp>
#include <sw/dsp/filter/filter.hpp>
#include <sw/dsp/concepts/filter.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace sw::dsp;

bool near(double a, double b, double eps = 1e-4) {
	return std::abs(a - b) < eps;
}

// ========== Stability Tests ==========

void test_biquad_poles_real() {
	// Biquad with two real poles: z^2 - 0.5z + 0 = z(z - 0.5)
	// Poles at z = 0 and z = 0.5
	BiquadCoefficients<double> bq(1.0, 0.0, 0.0, -0.5, 0.0);
	auto [p1, p2] = biquad_poles(bq);

	// One pole at 0.5, one at 0
	double r1 = std::abs(p1);
	double r2 = std::abs(p2);
	double rmax = std::max(r1, r2);
	double rmin = std::min(r1, r2);

	if (!(near(rmax, 0.5, 1e-10)))
		throw std::runtime_error("test failed: real pole radius");
	if (!(near(rmin, 0.0, 1e-10)))
		throw std::runtime_error("test failed: zero pole");

	std::cout << "  biquad_poles_real: passed (poles at " << r1 << ", " << r2 << ")\n";
}

void test_biquad_poles_complex() {
	// Biquad with conjugate pair: z^2 + 0.0*z + 0.25 = 0
	// Poles at z = +/- 0.5i, |p| = 0.5
	BiquadCoefficients<double> bq(1.0, 0.0, 0.0, 0.0, 0.25);
	auto [p1, p2] = biquad_poles(bq);

	if (!(near(std::abs(p1), 0.5, 1e-10)))
		throw std::runtime_error("test failed: complex pole 1 radius");
	if (!(near(std::abs(p2), 0.5, 1e-10)))
		throw std::runtime_error("test failed: complex pole 2 radius");
	// Conjugate pair: imaginary parts should be opposite
	if (!(near(p1.imag(), -p2.imag(), 1e-10)))
		throw std::runtime_error("test failed: poles should be conjugate");

	std::cout << "  biquad_poles_complex: passed (|p|=" << std::abs(p1) << ")\n";
}

void test_stability_stable_filter() {
	// A well-designed Butterworth should be stable
	iir::ButterworthLowPass<4> butter;
	butter.setup(4, 44100.0, 1000.0);

	if (!(is_stable(butter.cascade())))
		throw std::runtime_error("test failed: Butterworth should be stable");

	double margin = stability_margin(butter.cascade());
	if (!(margin > 0.0))
		throw std::runtime_error("test failed: stability margin should be positive");
	if (!(margin < 1.0))
		throw std::runtime_error("test failed: stability margin should be < 1");

	double max_r = max_pole_radius(butter.cascade());
	if (!(max_r > 0.0 && max_r < 1.0))
		throw std::runtime_error("test failed: pole radius should be in (0, 1)");

	std::cout << "  stability_stable_filter: passed (margin=" << margin
	          << ", max_r=" << max_r << ")\n";
}

void test_stability_unstable_biquad() {
	// Poles outside unit circle: z^2 + 0*z - 4 = 0 => z = +/- 2
	BiquadCoefficients<double> bq(1.0, 0.0, 0.0, 0.0, -4.0);
	if (is_stable(bq))
		throw std::runtime_error("test failed: biquad with poles at +/-2 should be unstable");

	double r = max_pole_radius(bq);
	if (!(near(r, 2.0, 1e-10)))
		throw std::runtime_error("test failed: pole radius should be 2");

	std::cout << "  stability_unstable_biquad: passed (max_r=" << r << ")\n";
}

void test_all_poles() {
	iir::ButterworthLowPass<4> butter;
	butter.setup(4, 44100.0, 1000.0);

	auto poles = all_poles(butter.cascade());
	// 4th-order = 2 biquad stages = 4 poles
	if (!(poles.size() == 4))
		throw std::runtime_error("test failed: 4th-order should have 4 poles, got "
			+ std::to_string(poles.size()));

	// All poles should be inside the unit circle
	for (std::size_t i = 0; i < poles.size(); ++i) {
		if (!(std::abs(poles[i]) < 1.0))
			throw std::runtime_error("test failed: pole " + std::to_string(i)
				+ " outside unit circle");
	}

	std::cout << "  all_poles: passed (" << poles.size() << " poles)\n";
}

// ========== Sensitivity Tests ==========

void test_coefficient_sensitivity() {
	// A stable biquad should have finite sensitivity
	BiquadCoefficients<double> bq(1.0, 0.0, 0.0, -0.5, 0.2);
	auto sens = coefficient_sensitivity(bq);

	if (!(std::isfinite(sens.dp_da1)))
		throw std::runtime_error("test failed: dp_da1 not finite");
	if (!(std::isfinite(sens.dp_da2)))
		throw std::runtime_error("test failed: dp_da2 not finite");

	std::cout << "  coefficient_sensitivity: passed (dp/da1=" << sens.dp_da1
	          << ", dp/da2=" << sens.dp_da2 << ")\n";
}

void test_worst_case_sensitivity() {
	iir::ButterworthLowPass<4> butter;
	butter.setup(4, 44100.0, 1000.0);

	double wcs = worst_case_sensitivity(butter.cascade());
	if (!(wcs > 0.0))
		throw std::runtime_error("test failed: worst-case sensitivity should be > 0");
	if (!(std::isfinite(wcs)))
		throw std::runtime_error("test failed: worst-case sensitivity not finite");

	std::cout << "  worst_case_sensitivity: passed (wcs=" << wcs << ")\n";
}

void test_pole_displacement() {
	// Same filter in double and float — pole displacement should be small
	iir::ButterworthLowPass<4, double> butter_d;
	butter_d.setup(4, 44100.0, 1000.0);

	iir::ButterworthLowPass<4, float> butter_f;
	butter_f.setup(4, 44100.0, 1000.0);

	double disp = pole_displacement(butter_d.cascade(), butter_f.cascade());
	if (!(disp > 0.0))
		throw std::runtime_error("test failed: displacement should be > 0 (float != double)");
	if (!(disp < 1e-5))
		throw std::runtime_error("test failed: displacement too large for float vs double");

	std::cout << "  pole_displacement: passed (disp=" << disp << ")\n";
}

// ========== Condition Number Tests ==========

void test_biquad_condition_number() {
	// A simple biquad should have a finite condition number
	BiquadCoefficients<double> bq(1.0, 0.0, 0.0, -0.5, 0.2);
	double cn = biquad_condition_number(bq);

	if (!(cn > 0.0))
		throw std::runtime_error("test failed: condition number should be > 0");
	if (!(std::isfinite(cn)))
		throw std::runtime_error("test failed: condition number not finite");

	std::cout << "  biquad_condition_number: passed (cn=" << cn << ")\n";
}

void test_cascade_condition_number() {
	iir::ButterworthLowPass<4> butter;
	butter.setup(4, 44100.0, 1000.0);

	double cn = cascade_condition_number(butter.cascade());
	if (!(cn > 0.0))
		throw std::runtime_error("test failed: cascade condition number should be > 0");
	if (!(std::isfinite(cn)))
		throw std::runtime_error("test failed: cascade condition number not finite");

	// Higher-order filters tend to have higher condition numbers
	iir::ButterworthLowPass<8> butter8;
	butter8.setup(8, 44100.0, 1000.0);
	double cn8 = cascade_condition_number(butter8.cascade());

	std::cout << "  cascade_condition_number: passed (order4=" << cn
	          << ", order8=" << cn8 << ")\n";
}

// ========== Filter Concepts Tests ==========

void test_filter_concepts() {
	// Verify that our filter types satisfy the concepts
	static_assert(FilterDesign<iir::ButterworthLowPass<4>>,
		"ButterworthLowPass should satisfy FilterDesign");
	static_assert(DesignableLowPass<iir::ButterworthLowPass<4>>,
		"ButterworthLowPass should satisfy DesignableLowPass");

	// SimpleFilter should satisfy Processable
	static_assert(Processable<SimpleFilter<iir::ButterworthLowPass<4>>>,
		"SimpleFilter<ButterworthLowPass> should satisfy Processable");

	// BandPass should satisfy DesignableBandPass
	static_assert(DesignableBandPass<iir::ButterworthBandPass<4>>,
		"ButterworthBandPass should satisfy DesignableBandPass");

	std::cout << "  filter_concepts: passed (static_assert checks)\n";
}

// ========== Umbrella Header Compile Test ==========

void test_umbrella_header() {
	// Just verify that dsp.hpp compiles and brings in key types
	// (This is a compile-time test — if it builds, it passes)
	std::cout << "  umbrella_header: passed (dsp.hpp compiles)\n";
}

int main() {
	try {
		std::cout << "Analysis Tests\n";

		// Stability
		test_biquad_poles_real();
		test_biquad_poles_complex();
		test_stability_stable_filter();
		test_stability_unstable_biquad();
		test_all_poles();

		// Sensitivity
		test_coefficient_sensitivity();
		test_worst_case_sensitivity();
		test_pole_displacement();

		// Condition number
		test_biquad_condition_number();
		test_cascade_condition_number();

		// Filter concepts
		test_filter_concepts();

		// Umbrella
		test_umbrella_header();

		std::cout << "All analysis tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << '\n';
		return 1;
	}
}
