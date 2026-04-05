// test_biquad.cpp: test biquad engine, state forms, cascade, and layout
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/filter/biquad/biquad.hpp>
#include <sw/dsp/filter/biquad/state.hpp>
#include <sw/dsp/filter/biquad/cascade.hpp>
#include <sw/dsp/filter/biquad/smooth.hpp>
#include <sw/dsp/filter/layout/layout.hpp>
#include <sw/dsp/math/constants.hpp>

#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

using namespace sw::dsp;

constexpr double tolerance = 1e-8;

bool near(double a, double b, double eps = tolerance) {
	return std::abs(a - b) < eps;
}

void test_biquad_coefficients() {
	// Identity filter: H(z) = 1
	BiquadCoefficients<double> id;
	id.set_identity();
	assert(id.b0 == 1.0);
	assert(id.b1 == 0.0);
	assert(id.b2 == 0.0);
	assert(id.a1 == 0.0);
	assert(id.a2 == 0.0);

	// Response of identity is 1 at all frequencies
	for (double f = 0.0; f <= 0.5; f += 0.1) {
		auto r = id.response(f);
		assert(near(std::abs(r), 1.0));
	}

	// Scale test
	BiquadCoefficients<double> bq(2.0, 1.0, 0.5, -0.3, 0.1);
	bq.apply_scale(0.5);
	assert(near(bq.b0, 1.0));
	assert(near(bq.b1, 0.5));
	assert(near(bq.b2, 0.25));
	assert(near(bq.a1, -0.3));  // denominator unchanged
	assert(near(bq.a2, 0.1));

	std::cout << "  biquad_coefficients: passed\n";
}

void test_biquad_pole_state() {
	// Create a simple lowpass biquad and verify pole/zero extraction
	// H(z) = (0.25 + 0.5*z^-1 + 0.25*z^-2) / (1 - 0.5*z^-1 + 0.1*z^-2)
	BiquadCoefficients<double> bq(0.25, 0.5, 0.25, -0.5, 0.1);

	BiquadPoleState<double> ps(bq);

	// Verify gain
	assert(near(ps.gain, 0.25));

	// Poles should satisfy z^2 - 0.5z + 0.1 = 0
	// z = (0.5 +/- sqrt(0.25 - 0.4)) / 2 = (0.5 +/- sqrt(-0.15)) / 2
	// Complex conjugate poles
	assert(ps.poles.is_conjugate() || ps.poles.is_real());

	std::cout << "  biquad_pole_state: passed\n";
}

void test_direct_form_I() {
	// Test impulse response of identity filter
	BiquadCoefficients<double> id;
	id.set_identity();
	DirectFormI<double> state;

	double y0 = state.process<double, double>(1.0, id);
	double y1 = state.process<double, double>(0.0, id);
	double y2 = state.process<double, double>(0.0, id);

	assert(near(y0, 1.0));
	assert(near(y1, 0.0));
	assert(near(y2, 0.0));

	// Test simple first-order IIR: y[n] = x[n] + 0.5*y[n-1]
	// H(z) = 1 / (1 - 0.5*z^-1)
	// Coefficients: b0=1, b1=0, b2=0, a1=-0.5, a2=0
	BiquadCoefficients<double> iir(1.0, 0.0, 0.0, -0.5, 0.0);
	state.reset();

	double h0 = state.process<double, double>(1.0, iir);  // 1.0
	double h1 = state.process<double, double>(0.0, iir);  // 0.5
	double h2 = state.process<double, double>(0.0, iir);  // 0.25
	double h3 = state.process<double, double>(0.0, iir);  // 0.125

	assert(near(h0, 1.0));
	assert(near(h1, 0.5));
	assert(near(h2, 0.25));
	assert(near(h3, 0.125));

	std::cout << "  direct_form_I: passed\n";
}

void test_direct_form_II() {
	// Same first-order IIR test
	BiquadCoefficients<double> iir(1.0, 0.0, 0.0, -0.5, 0.0);
	DirectFormII<double> state;

	double h0 = state.process<double, double>(1.0, iir);
	double h1 = state.process<double, double>(0.0, iir);
	double h2 = state.process<double, double>(0.0, iir);
	double h3 = state.process<double, double>(0.0, iir);

	assert(near(h0, 1.0));
	assert(near(h1, 0.5));
	assert(near(h2, 0.25));
	assert(near(h3, 0.125));

	std::cout << "  direct_form_II: passed\n";
}

void test_transposed_direct_form_II() {
	// Same first-order IIR test
	BiquadCoefficients<double> iir(1.0, 0.0, 0.0, -0.5, 0.0);
	TransposedDirectFormII<double> state;

	double h0 = state.process<double, double>(1.0, iir);
	double h1 = state.process<double, double>(0.0, iir);
	double h2 = state.process<double, double>(0.0, iir);
	double h3 = state.process<double, double>(0.0, iir);

	assert(near(h0, 1.0));
	assert(near(h1, 0.5));
	assert(near(h2, 0.25));
	assert(near(h3, 0.125));

	std::cout << "  transposed_direct_form_II: passed\n";
}

void test_all_forms_agree() {
	// All three state forms should produce identical output for the
	// same second-order biquad with the same input sequence.
	BiquadCoefficients<double> bq(0.2, 0.4, 0.2, -0.8, 0.3);

	DirectFormI<double>            df1;
	DirectFormII<double>           df2;
	TransposedDirectFormII<double> tdf2;

	// Process an impulse followed by zeros
	for (int n = 0; n < 20; ++n) {
		double x = (n == 0) ? 1.0 : 0.0;
		double y1 = df1.process<double, double>(x, bq);
		double y2 = df2.process<double, double>(x, bq);
		double y3 = tdf2.process<double, double>(x, bq);

		assert(near(y1, y2, 1e-12));
		assert(near(y1, y3, 1e-12));
	}

	// Process a step input
	df1.reset(); df2.reset(); tdf2.reset();
	for (int n = 0; n < 50; ++n) {
		double x = 1.0;
		double y1 = df1.process<double, double>(x, bq);
		double y2 = df2.process<double, double>(x, bq);
		double y3 = tdf2.process<double, double>(x, bq);

		assert(near(y1, y2, 1e-12));
		assert(near(y1, y3, 1e-12));
	}

	std::cout << "  all_forms_agree: passed\n";
}

void test_mixed_precision_state() {
	// Verify that float coefficients with double state compiles and works.
	// This tests the three-scalar parameterization.
	BiquadCoefficients<float> bq(0.2f, 0.4f, 0.2f, -0.8f, 0.3f);
	DirectFormII<double> state;  // state in double

	float y0 = state.process<float, float>(1.0f, bq);
	float y1 = state.process<float, float>(0.0f, bq);

	// Just verify it compiles and produces reasonable output
	assert(std::isfinite(y0));
	assert(std::isfinite(y1));
	assert(y0 != 0.0f);  // impulse response should be non-zero

	std::cout << "  mixed_precision_state: passed\n";
}

void test_pole_zero_layout() {
	PoleZeroLayout<double, 4> layout;

	assert(layout.num_poles() == 0);
	assert(layout.max_poles() == 4);

	// Add a conjugate pair
	std::complex<double> pole(-.5, 0.5);
	std::complex<double> zero(-1.0, 0.0);
	layout.add_conjugate_pairs(pole, zero);

	assert(layout.num_poles() == 2);
	assert(layout.num_pairs() == 1);
	assert(layout[0].poles.is_conjugate());

	// Add another conjugate pair
	std::complex<double> pole2(-0.3, 0.7);
	std::complex<double> zero2(-1.0, 0.0);
	layout.add_conjugate_pairs(pole2, zero2);

	assert(layout.num_poles() == 4);
	assert(layout.num_pairs() == 2);

	// Test reset
	layout.reset();
	assert(layout.num_poles() == 0);

	// Test odd-order: 2 conjugate poles + 1 real pole
	PoleZeroLayout<double, 5> odd_layout;
	odd_layout.add_conjugate_pairs(
		std::complex<double>(-0.5, 0.5),
		std::complex<double>(-1.0, 0.0));
	odd_layout.add_conjugate_pairs(
		std::complex<double>(-0.3, 0.7),
		std::complex<double>(-1.0, 0.0));
	odd_layout.add(
		std::complex<double>(-0.8, 0.0),
		std::complex<double>(-1.0, 0.0));

	assert(odd_layout.num_poles() == 5);
	assert(odd_layout.num_pairs() == 3);
	assert(odd_layout[2].is_single_pole());

	std::cout << "  pole_zero_layout: passed\n";
}

void test_cascade_from_layout() {
	// Build a simple 2nd-order layout (single conjugate pair)
	PoleZeroLayout<double, 2> layout;
	layout.set_normal(0.0, 1.0);  // unity gain at DC

	// Place poles at 0.9 * e^(+/- j*pi/4) and zeros at z = -1 (both)
	std::complex<double> pole(0.9 * std::cos(pi / 4.0),
	                          0.9 * std::sin(pi / 4.0));
	std::complex<double> zero(-1.0, 0.0);
	layout.add_conjugate_pairs(pole, zero);

	Cascade<double, 1> cascade;
	cascade.set_layout(layout);

	assert(cascade.num_stages() == 1);

	// The cascade should have finite, non-zero response at DC
	auto r_dc = cascade.response(0.0);
	assert(std::isfinite(std::abs(r_dc)));
	assert(std::abs(r_dc) > 0.0);

	// Process an impulse
	std::array<DirectFormII<double>, 1> state{};
	double h0 = cascade.process<DirectFormII<double>, double>(1.0, state);
	double h1 = cascade.process<DirectFormII<double>, double>(0.0, state);
	assert(std::isfinite(h0));
	assert(std::isfinite(h1));

	std::cout << "  cascade_from_layout: passed\n";
}

void test_cascade_response() {
	// Build a 4th-order cascade (2 stages) via two pole/zero pairs
	// and verify the combined response equals the product of
	// individual stage responses.
	PoleZeroLayout<double, 4> layout;
	layout.set_normal(0.0, 1.0);

	layout.add_conjugate_pairs(
		std::complex<double>(0.5, 0.3),
		std::complex<double>(-1.0, 0.0));
	layout.add_conjugate_pairs(
		std::complex<double>(0.3, 0.6),
		std::complex<double>(-1.0, 0.0));

	Cascade<double, 2> cascade;
	cascade.set_layout(layout);
	assert(cascade.num_stages() == 2);

	// Verify combined response is product of individual stages
	double f = 0.125;
	auto r0 = cascade.stage(0).response(f);
	auto r1 = cascade.stage(1).response(f);
	auto r_cascade = cascade.response(f);

	auto product = r0 * r1;
	assert(near(std::abs(r_cascade), std::abs(product), 1e-10));

	std::cout << "  cascade_response: passed\n";
}

void test_smoothed_cascade() {
	// Verify SmoothedCascade transitions without producing NaN/Inf
	SmoothedCascade<double, 1> sc(64);

	// Create two different cascades
	Cascade<double, 1> c1, c2;

	// Manually configure a simple first-order lowpass
	PoleZeroLayout<double, 2> layout1;
	layout1.set_normal(0.0, 1.0);
	layout1.add_conjugate_pairs(
		std::complex<double>(0.5, 0.3),
		std::complex<double>(-1.0, 0.0));
	c1.set_layout(layout1);

	PoleZeroLayout<double, 2> layout2;
	layout2.set_normal(0.0, 1.0);
	layout2.add_conjugate_pairs(
		std::complex<double>(0.3, 0.6),
		std::complex<double>(-1.0, 0.0));
	c2.set_layout(layout2);

	// Set initial cascade
	sc.set_cascade(c1);
	assert(!sc.in_transition());

	// Process some samples
	std::array<DirectFormII<double>, 1> state{};
	std::vector<double> buf(128, 0.0);
	buf[0] = 1.0;  // impulse
	std::span<double> samples(buf);
	sc.process_block(samples, state);

	// Trigger transition
	sc.set_cascade(c2);
	assert(sc.in_transition());

	// Process through transition
	std::fill(buf.begin(), buf.end(), 0.5);  // constant signal
	sc.process_block(samples, state);

	// All outputs should be finite
	for (auto v : buf) {
		assert(std::isfinite(v));
	}

	std::cout << "  smoothed_cascade: passed\n";
}

int main() {
	std::cout << "Phase 2: Biquad Engine Tests\n";

	test_biquad_coefficients();
	test_biquad_pole_state();
	test_direct_form_I();
	test_direct_form_II();
	test_transposed_direct_form_II();
	test_all_forms_agree();
	test_mixed_precision_state();
	test_pole_zero_layout();
	test_cascade_from_layout();
	test_cascade_response();
	test_smoothed_cascade();

	std::cout << "All Phase 2 tests passed.\n";
	return 0;
}
