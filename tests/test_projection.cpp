// test_projection.cpp: test project_onto / embed_into type conversion
//
// Validates the design → project → verify → embed workflow for
// mixed-precision coefficient management.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/types/projection.hpp>
#include <sw/dsp/filter/iir/butterworth.hpp>
#include <sw/dsp/analysis/stability.hpp>
#include <sw/dsp/analysis/sensitivity.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace sw::dsp;

bool near(double a, double b, double eps = 1e-6) {
	return std::abs(a - b) < eps;
}

// ========== BiquadCoefficients ==========

void test_biquad_project_onto() {
	BiquadCoefficients<double> bq_d(0.1234567890123456, -0.2, 0.3, -0.4, 0.5);

	auto bq_f = project_onto<float>(bq_d);

	// float has ~7 decimal digits — should be close but not identical
	if (!(near(static_cast<double>(bq_f.b0), 0.1234567890123456, 1e-6)))
		throw std::runtime_error("test failed: project biquad b0");
	if (!(static_cast<double>(bq_f.b0) != bq_d.b0))
		throw std::runtime_error("test failed: float should lose some precision");

	std::cout << "  biquad_project_onto: passed\n";
}

void test_biquad_embed_into() {
	BiquadCoefficients<float> bq_f(0.125f, -0.25f, 0.5f, -0.75f, 0.875f);

	auto bq_d = embed_into<double>(bq_f);

	// Embedding is exact — these float values are exactly representable in double
	if (!(bq_d.b0 == static_cast<double>(bq_f.b0)))
		throw std::runtime_error("test failed: embed biquad b0 should be exact");
	if (!(bq_d.a2 == static_cast<double>(bq_f.a2)))
		throw std::runtime_error("test failed: embed biquad a2 should be exact");

	std::cout << "  biquad_embed_into: passed\n";
}

void test_biquad_roundtrip() {
	// project then embed should preserve float precision
	BiquadCoefficients<double> original(1.0 / 3.0, -2.0 / 7.0, 0.5, -0.4, 0.2);

	auto projected = project_onto<float>(original);
	auto embedded = embed_into<double>(projected);

	// The roundtrip should equal the float-precision version
	if (!(embedded.b0 == static_cast<double>(static_cast<float>(original.b0))))
		throw std::runtime_error("test failed: roundtrip b0");

	std::cout << "  biquad_roundtrip: passed\n";
}

// ========== Cascade ==========

void test_cascade_project_onto() {
	// Design a Butterworth in double, project to float
	iir::ButterworthLowPass<4> butter;
	butter.setup(4, 44100.0, 1000.0);
	const auto& cascade_d = butter.cascade();

	auto cascade_f = project_onto<float>(cascade_d);

	if (!(cascade_f.num_stages() == cascade_d.num_stages()))
		throw std::runtime_error("test failed: projected cascade stage count");

	// Both should be stable
	if (!(is_stable(cascade_d)))
		throw std::runtime_error("test failed: original cascade not stable");
	if (!(is_stable(cascade_f)))
		throw std::runtime_error("test failed: projected cascade not stable");

	// Pole displacement should be small (float is close to double)
	double disp = pole_displacement(cascade_d, cascade_f);
	if (!(disp > 0.0))
		throw std::runtime_error("test failed: displacement should be > 0");
	if (!(disp < 1e-5))
		throw std::runtime_error("test failed: float displacement too large");

	std::cout << "  cascade_project_onto: passed (displacement=" << disp << ")\n";
}

void test_cascade_embed_into() {
	// Design in float, embed into double for analysis
	iir::ButterworthLowPass<4, float> butter_f;
	butter_f.setup(4, 44100.0, 1000.0);

	auto cascade_d = embed_into<double>(butter_f.cascade());

	if (!(cascade_d.num_stages() == butter_f.cascade().num_stages()))
		throw std::runtime_error("test failed: embedded cascade stage count");
	if (!(is_stable(cascade_d)))
		throw std::runtime_error("test failed: embedded cascade not stable");

	std::cout << "  cascade_embed_into: passed\n";
}

// ========== Dense Vector ==========

void test_vector_project_onto() {
	mtl::vec::dense_vector<double> v(5);
	for (std::size_t i = 0; i < 5; ++i) v[i] = static_cast<double>(i) * 0.1;

	auto v_f = project_onto<float>(v);
	if (!(v_f.size() == 5))
		throw std::runtime_error("test failed: projected vector size");
	if (!(near(static_cast<double>(v_f[3]), 0.3, 1e-6)))
		throw std::runtime_error("test failed: projected vector value");

	std::cout << "  vector_project_onto: passed\n";
}

void test_vector_embed_into() {
	mtl::vec::dense_vector<float> v(3);
	v[0] = 0.5f; v[1] = 1.0f; v[2] = 1.5f;

	auto v_d = embed_into<double>(v);
	if (!(v_d[1] == 1.0))
		throw std::runtime_error("test failed: embedded vector exact");

	std::cout << "  vector_embed_into: passed\n";
}

// ========== Dense 2D ==========

void test_matrix_project_onto() {
	mtl::mat::dense2D<double> m(3, 4);
	for (std::size_t r = 0; r < 3; ++r)
		for (std::size_t c = 0; c < 4; ++c)
			m(r, c) = static_cast<double>(r * 4 + c) / 12.0;

	auto m_f = project_onto<float>(m);
	if (!(m_f.num_rows() == 3 && m_f.num_cols() == 4))
		throw std::runtime_error("test failed: projected matrix dims");
	if (!(near(static_cast<double>(m_f(1, 2)), 6.0 / 12.0, 1e-6)))
		throw std::runtime_error("test failed: projected matrix value");

	std::cout << "  matrix_project_onto: passed\n";
}

void test_matrix_embed_into() {
	mtl::mat::dense2D<float> m(2, 2);
	m(0, 0) = 1.0f; m(0, 1) = 2.0f;
	m(1, 0) = 3.0f; m(1, 1) = 4.0f;

	auto m_d = embed_into<double>(m);
	if (!(m_d(1, 1) == 4.0))
		throw std::runtime_error("test failed: embedded matrix exact");

	std::cout << "  matrix_embed_into: passed\n";
}

// ========== Design Workflow ==========

void test_design_project_verify_workflow() {
	// The canonical mixed-precision workflow:
	// 1. Design in double
	// 2. Project onto target type
	// 3. Verify quality (pole displacement, stability)
	// 4. Embed back for comparison

	// Step 1: Design
	iir::ButterworthLowPass<4> design;
	design.setup(4, 44100.0, 1000.0);
	const auto& original = design.cascade();

	// Step 2: Project to float (simulating FPGA/embedded target)
	auto target = project_onto<float>(original);

	// Step 3: Verify
	double disp = pole_displacement(original, target);
	double margin_orig = stability_margin(original);
	double margin_target = stability_margin(target);

	if (!(is_stable(target)))
		throw std::runtime_error("test failed: target cascade should be stable");
	if (!(disp < 1e-5))
		throw std::runtime_error("test failed: pole displacement too large");
	if (!(std::abs(margin_orig - margin_target) < 1e-4))
		throw std::runtime_error("test failed: stability margins should be close");

	// Step 4: Embed back for comparison
	auto recovered = embed_into<double>(target);
	double roundtrip_disp = pole_displacement(original, recovered);
	// Should equal the projection displacement (no additional loss from embed)
	if (!(near(roundtrip_disp, disp, 1e-15)))
		throw std::runtime_error("test failed: embed should not add error");

	std::cout << "  design_project_verify: passed (disp=" << disp
	          << ", margin_orig=" << margin_orig
	          << ", margin_target=" << margin_target << ")\n";
}

int main() {
	try {
		std::cout << "Projection/Embedding Tests\n";

		// BiquadCoefficients
		test_biquad_project_onto();
		test_biquad_embed_into();
		test_biquad_roundtrip();

		// Cascade
		test_cascade_project_onto();
		test_cascade_embed_into();

		// Dense vector
		test_vector_project_onto();
		test_vector_embed_into();

		// Dense 2D matrix
		test_matrix_project_onto();
		test_matrix_embed_into();

		// Full workflow
		test_design_project_verify_workflow();

		std::cout << "All projection/embedding tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << '\n';
		return 1;
	}
}
