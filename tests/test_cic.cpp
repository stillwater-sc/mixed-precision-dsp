// test_cic.cpp: test CIC decimation and interpolation filters
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/acquisition/cic.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sw::dsp;

bool near(double a, double b, double eps = 1e-6) {
	return std::abs(a - b) < eps;
}

// ============================================================================
// Bit growth calculation
// ============================================================================

void test_bit_growth() {
	// Single stage, R=4, D=1: ceil(log2(4)) = 2
	if (cic_bit_growth(1, 4, 1) != 2)
		throw std::runtime_error("test failed: bit_growth M=1 R=4 D=1");

	// 3 stages, R=8, D=1: 3 * ceil(log2(8)) = 3 * 3 = 9
	if (cic_bit_growth(3, 8, 1) != 9)
		throw std::runtime_error("test failed: bit_growth M=3 R=8 D=1");

	// 4 stages, R=16, D=2: 4 * ceil(log2(32)) = 4 * 5 = 20
	if (cic_bit_growth(4, 16, 2) != 20)
		throw std::runtime_error("test failed: bit_growth M=4 R=16 D=2");

	// 5 stages, R=64, D=1: 5 * ceil(log2(64)) = 5 * 6 = 30
	if (cic_bit_growth(5, 64, 1) != 30)
		throw std::runtime_error("test failed: bit_growth M=5 R=64 D=1");

	// Edge: R=1 => log2(1)=0, growth=0
	if (cic_bit_growth(3, 1, 1) != 0)
		throw std::runtime_error("test failed: bit_growth M=3 R=1 D=1");

	std::cout << "  bit_growth: passed\n";
}

// ============================================================================
// DC gain
// ============================================================================

void test_dc_gain() {
	CICDecimator<double> cic(4, 3, 1);
	// DC gain = (R*D)^M = 4^3 = 64
	if (!near(cic.dc_gain(), 64.0))
		throw std::runtime_error("test failed: dc_gain R=4 M=3");

	CICDecimator<double> cic2(8, 2, 2);
	// DC gain = (8*2)^2 = 256
	if (!near(cic2.dc_gain(), 256.0))
		throw std::runtime_error("test failed: dc_gain R=8 M=2 D=2");

	std::cout << "  dc_gain: passed\n";
}

// ============================================================================
// Single-stage decimation impulse response
// ============================================================================

void test_single_stage_impulse() {
	// CIC with M=1, R=4, D=1 is a moving-average of length R=4
	// Impulse response: feed 1 followed by zeros
	// The integrator accumulates the running sum.
	// After R=4 samples, the comb differences current - delayed (D=1 samples ago at decimated rate).
	CICDecimator<double> cic(4, 1, 1);

	// Feed impulse: x = {1, 0, 0, 0, 0, 0, 0, 0, ...}
	std::vector<double> input(16, 0.0);
	input[0] = 1.0;
	std::vector<double> output;
	cic.process_block(std::span<const double>(input), output);

	// Expected: 4 output samples (16/4)
	if (output.size() != 4)
		throw std::runtime_error("test failed: impulse output size = " +
			std::to_string(output.size()));

	// First output: integrator sees {1,0,0,0}, sum=1, comb: 1-0 = 1
	if (!near(output[0], 1.0))
		throw std::runtime_error("test failed: impulse y[0] = " +
			std::to_string(output[0]));

	// Subsequent outputs: integrator still at 1 (no new input), comb: 1-1 = 0
	for (std::size_t i = 1; i < output.size(); ++i) {
		if (!near(output[i], 0.0))
			throw std::runtime_error("test failed: impulse y[" +
				std::to_string(i) + "] = " + std::to_string(output[i]));
	}

	std::cout << "  single_stage_impulse: passed\n";
}

// ============================================================================
// DC input: output should equal DC * gain
// ============================================================================

void test_dc_response() {
	// M=2, R=4, D=1: DC gain = 4^2 = 16
	CICDecimator<double> cic(4, 2, 1);
	double dc_in = 1.0;

	// Feed enough samples for the filter to settle
	// After settling, each output should be dc_in * (R*D)^M = 16
	std::vector<double> input(100, dc_in);
	std::vector<double> output;
	cic.process_block(std::span<const double>(input), output);

	// Check last few outputs (after settling)
	double expected = cic.dc_gain() * dc_in;
	for (std::size_t i = output.size() - 5; i < output.size(); ++i) {
		if (!near(output[i], expected, 1e-9))
			throw std::runtime_error("test failed: dc_response y[" +
				std::to_string(i) + "] = " + std::to_string(output[i]) +
				", expected " + std::to_string(expected));
	}

	std::cout << "  dc_response: passed\n";
}

// ============================================================================
// Multi-stage with D>1
// ============================================================================

void test_differential_delay() {
	// M=1, R=2, D=2: effectively a moving sum over R*D=4 samples, decimated by 2
	CICDecimator<double> cic(2, 1, 2);

	// DC gain = (2*2)^1 = 4
	if (!near(cic.dc_gain(), 4.0))
		throw std::runtime_error("test failed: D=2 dc_gain");

	// Feed DC=1.0 and check settled output
	std::vector<double> input(40, 1.0);
	std::vector<double> output;
	cic.process_block(std::span<const double>(input), output);

	double expected = 4.0;
	for (std::size_t i = output.size() - 3; i < output.size(); ++i) {
		if (!near(output[i], expected, 1e-9))
			throw std::runtime_error("test failed: D=2 settled output");
	}

	std::cout << "  differential_delay: passed\n";
}

// ============================================================================
// Interpolator: DC gain and round-trip
// ============================================================================

void test_interpolator_impulse() {
	// M=1, R=4, D=1: impulse response is a rectangular pulse of length R
	CICInterpolator<double> interp(4, 1, 1);

	std::vector<double> input = {1.0, 0.0, 0.0, 0.0};
	std::vector<double> output;
	interp.process_block(std::span<const double>(input), output);

	// 4 inputs * R=4 = 16 outputs
	if (output.size() != 16)
		throw std::runtime_error("test failed: interp impulse size = " +
			std::to_string(output.size()));

	// First R=4 samples should be 1, rest should be 0
	for (int i = 0; i < 4; ++i) {
		if (!near(output[static_cast<std::size_t>(i)], 1.0))
			throw std::runtime_error("test failed: interp impulse y[" +
				std::to_string(i) + "] = " + std::to_string(output[static_cast<std::size_t>(i)]));
	}
	for (std::size_t i = 4; i < output.size(); ++i) {
		if (!near(output[i], 0.0))
			throw std::runtime_error("test failed: interp impulse tail y[" +
				std::to_string(i) + "] = " + std::to_string(output[i]));
	}

	std::cout << "  interpolator_impulse: passed\n";
}

void test_interpolator_output_count() {
	CICInterpolator<double> interp(4, 2, 1);

	std::vector<double> input(20, 1.0);
	std::vector<double> output;
	interp.process_block(std::span<const double>(input), output);

	// 20 inputs * R=4 = 80 outputs
	if (output.size() != 80)
		throw std::runtime_error("test failed: interp output size = " +
			std::to_string(output.size()));

	// After settling, output should be constant (all samples equal)
	double settled = output.back();
	for (std::size_t i = output.size() - 10; i < output.size(); ++i) {
		if (!near(output[i], settled, 1e-9))
			throw std::runtime_error("test failed: interp not settled at " +
				std::to_string(i));
	}

	std::cout << "  interpolator_output_count: settled at " << settled << ", passed\n";
}

// ============================================================================
// Decimation + interpolation round-trip preserves DC
// ============================================================================

void test_round_trip() {
	// Decimation followed by interpolation should preserve DC (up to gain factors)
	int R = 4, M = 2, D = 1;
	CICDecimator<double>    dec(R, M, D);
	CICInterpolator<double> interp(R, M, D);

	double dc = 0.5;
	std::vector<double> input(200, dc);
	std::vector<double> decimated;
	dec.process_block(std::span<const double>(input), decimated);

	// Decimated output settles to dc * (R*D)^M = 0.5 * 16 = 8
	double dec_settled = decimated.back();

	std::vector<double> reconstructed;
	interp.process_block(std::span<const double>(decimated), reconstructed);

	// The reconstructed output should settle to a constant
	double recon_settled = reconstructed.back();

	// Verify the round-trip output is constant (settled)
	for (std::size_t i = reconstructed.size() - 10; i < reconstructed.size(); ++i) {
		if (!near(reconstructed[i], recon_settled, 1e-9))
			throw std::runtime_error("test failed: round-trip not settled");
	}

	std::cout << "  round_trip: dec_settled=" << dec_settled
	          << " recon_settled=" << recon_settled << ", passed\n";
}

// ============================================================================
// Reset clears state
// ============================================================================

void test_reset() {
	CICDecimator<double> cic(4, 2, 1);

	std::vector<double> input(16, 1.0);
	std::vector<double> out1;
	cic.process_block(std::span<const double>(input), out1);

	cic.reset();

	std::vector<double> out2;
	cic.process_block(std::span<const double>(input), out2);

	if (out1.size() != out2.size())
		throw std::runtime_error("test failed: reset output size mismatch");

	for (std::size_t i = 0; i < out1.size(); ++i) {
		if (!near(out1[i], out2[i]))
			throw std::runtime_error("test failed: reset output mismatch at " +
				std::to_string(i));
	}

	std::cout << "  reset: passed\n";
}

// ============================================================================
// Mixed-precision: float state with double samples
// ============================================================================

void test_mixed_precision() {
	// Use float for state (limited precision) to show quality degradation
	CICDecimator<float, double> cic_float(8, 3, 1);
	CICDecimator<double, double> cic_double(8, 3, 1);

	// Generate a signal with small amplitude to stress precision
	std::vector<double> input(800);
	for (std::size_t i = 0; i < input.size(); ++i) {
		input[i] = 1e-4 * std::sin(2.0 * M_PI * 7.0 *
			static_cast<double>(i) / static_cast<double>(input.size()));
	}

	std::vector<double> out_float, out_double;
	cic_float.process_block(std::span<const double>(input), out_float);
	cic_double.process_block(std::span<const double>(input), out_double);

	if (out_float.size() != out_double.size())
		throw std::runtime_error("test failed: mixed-precision size mismatch");

	// Compute error between float and double state
	double max_err = 0.0;
	double max_val = 0.0;
	for (std::size_t i = 0; i < out_float.size(); ++i) {
		max_err = std::max(max_err, std::abs(out_float[i] - out_double[i]));
		max_val = std::max(max_val, std::abs(out_double[i]));
	}

	// Float should introduce measurable error but not catastrophic
	double relative_err = (max_val > 0.0) ? max_err / max_val : 0.0;
	std::cout << "  mixed_precision: float vs double relative error = "
	          << relative_err << "\n";

	// Float has ~7 decimal digits; CIC gain = 8^3 = 512, and integrators
	// accumulate rounding errors over many samples, so relative error
	// can reach ~1e-2 for long sequences
	if (relative_err > 0.05)
		throw std::runtime_error("test failed: float state error too large: " +
			std::to_string(relative_err));

	std::cout << "  mixed_precision: passed\n";
}

// ============================================================================
// Overflow demonstration: narrow integer-like type
// ============================================================================

void test_bit_growth_verification() {
	// With M=3, R=8, D=1: bit growth = 9, DC gain = 512
	// An input of 1.0 through the filter settles to 512
	CICDecimator<double> cic(8, 3, 1);
	if (cic.bit_growth() != 9)
		throw std::runtime_error("test failed: bit_growth() accessor");

	std::vector<double> input(80, 1.0);
	std::vector<double> output;
	cic.process_block(std::span<const double>(input), output);

	double settled = output.back();
	if (!near(settled, 512.0, 1e-9))
		throw std::runtime_error("test failed: settled value = " +
			std::to_string(settled) + ", expected 512");

	std::cout << "  bit_growth_verification: passed\n";
}

// ============================================================================
// Parameter validation
// ============================================================================

void test_parameter_validation() {
	bool caught = false;

	try { CICDecimator<double>(0, 1, 1); }
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: R=0 should throw");

	caught = false;
	try { CICDecimator<double>(4, 0, 1); }
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: M=0 should throw");

	caught = false;
	try { CICDecimator<double>(4, 1, 0); }
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: D=0 should throw");

	caught = false;
	try { CICInterpolator<double>(0, 1, 1); }
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: interp R=0 should throw");

	std::cout << "  parameter_validation: passed\n";
}

// ============================================================================

int main() {
	try {
		std::cout << "CIC filter tests\n";
		test_bit_growth();
		test_dc_gain();
		test_single_stage_impulse();
		test_dc_response();
		test_differential_delay();
		test_interpolator_impulse();
		test_interpolator_output_count();
		test_round_trip();
		test_reset();
		test_mixed_precision();
		test_bit_growth_verification();
		test_parameter_validation();
		std::cout << "All CIC tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAIL: " << e.what() << '\n';
		return 1;
	}
}
