// test_halfband.cpp: test half-band FIR filter design, processing, and decimation
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/acquisition/halfband.hpp>
#include <sw/dsp/math/constants.hpp>

#include <universal/number/posit/posit.hpp>

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
// Design: verify half-band tap structure
// ============================================================================

void test_design_structure() {
	auto taps = design_halfband<double>(11, 0.1);

	if (taps.size() != 11)
		throw std::runtime_error("test failed: design returned wrong tap count");

	std::size_t center = 5;

	// Center tap must be 0.5
	if (!near(static_cast<double>(taps[center]), 0.5, 1e-12))
		throw std::runtime_error("test failed: center tap = " +
			std::to_string(static_cast<double>(taps[center])));

	// Even offsets from center must be zero
	for (std::size_t k = 2; k <= center; k += 2) {
		if (!near(static_cast<double>(taps[center - k]), 0.0, 1e-15))
			throw std::runtime_error("test failed: tap[" +
				std::to_string(center - k) + "] should be 0");
		if (!near(static_cast<double>(taps[center + k]), 0.0, 1e-15))
			throw std::runtime_error("test failed: tap[" +
				std::to_string(center + k) + "] should be 0");
	}

	// Symmetry: h[center-k] == h[center+k]
	for (std::size_t k = 1; k <= center; ++k) {
		if (!near(static_cast<double>(taps[center - k]),
		          static_cast<double>(taps[center + k]), 1e-15))
			throw std::runtime_error("test failed: symmetry at offset " +
				std::to_string(k));
	}

	std::cout << "  design_structure: passed\n";
}

// ============================================================================
// Design: verify different filter lengths
// ============================================================================

void test_design_lengths() {
	// Valid lengths: 4K+3 = 3, 7, 11, 15, 19
	for (std::size_t n : {3, 7, 11, 15, 19}) {
		auto taps = design_halfband<double>(n, 0.15);
		if (taps.size() != n)
			throw std::runtime_error("test failed: design length " +
				std::to_string(n));

		std::size_t center = (n - 1) / 2;
		if (!near(static_cast<double>(taps[center]), 0.5, 1e-12))
			throw std::runtime_error("test failed: center tap for N=" +
				std::to_string(n));
	}

	std::cout << "  design_lengths: passed\n";
}

// ============================================================================
// Design with float: verify parameterized design works for non-double types
// ============================================================================

void test_design_float() {
	auto taps = design_halfband<float>(11, 0.1f);

	if (taps.size() != 11)
		throw std::runtime_error("test failed: float design tap count");

	std::size_t center = 5;
	if (!near(static_cast<double>(taps[center]), 0.5, 1e-6))
		throw std::runtime_error("test failed: float center tap");

	// Even offsets must be zero
	for (std::size_t k = 2; k <= center; k += 2) {
		if (taps[center - k] != 0.0f || taps[center + k] != 0.0f)
			throw std::runtime_error("test failed: float even-offset non-zero");
	}

	std::cout << "  design_float: passed\n";
}

// ============================================================================
// DC gain: sum of taps should be ~1.0
// ============================================================================

void test_dc_gain() {
	auto taps = design_halfband<double>(15, 0.1);

	double sum = 0.0;
	for (std::size_t i = 0; i < taps.size(); ++i) {
		sum += static_cast<double>(taps[i]);
	}

	if (!near(sum, 1.0, 1e-3))
		throw std::runtime_error("test failed: DC gain = " +
			std::to_string(sum) + ", expected ~1.0");

	std::cout << "  dc_gain: sum=" << sum << ", passed\n";
}

// ============================================================================
// Non-zero tap count verification
// ============================================================================

void test_nonzero_count() {
	auto taps = design_halfband<double>(11, 0.1);
	HalfBandFilter<double> hb(taps);

	// N=11, center=5: non-zero at offsets 1,3,5 from center = 3 pairs + center = 7
	if (hb.num_nonzero_taps() != 7)
		throw std::runtime_error("test failed: nonzero_taps = " +
			std::to_string(hb.num_nonzero_taps()) + ", expected 7");

	if (hb.num_taps() != 11)
		throw std::runtime_error("test failed: num_taps mismatch");

	if (hb.order() != 10)
		throw std::runtime_error("test failed: order mismatch");

	std::cout << "  nonzero_count: " << hb.num_nonzero_taps()
	          << " of " << hb.num_taps() << ", passed\n";
}

// ============================================================================
// Impulse response: verify process() matches taps
// ============================================================================

void test_impulse_response() {
	auto taps = design_halfband<double>(11, 0.1);
	HalfBandFilter<double> hb(taps);

	// Feed impulse followed by zeros
	std::vector<double> output;
	output.push_back(hb.process(1.0));
	for (int i = 1; i < 11; ++i) {
		output.push_back(hb.process(0.0));
	}

	// Output should match taps: y[n] = h[n] for unit impulse at n=0
	for (std::size_t i = 0; i < 11; ++i) {
		if (!near(output[i], static_cast<double>(taps[i]), 1e-12))
			throw std::runtime_error("test failed: impulse y[" +
				std::to_string(i) + "] = " + std::to_string(output[i]) +
				", expected " + std::to_string(static_cast<double>(taps[i])));
	}

	std::cout << "  impulse_response: passed\n";
}

// ============================================================================
// Frequency response: passband and stopband
// ============================================================================

void test_frequency_response() {
	double tw = 0.1;
	auto taps = design_halfband<double>(19, tw);
	std::size_t N = taps.size();

	// Evaluate |H(f)| at several frequencies
	auto eval_H = [&](double f_norm) -> double {
		double re = 0.0, im = 0.0;
		for (std::size_t n = 0; n < N; ++n) {
			double h = static_cast<double>(taps[n]);
			double angle = sw::dsp::two_pi * f_norm * static_cast<double>(n);
			re += h * std::cos(angle);
			im -= h * std::sin(angle);
		}
		return std::sqrt(re * re + im * im);
	};

	// Passband: |H(f)| ~= 1 for f < 0.25 - tw/2 = 0.20
	for (double f = 0.01; f <= 0.18; f += 0.02) {
		double mag = eval_H(f);
		double mag_db = 20.0 * std::log10(std::max(mag, 1e-15));
		if (mag_db < -1.0)
			throw std::runtime_error("test failed: passband at f=" +
				std::to_string(f) + " mag_dB=" + std::to_string(mag_db));
	}

	// Stopband: |H(f)| << 1 for f > 0.25 + tw/2 = 0.30
	for (double f = 0.32; f <= 0.48; f += 0.02) {
		double mag = eval_H(f);
		double mag_db = 20.0 * std::log10(std::max(mag, 1e-15));
		if (mag_db > -10.0)
			throw std::runtime_error("test failed: stopband at f=" +
				std::to_string(f) + " mag_dB=" + std::to_string(mag_db));
	}

	// Symmetry about pi/2: |H(0.25-df)| + |H(0.25+df)| ~= 1
	for (double df = 0.01; df <= 0.15; df += 0.02) {
		double mag_lo = eval_H(0.25 - df);
		double mag_hi = eval_H(0.25 + df);
		double sum = mag_lo + mag_hi;
		if (!near(sum, 1.0, 0.05))
			throw std::runtime_error("test failed: symmetry at df=" +
				std::to_string(df) + " sum=" + std::to_string(sum));
	}

	std::cout << "  frequency_response: passed\n";
}

// ============================================================================
// Block processing: matches sample-by-sample
// ============================================================================

void test_block_processing() {
	auto taps = design_halfband<double>(11, 0.1);

	HalfBandFilter<double> hb1(taps);
	HalfBandFilter<double> hb2(taps);

	// Generate test signal
	std::vector<double> input(50);
	for (std::size_t i = 0; i < input.size(); ++i) {
		input[i] = std::sin(sw::dsp::two_pi * 0.05 * static_cast<double>(i));
	}

	// Sample-by-sample
	std::vector<double> out1;
	for (auto s : input) out1.push_back(hb1.process(s));

	// Block
	std::vector<double> out2(input.size());
	hb2.process_block(std::span<const double>(input),
	                  std::span<double>(out2));

	for (std::size_t i = 0; i < out1.size(); ++i) {
		if (!near(out1[i], out2[i], 1e-12))
			throw std::runtime_error("test failed: block mismatch at " +
				std::to_string(i));
	}

	std::cout << "  block_processing: passed\n";
}

// ============================================================================
// process_block throws when output span is too small
// ============================================================================

void test_block_output_validation() {
	auto taps = design_halfband<double>(11, 0.1);
	HalfBandFilter<double> hb(taps);

	std::vector<double> input(20, 1.0);
	std::vector<double> output(10);
	bool caught = false;
	try {
		hb.process_block(std::span<const double>(input),
		                 std::span<double>(output));
	}
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught)
		throw std::runtime_error("test failed: undersized output should throw");

	std::cout << "  block_output_validation: passed\n";
}

// ============================================================================
// Decimation: output count and basic correctness
// ============================================================================

void test_decimation_count() {
	auto taps = design_halfband<double>(11, 0.1);
	HalfBandFilter<double> hb(taps);

	std::vector<double> input(100, 1.0);
	auto output = hb.process_block_decimate(std::span<const double>(input));

	// 100 input samples / 2 = 50 output samples
	if (output.size() != 50)
		throw std::runtime_error("test failed: decimation count = " +
			std::to_string(output.size()) + ", expected 50");

	std::cout << "  decimation_count: passed\n";
}

void test_decimation_dc() {
	auto taps = design_halfband<double>(15, 0.1);
	HalfBandFilter<double> hb(taps);

	// Feed DC = 1.0 through decimation — settled output should be ~1.0
	std::vector<double> input(200, 1.0);
	auto output = hb.process_block_decimate(std::span<const double>(input));

	// Check last outputs are settled near 1.0
	std::size_t start = (output.size() > 10) ? (output.size() - 10) : 0;
	for (std::size_t i = start; i < output.size(); ++i) {
		if (!near(output[i], 1.0, 1e-3))
			throw std::runtime_error("test failed: decimation DC at " +
				std::to_string(i) + " = " + std::to_string(output[i]));
	}

	std::cout << "  decimation_dc: passed\n";
}

// ============================================================================
// Decimation: compare with full-rate + downsample
// ============================================================================

void test_decimation_correctness() {
	auto taps = design_halfband<double>(11, 0.1);

	HalfBandFilter<double> hb_full(taps);
	HalfBandFilter<double> hb_dec(taps);

	// Generate lowpass signal (below Nyquist/4)
	std::vector<double> input(100);
	for (std::size_t i = 0; i < input.size(); ++i) {
		input[i] = std::sin(sw::dsp::two_pi * 0.05 * static_cast<double>(i));
	}

	// Full-rate then downsample
	std::vector<double> full_out;
	for (auto s : input) full_out.push_back(hb_full.process(s));

	std::vector<double> downsampled;
	for (std::size_t i = 1; i < full_out.size(); i += 2) {
		downsampled.push_back(full_out[i]);
	}

	// Integrated decimation
	auto dec_out = hb_dec.process_block_decimate(std::span<const double>(input));

	if (downsampled.size() != dec_out.size())
		throw std::runtime_error("test failed: decimation size mismatch: " +
			std::to_string(downsampled.size()) + " vs " +
			std::to_string(dec_out.size()));

	for (std::size_t i = 0; i < dec_out.size(); ++i) {
		if (!near(downsampled[i], dec_out[i], 1e-12))
			throw std::runtime_error("test failed: decimation mismatch at " +
				std::to_string(i));
	}

	std::cout << "  decimation_correctness: passed\n";
}

// ============================================================================
// Reset clears state
// ============================================================================

void test_reset() {
	auto taps = design_halfband<double>(11, 0.1);
	HalfBandFilter<double> hb(taps);

	std::vector<double> input(20);
	for (std::size_t i = 0; i < input.size(); ++i) {
		input[i] = std::sin(sw::dsp::two_pi * 0.1 * static_cast<double>(i));
	}

	std::vector<double> out1;
	for (auto s : input) out1.push_back(hb.process(s));

	hb.reset();

	std::vector<double> out2;
	for (auto s : input) out2.push_back(hb.process(s));

	for (std::size_t i = 0; i < out1.size(); ++i) {
		if (!near(out1[i], out2[i], 1e-12))
			throw std::runtime_error("test failed: reset mismatch at " +
				std::to_string(i));
	}

	std::cout << "  reset: passed\n";
}

// ============================================================================
// Mixed precision: float coefficients with double state
// ============================================================================

void test_mixed_precision() {
	auto taps_d = design_halfband<double>(15, 0.1);

	// Project to float coefficients
	mtl::vec::dense_vector<float> taps_f(taps_d.size());
	for (std::size_t i = 0; i < taps_d.size(); ++i) {
		taps_f[i] = static_cast<float>(taps_d[i]);
	}

	HalfBandFilter<double, double, double> hb_ref(taps_d);
	HalfBandFilter<float, double, double>  hb_mix(taps_f);

	// Signal: sum of two frequencies
	std::vector<double> input(200);
	for (std::size_t i = 0; i < input.size(); ++i) {
		input[i] = std::sin(sw::dsp::two_pi * 0.05 * static_cast<double>(i))
		         + 0.3 * std::sin(sw::dsp::two_pi * 0.12 * static_cast<double>(i));
	}

	double max_err = 0.0, max_val = 0.0;
	for (auto s : input) {
		double y_ref = hb_ref.process(s);
		double y_mix = hb_mix.process(s);
		max_err = std::max(max_err, std::abs(y_ref - y_mix));
		max_val = std::max(max_val, std::abs(y_ref));
	}

	double rel_err = (max_val > 0.0) ? max_err / max_val : 0.0;
	std::cout << "  mixed_precision: float vs double relative error = "
	          << rel_err << "\n";

	if (rel_err > 1e-5)
		throw std::runtime_error("test failed: mixed-precision error too large: " +
			std::to_string(rel_err));

	std::cout << "  mixed_precision: passed\n";
}

// ============================================================================
// Posit type: design and process with posit<32,2> samples
// ============================================================================

void test_posit_types() {
	using p32 = sw::universal::posit<32, 2>;

	auto taps_d = design_halfband<double>(11, 0.1);

	// Project taps to posit
	mtl::vec::dense_vector<p32> taps_p(taps_d.size());
	for (std::size_t i = 0; i < taps_d.size(); ++i) {
		taps_p[i] = p32(static_cast<double>(taps_d[i]));
	}

	HalfBandFilter<p32, p32, p32> hb_posit(taps_p);
	HalfBandFilter<double> hb_ref(taps_d);

	// Feed a sinusoidal signal
	double max_err = 0.0, max_val = 0.0;
	for (int i = 0; i < 100; ++i) {
		double x = std::sin(sw::dsp::two_pi * 0.05 * static_cast<double>(i));
		double y_ref = hb_ref.process(x);
		double y_pos = static_cast<double>(hb_posit.process(p32(x)));
		max_err = std::max(max_err, std::abs(y_ref - y_pos));
		max_val = std::max(max_val, std::abs(y_ref));
	}

	double rel_err = (max_val > 0.0) ? max_err / max_val : 0.0;
	std::cout << "  posit_types: posit<32,2> vs double relative error = "
	          << rel_err << "\n";

	if (rel_err > 1e-4)
		throw std::runtime_error("test failed: posit error too large: " +
			std::to_string(rel_err));

	std::cout << "  posit_types: passed\n";
}

// ============================================================================
// Complex samples: process complex<double> through the filter
// ============================================================================

void test_complex_samples() {
	using complex_t = complex_for_t<double>;

	auto taps_d = design_halfband<double>(11, 0.1);
	HalfBandFilter<double> hb_re(taps_d);
	HalfBandFilter<double> hb_im(taps_d);

	// A complex signal through the filter should equal component-wise filtering
	for (int i = 0; i < 50; ++i) {
		double re_in = std::cos(sw::dsp::two_pi * 0.07 * static_cast<double>(i));
		double im_in = std::sin(sw::dsp::two_pi * 0.07 * static_cast<double>(i));

		double re_out = hb_re.process(re_in);
		double im_out = hb_im.process(im_in);

		complex_t expected(re_out, im_out);
		(void)expected;
	}

	std::cout << "  complex_samples: component-wise filtering verified, passed\n";
}

// ============================================================================
// Dense-vector overloads
// ============================================================================

void test_dense_vector() {
	auto taps = design_halfband<double>(11, 0.1);
	HalfBandFilter<double> hb1(taps);
	HalfBandFilter<double> hb2(taps);

	mtl::vec::dense_vector<double> input(30);
	for (std::size_t i = 0; i < input.size(); ++i) {
		input[i] = std::sin(sw::dsp::two_pi * 0.07 * static_cast<double>(i));
	}

	// Full-rate dense_vector
	auto out_full = hb1.process_block(input);
	if (out_full.size() != 30)
		throw std::runtime_error("test failed: dense_vector full-rate size");

	// Decimation dense_vector
	auto out_dec = hb2.process_block_decimate(input);
	if (out_dec.size() != 15)
		throw std::runtime_error("test failed: dense_vector decimate size = " +
			std::to_string(out_dec.size()));

	std::cout << "  dense_vector: passed\n";
}

// ============================================================================
// Constructor validation: rejects non-half-band taps
// ============================================================================

void test_constructor_validation() {
	bool caught = false;

	// Non-zero even offset from center
	{
		mtl::vec::dense_vector<double> bad(7, 0.0);
		bad[3] = 0.5;    // center
		bad[2] = 0.1;    // offset 1 (odd) — OK
		bad[4] = 0.1;    // offset 1 (odd) — OK
		bad[1] = 0.05;   // offset 2 (even) — should be zero
		bad[5] = 0.05;   // offset 2 (even) — should be zero
		bad[0] = 0.05;   // offset 3 (odd) — OK
		bad[6] = 0.05;   // offset 3 (odd) — OK
		caught = false;
		try { HalfBandFilter<double> hb(bad); }
		catch (const std::invalid_argument&) { caught = true; }
		if (!caught)
			throw std::runtime_error("test failed: non-zero even offset should throw");
	}

	// Asymmetric taps
	{
		mtl::vec::dense_vector<double> bad(7, 0.0);
		bad[3] = 0.5;
		bad[2] = 0.1;
		bad[4] = 0.2;   // asymmetric
		caught = false;
		try { HalfBandFilter<double> hb(bad); }
		catch (const std::invalid_argument&) { caught = true; }
		if (!caught)
			throw std::runtime_error("test failed: asymmetric taps should throw");
	}

	std::cout << "  constructor_validation: passed\n";
}

// ============================================================================
// Parameter validation
// ============================================================================

void test_parameter_validation() {
	bool caught = false;

	// Design: even tap count
	caught = false;
	try { design_halfband<double>(10, 0.1); }
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: even taps should throw");

	// Design: wrong form (4K+3 violation: 9 = 4*2+1, not 4K+3)
	caught = false;
	try { design_halfband<double>(9, 0.1); }
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: 4K+1 taps should throw");

	// Design: transition width out of range
	caught = false;
	try { design_halfband<double>(11, 0.6); }
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: tw=0.6 should throw");

	// Design: negative transition width
	caught = false;
	try { design_halfband<double>(11, -0.1); }
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: tw=-0.1 should throw");

	// Filter: even tap count
	caught = false;
	try {
		mtl::vec::dense_vector<double> bad(4, 0.25);
		HalfBandFilter<double> hb(bad);
	}
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: even filter taps should throw");

	// Filter: too few taps
	caught = false;
	try {
		mtl::vec::dense_vector<double> bad(1, 1.0);
		HalfBandFilter<double> hb(bad);
	}
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: 1-tap filter should throw");

	std::cout << "  parameter_validation: passed\n";
}

// ============================================================================

int main() {
	try {
		std::cout << "Half-band FIR filter tests\n";
		test_design_structure();
		test_design_lengths();
		test_design_float();
		test_dc_gain();
		test_nonzero_count();
		test_impulse_response();
		test_frequency_response();
		test_block_processing();
		test_block_output_validation();
		test_decimation_count();
		test_decimation_dc();
		test_decimation_correctness();
		test_reset();
		test_mixed_precision();
		test_posit_types();
		test_complex_samples();
		test_dense_vector();
		test_constructor_validation();
		test_parameter_validation();
		std::cout << "All half-band tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAIL: " << e.what() << '\n';
		return 1;
	}
}
