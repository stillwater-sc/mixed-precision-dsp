// test_halfband.cpp: test half-band FIR filter design, processing, and decimation
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/acquisition/halfband.hpp>
#include <sw/dsp/math/constants.hpp>

#include <universal/number/posit/posit.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sw::dsp;

bool near(double a, double b, double eps = 1e-6) {
	return std::abs(a - b) < eps;
}

// Zero-phase amplitude of an odd-length symmetric filter.
double hb_amplitude(const mtl::vec::dense_vector<double>& h, double f) {
	std::size_t L = (h.size() - 1) / 2;
	double v = static_cast<double>(h[L]);
	for (std::size_t k = 1; k <= L; ++k)
		v += 2.0 * static_cast<double>(h[L + k]) *
		     std::cos(sw::dsp::two_pi * f * static_cast<double>(k));
	return v;
}

// The design's own ripple: the worst |A(f)| over its stopband. For a
// half-band this is also its passband ripple and, by A(0) + A(0.5) = 1,
// exactly its DC gain error when the odd taps are left unscaled.
double hb_ripple(const mtl::vec::dense_vector<double>& h, double transition_width) {
	const double f_start = 0.25 + transition_width / 2;
	double worst = 0.0;
	for (int i = 0; i <= 2000; ++i)
		worst = std::max(worst, std::abs(
		    hb_amplitude(h, f_start + (0.5 - f_start) * i / 2000.0)));
	return worst;
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
	const std::size_t N  = 15;
	const double      tw = 0.1;

	// Default (equiripple): DC gain is 1 -/+ delta. That is forced by the
	// half-band identity A(0) + A(0.5) = 1 with A(0.5) a stopband extremum,
	// so the right assertion ties the DC error to the design's own ripple
	// rather than to a fixed tolerance. (issue #206)
	auto taps = design_halfband<double>(N, tw);
	double sum = 0.0;
	for (std::size_t i = 0; i < taps.size(); ++i) sum += static_cast<double>(taps[i]);

	const double delta = hb_ripple(taps, tw);
	if (!(std::abs(sum - 1.0) <= delta * 1.01))
		throw std::runtime_error("test failed: DC gain = " + std::to_string(sum) +
			", error exceeds the design ripple " + std::to_string(delta));
	// And it really is the ripple, not merely bounded by it.
	if (!near(std::abs(sum - 1.0), delta, delta * 0.01))
		throw std::runtime_error("test failed: DC error " +
			std::to_string(std::abs(sum - 1.0)) + " != ripple " +
			std::to_string(delta));

	// exact_dc_gain=true buys unity DC gain outright.
	auto exact = design_halfband<double>(N, tw, true);
	double exact_sum = 0.0;
	for (std::size_t i = 0; i < exact.size(); ++i) exact_sum += static_cast<double>(exact[i]);
	if (!near(exact_sum, 1.0, 1e-12))
		throw std::runtime_error("test failed: exact_dc_gain sum = " +
			std::to_string(exact_sum) + ", expected 1.0");

	std::cout << "  dc_gain: equiripple sum=" << sum << " (ripple " << delta
	          << "), exact sum=" << exact_sum << ", passed\n";
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
	std::array<double, 11> output;
	output[0] = hb.process(1.0);
	for (int i = 1; i < 11; ++i) {
		output[static_cast<std::size_t>(i)] = hb.process(0.0);
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
	const std::size_t N  = 15;
	const double      tw = 0.1;

	// Default (equiripple): settled DC output is the filter's DC gain,
	// 1 -/+ delta.
	auto taps = design_halfband<double>(N, tw);
	const double delta = hb_ripple(taps, tw);
	HalfBandFilter<double> hb(taps);
	std::vector<double> input(200, 1.0);
	auto output = hb.process_block_decimate(std::span<const double>(input));

	std::size_t start = (output.size() > 10) ? (output.size() - 10) : 0;
	for (std::size_t i = start; i < output.size(); ++i) {
		if (!near(output[i], 1.0, delta * 1.01))
			throw std::runtime_error("test failed: decimation DC at " +
				std::to_string(i) + " = " + std::to_string(output[i]) +
				", outside 1 +/- ripple (" + std::to_string(delta) + ")");
	}

	// exact_dc_gain=true passes DC through untouched. The tolerance sits just
	// above the +/-1e-8 alternating dither HalfBandFilter injects for denormal
	// prevention, which is what sets the floor here rather than the design.
	auto exact_taps = design_halfband<double>(N, tw, true);
	HalfBandFilter<double> hb_exact(exact_taps);
	auto exact_out = hb_exact.process_block_decimate(std::span<const double>(input));
	start = (exact_out.size() > 10) ? (exact_out.size() - 10) : 0;
	for (std::size_t i = start; i < exact_out.size(); ++i) {
		if (!near(exact_out[i], 1.0, 1e-7))
			throw std::runtime_error("test failed: exact-DC decimation at " +
				std::to_string(i) + " = " + std::to_string(exact_out[i]));
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
	HalfBandFilter<double, complex_t, complex_t> hb_cx(taps_d);

	// Complex filter output should match component-wise real filtering
	for (int i = 0; i < 50; ++i) {
		double re_in = std::cos(sw::dsp::two_pi * 0.07 * static_cast<double>(i));
		double im_in = std::sin(sw::dsp::two_pi * 0.07 * static_cast<double>(i));

		double re_out = hb_re.process(re_in);
		double im_out = hb_im.process(im_in);

		complex_t x(re_in, im_in);
		complex_t y = hb_cx.process(x);
		if (!near(y.real(), re_out, 1e-12) ||
		    !near(y.imag(), im_out, 1e-12))
			throw std::runtime_error("test failed: complex filtering mismatch at " +
				std::to_string(i));
	}

	std::cout << "  complex_samples: passed\n";
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
// Issue #203 regression: stopband attenuation must grow with length
// ============================================================================

// design_halfband() shares the Remez code path, and inherited its failure to
// produce equiripple filters: attenuation plateaued near 20 dB regardless of
// length and then collapsed, reaching -24.7 dB (stopband ABOVE passband) at
// 127 taps with a 0.15 transition. Both monotonicity properties are pinned
// here — more taps must reject more, and so must a wider transition band.
void test_stopband_scaling() {
	auto stopband_db = [](const mtl::vec::dense_vector<double>& taps, double f_start) {
		double worst = 0.0;
		for (int i = 0; i <= 1000; ++i) {
			double f = f_start + (0.5 - f_start) * i / 1000.0;
			double re = 0.0, im = 0.0;
			for (std::size_t n = 0; n < taps.size(); ++n) {
				double angle = sw::dsp::two_pi * f * static_cast<double>(n);
				re += static_cast<double>(taps[n]) * std::cos(angle);
				im -= static_cast<double>(taps[n]) * std::sin(angle);
			}
			worst = std::max(worst, std::sqrt(re * re + im * im));
		}
		return -20.0 * std::log10(std::max(worst, 1e-300));
	};

	// More taps at a fixed transition width must reject more.
	const double tw = 0.10;
	double prev_db = 0.0;
	for (std::size_t N : {23u, 31u, 51u}) {
		double atten = stopband_db(design_halfband<double>(N, tw), 0.25 + tw / 2);
		if (!(atten > prev_db))
			throw std::runtime_error("test failed: halfband N=" + std::to_string(N) +
				" attenuation " + std::to_string(atten) +
				" dB did not improve on " + std::to_string(prev_db));
		prev_db = atten;
	}
	// The plateau was ~20 dB; 51 taps at tw=0.10 now reaches ~81 dB.
	if (!(prev_db > 60.0))
		throw std::runtime_error("test failed: halfband N=51 tw=0.10 attenuation " +
			std::to_string(prev_db) + " dB, expected > 60");

	// A wider transition band at fixed length must also reject more.
	prev_db = 0.0;
	for (double w : {0.05, 0.10, 0.15}) {
		double atten = stopband_db(design_halfband<double>(31, w), 0.25 + w / 2);
		if (!(atten > prev_db))
			throw std::runtime_error("test failed: halfband tw=" + std::to_string(w) +
				" attenuation " + std::to_string(atten) +
				" dB did not improve on " + std::to_string(prev_db));
		prev_db = atten;
	}

	std::cout << "  stopband_scaling: passed (31 taps, tw=0.15 -> " +
	             std::to_string(prev_db) + " dB)\n";
}

// ============================================================================
// Issue #206: exact DC gain vs. full equiripple
// ============================================================================

// design_halfband() rescales the odd-offset taps so the DC gain is exactly 1,
// which costs ~6 dB of stopband. The two properties are mutually exclusive:
// the half-band structure forces A(0) + A(0.5) = 1 identically, and A(0.5) is
// a stopband extremum, so an equiripple half-band has A(0) = 1 -/+ delta.
// exact_dc_gain=false returns the untouched equiripple design.
void test_exact_dc_gain_tradeoff() {
	auto amplitude = [](const mtl::vec::dense_vector<double>& h, double f) {
		std::size_t L = (h.size() - 1) / 2;
		double v = static_cast<double>(h[L]);
		for (std::size_t k = 1; k <= L; ++k)
			v += 2.0 * static_cast<double>(h[L + k]) *
			     std::cos(sw::dsp::two_pi * f * static_cast<double>(k));
		return v;
	};
	auto stopband_delta = [&](const mtl::vec::dense_vector<double>& h, double f_start) {
		double worst = 0.0;
		for (int i = 0; i <= 2000; ++i)
			worst = std::max(worst, std::abs(
			    amplitude(h, f_start + (0.5 - f_start) * i / 2000.0)));
		return worst;
	};

	for (auto spec : {std::pair<std::size_t, double>{31, 0.10},
	                  std::pair<std::size_t, double>{51, 0.10},
	                  std::pair<std::size_t, double>{51, 0.05}}) {
		const std::size_t N  = spec.first;
		const double      tw = spec.second;
		const double      se = 0.25 + tw / 2;

		auto forced = design_halfband<double>(N, tw, true);
		auto ripple = design_halfband<double>(N, tw);            // default: false

		const double d_forced = stopband_delta(forced, se);
		const double d_ripple = stopband_delta(ripple, se);

		// The equiripple design must reject strictly better.
		if (!(d_ripple < d_forced))
			throw std::runtime_error("test failed: equiripple half-band N=" +
				std::to_string(N) + " did not beat the DC-normalized one");

		// And by close to a factor of two — the ~6 dB the normalization costs.
		const double gain_db = 20.0 * std::log10(d_forced / d_ripple);
		if (!(gain_db > 4.0 && gain_db < 8.0))
			throw std::runtime_error("test failed: expected ~6 dB, got " +
				std::to_string(gain_db) + " dB at N=" + std::to_string(N));

		// The forced design has exactly unity DC gain...
		if (!near(amplitude(forced, 0.0), 1.0, 1e-9))
			throw std::runtime_error("test failed: forced DC gain = " +
				std::to_string(amplitude(forced, 0.0)));

		// ...and the equiripple one is off by exactly delta, no more.
		const double dc_err = std::abs(amplitude(ripple, 0.0) - 1.0);
		if (!(dc_err <= d_ripple * 1.01))
			throw std::runtime_error("test failed: equiripple DC error " +
				std::to_string(dc_err) + " exceeds its own ripple " +
				std::to_string(d_ripple));

		// The identity that makes the two mutually exclusive.
		const double identity = amplitude(ripple, 0.0) + amplitude(ripple, 0.5);
		if (!near(identity, 1.0, 1e-9))
			throw std::runtime_error("test failed: A(0)+A(0.5) = " +
				std::to_string(identity) + ", must be 1 for a half-band");
	}

	std::cout << "  exact_dc_gain_tradeoff: passed\n";
}

// The half-band structural constraints must survive either mode.
void test_structure_both_modes() {
	for (bool exact : {true, false}) {
		auto taps = design_halfband<double>(31, 0.1, exact);
		std::size_t center = (taps.size() - 1) / 2;

		if (!near(static_cast<double>(taps[center]), 0.5, 1e-12))
			throw std::runtime_error("test failed: center tap must be 0.5");
		for (std::size_t k = 2; k <= center; k += 2) {
			if (taps[center - k] != 0.0 || taps[center + k] != 0.0)
				throw std::runtime_error("test failed: even-offset tap nonzero");
		}
		for (std::size_t k = 1; k <= center; ++k) {
			if (!near(static_cast<double>(taps[center - k]),
			          static_cast<double>(taps[center + k]), 1e-15))
				throw std::runtime_error("test failed: taps not symmetric");
		}
	}
	std::cout << "  structure_both_modes: passed\n";
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
		test_stopband_scaling();
		test_exact_dc_gain_tradeoff();
		test_structure_both_modes();
		std::cout << "All half-band tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAIL: " << e.what() << '\n';
		return 1;
	}
}
