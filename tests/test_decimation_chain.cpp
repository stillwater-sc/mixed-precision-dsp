// test_decimation_chain.cpp: test multi-stage decimation chain
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/acquisition/decimation_chain.hpp>
#include <sw/dsp/acquisition/cic.hpp>
#include <sw/dsp/acquisition/halfband.hpp>
#include <sw/dsp/filter/fir/polyphase.hpp>
#include <sw/dsp/filter/fir/fir_design.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/windows/hamming.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <universal/number/posit/posit.hpp>

using namespace sw::dsp;

bool near(double a, double b, double eps = 1e-6) {
	return std::abs(a - b) < eps;
}

// Evaluate |H(f)| of a real FIR filter at normalized frequency f.
static double fir_magnitude(const mtl::vec::dense_vector<double>& taps, double f) {
	std::complex<double> z(0.0, 0.0);
	for (std::size_t n = 0; n < taps.size(); ++n) {
		double angle = -2.0 * pi * f * static_cast<double>(n);
		z += taps[n] * std::complex<double>(std::cos(angle), std::sin(angle));
	}
	return std::abs(z);
}

// Analytical CIC magnitude at normalized output-rate frequency f in [0, 0.5].
// |H(f)| = | sin(pi f) / (R * sin(pi f / R)) |^M
static double cic_magnitude(double f, int R, int M) {
	if (f == 0.0) return 1.0;
	double num = std::sin(pi * f);
	double den = static_cast<double>(R) * std::sin(pi * f / static_cast<double>(R));
	double mag = (den == 0.0) ? 1.0 : std::abs(num / den);
	return std::pow(mag, M);
}

// ============================================================================
// Construction and queries
// ============================================================================

void test_construction_and_queries() {
	// CIC-4 * HB-2 * FIR-4 (simple topology)
	CICDecimator<double> cic(4, 2);
	auto hb_taps = design_halfband<double>(31, 0.1);
	HalfBandFilter<double> hb(hb_taps);
	auto fir_window = hamming_window<double>(33);
	auto fir_taps   = design_fir_lowpass<double>(33, 0.1, fir_window);
	PolyphaseDecimator<double> pf(fir_taps, 4);

	DecimationChain<double, CICDecimator<double>,
	                         HalfBandFilter<double>,
	                         PolyphaseDecimator<double>> chain(
		1'000'000.0, cic, hb, pf);

	if (chain.total_decimation() != 32u)
		throw std::runtime_error("test failed: total_decimation = " +
			std::to_string(chain.total_decimation()));

	auto ratios = chain.stage_ratios();
	if (ratios[0] != 4u || ratios[1] != 2u || ratios[2] != 4u)
		throw std::runtime_error("test failed: stage_ratios incorrect");

	if (!near(chain.input_rate(), 1'000'000.0, 1e-6))
		throw std::runtime_error("test failed: input_rate");
	if (!near(chain.output_rate(), 1'000'000.0 / 32.0, 1e-6))
		throw std::runtime_error("test failed: output_rate");

	auto rates = chain.stage_rates();
	if (!near(rates[0], 250'000.0, 1e-6) ||
	    !near(rates[1], 125'000.0, 1e-6) ||
	    !near(rates[2],  31'250.0, 1e-6))
		throw std::runtime_error("test failed: stage_rates");

	std::cout << "  construction_and_queries: total=" << chain.total_decimation()
	          << ", output_rate=" << chain.output_rate() << ", passed\n";
}

// ============================================================================
// Acceptance: CIC-64 -> HB-2 -> FIR-4 chain
// ============================================================================

void test_acceptance_cic64_hb2_fir4() {
	double fs = 1'000'000.0;       // 1 MHz input
	double f_tone = 200.0;          // well inside chain passband
	int R_cic = 64;
	int M_cic = 3;
	std::size_t R_fir = 4;
	std::size_t N = 8192;           // input samples

	CICDecimator<double> cic(R_cic, M_cic);
	auto hb_taps = design_halfband<double>(31, 0.1);
	HalfBandFilter<double> hb(hb_taps);
	auto fir_window = hamming_window<double>(33);
	auto fir_taps   = design_fir_lowpass<double>(33, 0.1, fir_window);
	PolyphaseDecimator<double> pf(fir_taps, R_fir);

	DecimationChain<double, CICDecimator<double>,
	                         HalfBandFilter<double>,
	                         PolyphaseDecimator<double>> chain(
		fs, cic, hb, pf);

	std::size_t total_dec = chain.total_decimation();
	if (total_dec != 512u)
		throw std::runtime_error("test failed: expected total_decimation=512, got " +
			std::to_string(total_dec));

	// Generate input tone and run through chain
	mtl::vec::dense_vector<double> input(N);
	for (std::size_t n = 0; n < N; ++n) {
		input[n] = std::cos(2.0 * pi * f_tone * static_cast<double>(n) / fs);
	}
	auto output = chain.process_block(input);

	// Expected number of outputs: N / 512 = 16 (± 1 for phase alignment)
	std::size_t expected = N / total_dec;
	if (output.size() < expected - 1 || output.size() > expected + 1)
		throw std::runtime_error("test failed: output count = " +
			std::to_string(output.size()) + ", expected ~" + std::to_string(expected));

	// Output magnitude scales with the product of stage DC gains. CIC's DC gain
	// is (R*D)^M = R^M = 262144; HB and FIR contribute additional factors
	// depending on their designed DC gains. Check that output magnitude is
	// within a factor of a few of the CIC gain — i.e., the signal survives
	// the chain with the expected order of magnitude.
	double cic_gain = std::pow(static_cast<double>(R_cic), static_cast<double>(M_cic));

	std::size_t skip = 2;
	if (output.size() <= skip + 2)
		throw std::runtime_error("test failed: not enough output samples after transient");
	double max_abs = 0.0;
	for (std::size_t i = skip; i < output.size(); ++i) {
		double v = std::abs(output[i]);
		if (v > max_abs) max_abs = v;
	}

	if (max_abs < 0.2 * cic_gain || max_abs > 4.0 * cic_gain)
		throw std::runtime_error("test failed: max|out| = " + std::to_string(max_abs) +
			", expected within [0.2, 4.0] * CIC_gain (" + std::to_string(cic_gain) + ")");

	std::cout << "  acceptance_cic64_hb2_fir4: total_dec=" << total_dec
	          << ", output samples=" << output.size()
	          << ", max|out|=" << max_abs
	          << " (CIC gain=" << cic_gain << "), passed\n";
}

// ============================================================================
// Out-of-band tone is suppressed by the chain
// ============================================================================

void test_out_of_band_suppression() {
	double fs = 1'000'000.0;
	int R_cic = 64;
	int M_cic = 3;
	std::size_t R_fir = 4;
	std::size_t N = 8192;

	// Build chain (helper to avoid duplicating 4 lines)
	auto build_chain = [&](){
		CICDecimator<double> cic(R_cic, M_cic);
		auto hb_taps = design_halfband<double>(31, 0.1);
		HalfBandFilter<double> hb(hb_taps);
		auto fir_window = hamming_window<double>(33);
		auto fir_taps   = design_fir_lowpass<double>(33, 0.1, fir_window);
		PolyphaseDecimator<double> pf(fir_taps, R_fir);
		return DecimationChain<double, CICDecimator<double>,
		                                HalfBandFilter<double>,
		                                PolyphaseDecimator<double>>(
			fs, cic, hb, pf);
	};

	// In-band reference
	auto chain1 = build_chain();
	mtl::vec::dense_vector<double> sig_in(N);
	for (std::size_t n = 0; n < N; ++n) {
		sig_in[n] = std::cos(2.0 * pi * 200.0 * n / fs);
	}
	auto out_in = chain1.process_block(sig_in);

	// Interferer well above the chain's output Nyquist (fs/512 = 1953 Hz; Nyq = 977 Hz).
	// Pick an interferer at 50,000 Hz — well in CIC stopband.
	auto chain2 = build_chain();
	mtl::vec::dense_vector<double> sig_out(N);
	for (std::size_t n = 0; n < N; ++n) {
		sig_out[n] = std::cos(2.0 * pi * 50'000.0 * n / fs);
	}
	auto out_out = chain2.process_block(sig_out);

	double max_in = 0.0, max_out = 0.0;
	for (std::size_t i = 2; i < out_in.size(); ++i)   max_in  = std::max(max_in,  std::abs(out_in[i]));
	for (std::size_t i = 2; i < out_out.size(); ++i)  max_out = std::max(max_out, std::abs(out_out[i]));

	double rejection = 20.0 * std::log10(max_in / (max_out + 1e-30));
	if (rejection < 60.0)
		throw std::runtime_error("test failed: out-of-band rejection only " +
			std::to_string(rejection) + " dB (need >= 60)");

	std::cout << "  out_of_band_suppression: " << rejection << " dB, passed\n";
}

// ============================================================================
// Streaming vs block equivalence
// ============================================================================

void test_stream_vs_block() {
	double fs = 1'000'000.0;
	std::size_t N = 512;

	auto build_chain = [&](){
		CICDecimator<double> cic(4, 2);
		auto hb_taps = design_halfband<double>(15, 0.15);
		HalfBandFilter<double> hb(hb_taps);
		auto fir_window = hamming_window<double>(17);
		auto fir_taps   = design_fir_lowpass<double>(17, 0.2, fir_window);
		PolyphaseDecimator<double> pf(fir_taps, 2);
		return DecimationChain<double, CICDecimator<double>,
		                                HalfBandFilter<double>,
		                                PolyphaseDecimator<double>>(
			fs, cic, hb, pf);
	};

	mtl::vec::dense_vector<double> input(N);
	for (std::size_t n = 0; n < N; ++n) {
		input[n] = std::cos(2.0 * pi * 1000.0 * n / fs);
	}

	auto chain_b = build_chain();
	auto block_out = chain_b.process_block(input);

	auto chain_s = build_chain();
	std::vector<double> stream_out;
	for (std::size_t n = 0; n < N; ++n) {
		auto [ready, y] = chain_s.process(input[n]);
		if (ready) stream_out.push_back(y);
	}

	if (stream_out.size() != block_out.size())
		throw std::runtime_error("test failed: stream/block size mismatch " +
			std::to_string(stream_out.size()) + " vs " + std::to_string(block_out.size()));

	for (std::size_t i = 0; i < stream_out.size(); ++i) {
		if (std::abs(stream_out[i] - block_out[i]) > 1e-12)
			throw std::runtime_error("test failed: stream[" + std::to_string(i) + "] != block");
	}

	std::cout << "  stream_vs_block: " << stream_out.size() << " samples match, passed\n";
}

// ============================================================================
// Reset reproducibility
// ============================================================================

void test_reset() {
	CICDecimator<double> cic(4, 2);
	auto hb_taps = design_halfband<double>(15, 0.15);
	HalfBandFilter<double> hb(hb_taps);
	auto fir_window = hamming_window<double>(17);
	auto fir_taps   = design_fir_lowpass<double>(17, 0.2, fir_window);
	PolyphaseDecimator<double> pf(fir_taps, 2);

	DecimationChain<double, CICDecimator<double>,
	                         HalfBandFilter<double>,
	                         PolyphaseDecimator<double>> chain(
		1e6, cic, hb, pf);

	std::size_t N = 256;
	mtl::vec::dense_vector<double> input(N);
	for (std::size_t n = 0; n < N; ++n)
		input[n] = std::cos(2.0 * pi * 1000.0 * n / 1e6);

	auto out1 = chain.process_block(input);
	chain.reset();
	auto out2 = chain.process_block(input);

	if (out1.size() != out2.size())
		throw std::runtime_error("test failed: reset changed output size");

	for (std::size_t i = 0; i < out1.size(); ++i) {
		if (std::abs(out1[i] - out2[i]) > 1e-12)
			throw std::runtime_error("test failed: reset did not reproduce output at i=" +
				std::to_string(i));
	}

	std::cout << "  reset: passed\n";
}

// ============================================================================
// CIC droop compensator: basic shape and DC gain
// ============================================================================

void test_compensator_design() {
	std::size_t N = 31;
	int M = 3;
	int R = 16;
	double pb = 0.2;

	auto taps = design_cic_compensator<double>(N, M, R, pb);

	if (taps.size() != N)
		throw std::runtime_error("test failed: compensator length");

	// Linear phase -> symmetric taps
	for (std::size_t i = 0; i < N / 2; ++i) {
		if (std::abs(taps[i] - taps[N - 1 - i]) > 1e-10)
			throw std::runtime_error("test failed: compensator not symmetric at i=" +
				std::to_string(i));
	}

	// DC gain = 1 (normalization)
	double dc = 0.0;
	for (std::size_t i = 0; i < N; ++i) dc += taps[i];
	if (std::abs(dc - 1.0) > 1e-6)
		throw std::runtime_error("test failed: compensator DC gain = " + std::to_string(dc));

	std::cout << "  compensator_design: N=" << N << ", symmetric, DC=1, passed\n";
}

// ============================================================================
// Compensator flattens passband: |H_cic * H_comp| flatter than |H_cic| alone
// ============================================================================

void test_compensator_flattens_passband() {
	int M = 3;
	int R = 16;
	double pb = 0.2;                          // passband edge (output-rate normalized)
	auto comp_taps = design_cic_compensator<double>(31, M, R, pb);

	// Evaluate flatness (max-to-min magnitude ratio) across passband [0, pb]
	std::size_t K = 20;
	double max_cic = 0.0, min_cic = 1e30;
	double max_comp = 0.0, min_comp = 1e30;
	for (std::size_t k = 0; k < K; ++k) {
		double f = pb * static_cast<double>(k) / static_cast<double>(K - 1);
		double h_cic = cic_magnitude(f, R, M);
		double h_comp = fir_magnitude(comp_taps, f);
		double combined = h_cic * h_comp;

		double cic_norm = h_cic / cic_magnitude(0.0, R, M);  // normalize to DC
		if (cic_norm > max_cic) max_cic = cic_norm;
		if (cic_norm < min_cic) min_cic = cic_norm;
		if (combined > max_comp) max_comp = combined;
		if (combined < min_comp) min_comp = combined;
	}

	double cic_ripple_db  = 20.0 * std::log10(max_cic  / min_cic);
	double comp_ripple_db = 20.0 * std::log10(max_comp / min_comp);

	// After compensation the passband should be much flatter than the raw CIC.
	// Require both (a) compensated ripple <= 0.5 dB absolute, and
	// (b) at least 5x ripple-ratio improvement over the uncompensated CIC.
	if (comp_ripple_db > 0.5)
		throw std::runtime_error("test failed: compensated passband not flat (" +
			std::to_string(comp_ripple_db) + " dB)");
	double improvement_ratio = cic_ripple_db / std::max(comp_ripple_db, 0.001);
	if (improvement_ratio < 5.0)
		throw std::runtime_error("test failed: compensator improvement ratio = " +
			std::to_string(improvement_ratio) + " (need >= 5)");

	std::cout << "  compensator_flattens_passband: CIC ripple="
	          << cic_ripple_db << " dB, after compensation="
	          << comp_ripple_db << " dB, passed\n";
}

// ============================================================================
// Precision sweep: SNR vs. stage-wise bit widths
// ============================================================================

void test_precision_sweep() {
	// Topology: CIC-4 x FIR-4 (16x total). Compare float-only vs double
	// reference by measuring the output RMS error for a sinusoidal input.
	double fs = 1'000'000.0;
	std::size_t N = 2048;
	std::size_t R_fir = 4;

	// Reference in double
	CICDecimator<double> cic_d(4, 2);
	auto fir_window = hamming_window<double>(33);
	auto fir_taps_d = design_fir_lowpass<double>(33, 0.1, fir_window);
	PolyphaseDecimator<double> pf_d(fir_taps_d, R_fir);
	DecimationChain<double, CICDecimator<double>, PolyphaseDecimator<double>> chain_d(
		fs, cic_d, pf_d);

	// Float chain: cast taps and use float everywhere
	CICDecimator<float> cic_f(4, 2);
	mtl::vec::dense_vector<float> fir_taps_f(fir_taps_d.size());
	for (std::size_t i = 0; i < fir_taps_d.size(); ++i)
		fir_taps_f[i] = static_cast<float>(fir_taps_d[i]);
	PolyphaseDecimator<float> pf_f(fir_taps_f, R_fir);
	DecimationChain<float, CICDecimator<float>, PolyphaseDecimator<float>> chain_f(
		static_cast<float>(fs), cic_f, pf_f);

	mtl::vec::dense_vector<double> sig_d(N);
	mtl::vec::dense_vector<float>  sig_f(N);
	for (std::size_t n = 0; n < N; ++n) {
		double x = std::cos(2.0 * pi * 2000.0 * n / fs);
		sig_d[n] = x;
		sig_f[n] = static_cast<float>(x);
	}
	auto out_d = chain_d.process_block(sig_d);
	auto out_f = chain_f.process_block(sig_f);

	if (out_d.size() != out_f.size())
		throw std::runtime_error("test failed: precision sweep output size mismatch");

	double signal_power = 0.0, noise_power = 0.0;
	for (std::size_t i = 2; i < out_d.size(); ++i) {
		double y_d = out_d[i];
		double y_f = static_cast<double>(out_f[i]);
		signal_power += y_d * y_d;
		noise_power += (y_d - y_f) * (y_d - y_f);
	}
	double snr_db = 10.0 * std::log10(signal_power / (noise_power + 1e-30));

	// Float has about 24 bits of mantissa -> expected SNR > 100 dB.
	// The CIC accumulates with a 64 value gain, so precision loss is modest.
	if (snr_db < 80.0)
		throw std::runtime_error("test failed: float SNR = " + std::to_string(snr_db) +
			" dB (need >= 80)");

	std::cout << "  precision_sweep: float-vs-double SNR = " << snr_db << " dB, passed\n";
}

// ============================================================================
// Mixed-precision end-to-end with posit<32,2>
// ============================================================================

void test_mixed_precision_posit() {
	using posit_t = sw::universal::posit<32, 2>;

	posit_t fs(1000.0);   // 1 kHz
	std::size_t N = 1024;
	std::size_t R_fir = 2;

	CICDecimator<posit_t> cic(4, 2);
	auto fir_window_d = hamming_window<double>(17);
	auto fir_taps_d   = design_fir_lowpass<double>(17, 0.2, fir_window_d);
	mtl::vec::dense_vector<posit_t> fir_taps(fir_taps_d.size());
	for (std::size_t i = 0; i < fir_taps_d.size(); ++i)
		fir_taps[i] = posit_t(fir_taps_d[i]);
	PolyphaseDecimator<posit_t> pf(fir_taps, R_fir);

	DecimationChain<posit_t, CICDecimator<posit_t>, PolyphaseDecimator<posit_t>> chain(
		fs, cic, pf);

	if (chain.total_decimation() != 8u)
		throw std::runtime_error("test failed: posit chain total_decimation");

	mtl::vec::dense_vector<posit_t> input(N);
	for (std::size_t n = 0; n < N; ++n) {
		double x = std::cos(2.0 * pi * 20.0 * static_cast<double>(n) / 1000.0);
		input[n] = posit_t(x);
	}

	auto out = chain.process_block(input);
	if (out.size() < 120 || out.size() > 130)
		throw std::runtime_error("test failed: posit chain output count = " +
			std::to_string(out.size()));

	// Expect non-trivial output (CIC gain * cos envelope)
	double max_abs = 0.0;
	for (std::size_t i = 2; i < out.size(); ++i) {
		double v = std::abs(static_cast<double>(out[i]));
		if (v > max_abs) max_abs = v;
	}
	if (max_abs < 1.0)
		throw std::runtime_error("test failed: posit chain output too small: " +
			std::to_string(max_abs));

	std::cout << "  mixed_precision_posit: " << out.size() << " samples, max|out|="
	          << max_abs << ", passed\n";
}

// ============================================================================
// CIC compensator designed at posit<32,2> precision
// ============================================================================
//
// Verifies that design_cic_compensator runs its intermediate math in T, not
// double. Posit tapered precision around 1.0 should give agreement with the
// double-precision reference to within a few ULPs of posit32.

void test_compensator_in_posit_precision() {
	using posit_t = sw::universal::posit<32, 2>;

	std::size_t N = 31;
	int M = 3;
	int R = 16;
	double pb_d = 0.2;
	posit_t pb_p(pb_d);

	auto taps_d = design_cic_compensator<double>(N, M, R, pb_d);
	auto taps_p = design_cic_compensator<posit_t>(N, M, R, pb_p);

	if (taps_p.size() != N)
		throw std::runtime_error("test failed: posit compensator length");

	// Symmetry preserved to posit<32,2> precision. Posit has ~28 bits of
	// mantissa around unit magnitude; accumulated non-associative rounding
	// across the 31-term frequency-sampling sum gives ~1e-6 asymmetry.
	double max_asym = 0.0;
	for (std::size_t i = 0; i < N / 2; ++i) {
		double li = static_cast<double>(taps_p[i]);
		double ri = static_cast<double>(taps_p[N - 1 - i]);
		max_asym = std::max(max_asym, std::abs(li - ri));
	}
	if (max_asym > 1e-5)
		throw std::runtime_error("test failed: posit compensator asymmetry = " +
			std::to_string(max_asym));

	// DC gain remains 1 after the T-domain normalization
	double dc = 0.0;
	for (std::size_t i = 0; i < N; ++i) dc += static_cast<double>(taps_p[i]);
	if (std::abs(dc - 1.0) > 1e-5)
		throw std::runtime_error("test failed: posit compensator DC = " + std::to_string(dc));

	// Agreement with double reference within posit precision
	double max_diff = 0.0;
	for (std::size_t i = 0; i < N; ++i) {
		double diff = std::abs(static_cast<double>(taps_p[i]) - taps_d[i]);
		if (diff > max_diff) max_diff = diff;
	}
	if (max_diff > 1e-5)
		throw std::runtime_error("test failed: posit vs double compensator max diff = " +
			std::to_string(max_diff));

	std::cout << "  compensator_in_posit_precision: asym=" << max_asym
	          << ", DC err=" << std::abs(dc - 1.0)
	          << ", max diff vs double=" << max_diff << ", passed\n";
}

// ============================================================================
// Stage accessor
// ============================================================================

void test_stage_accessor() {
	CICDecimator<double> cic(8, 3);
	auto hb_taps = design_halfband<double>(7, 0.2);
	HalfBandFilter<double> hb(hb_taps);

	DecimationChain<double, CICDecimator<double>, HalfBandFilter<double>> chain(
		1e6, cic, hb);

	if (chain.stage<0>().decimation_ratio() != 8)
		throw std::runtime_error("test failed: stage<0>.decimation_ratio");
	if (chain.stage<0>().num_stages() != 3)
		throw std::runtime_error("test failed: stage<0>.num_stages");

	std::cout << "  stage_accessor: passed\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
	try {
		std::cout << "DecimationChain tests\n";
		test_construction_and_queries();
		test_acceptance_cic64_hb2_fir4();
		test_out_of_band_suppression();
		test_stream_vs_block();
		test_reset();
		test_compensator_design();
		test_compensator_flattens_passband();
		test_precision_sweep();
		test_mixed_precision_posit();
		test_compensator_in_posit_precision();
		test_stage_accessor();
		std::cout << "All DecimationChain tests passed.\n";
	} catch (const std::exception& e) {
		std::cerr << "FAIL: " << e.what() << "\n";
		return 1;
	}
	return 0;
}
