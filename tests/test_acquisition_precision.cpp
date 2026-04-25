// test_acquisition_precision.cpp: tests for the acquisition precision-analysis
// primitives.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/analysis/acquisition_precision.hpp>
#include <sw/dsp/acquisition/cic.hpp>
#include <sw/dsp/acquisition/decimation_chain.hpp>
#include <sw/dsp/acquisition/halfband.hpp>
#include <sw/dsp/acquisition/nco.hpp>
#include <sw/dsp/filter/fir/fir_design.hpp>
#include <sw/dsp/filter/fir/polyphase.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/windows/hamming.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <system_error>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <universal/number/posit/posit.hpp>

using namespace sw::dsp;
using namespace sw::dsp::analysis;

bool near(double a, double b, double eps = 1e-6) {
	return std::abs(a - b) < eps;
}

// ============================================================================
// ENOB formula
// ============================================================================

void test_enob_formula() {
	// 6.02 N + 1.76 inverted: at 6.02 dB, ENOB ~= 0.703; at 98.09 dB ENOB = 16.
	if (!near(enob_from_snr_db(6.02), (6.02 - 1.76) / 6.02, 1e-12))
		throw std::runtime_error("test failed: enob @ 6.02");
	if (!near(enob_from_snr_db(98.09), (98.09 - 1.76) / 6.02, 1e-12))
		throw std::runtime_error("test failed: enob @ 98.09");
	// 16-bit converter SNR ~= 6.02*16 + 1.76 = 98.08 dB
	if (!near(enob_from_snr_db(98.08), 16.0, 0.01))
		throw std::runtime_error("test failed: enob ~= 16 at 98.08 dB");
	std::cout << "  enob_formula: passed\n";
}

// ============================================================================
// SNR identity and basic measurement
// ============================================================================

void test_snr_identity() {
	mtl::vec::dense_vector<double> a(64);
	mtl::vec::dense_vector<double> b(64);
	for (std::size_t i = 0; i < 64; ++i) {
		a[i] = std::sin(2.0 * pi * static_cast<double>(i) / 64.0);
		b[i] = a[i];
	}
	// Bit-identical: should clip to the underflow guard at 300 dB.
	double s = snr_db(a, b);
	if (!(s >= 299.0))
		throw std::runtime_error("test failed: identity SNR = " + std::to_string(s));
	std::cout << "  snr_identity: " << s << " dB, passed\n";
}

void test_snr_with_known_noise() {
	// Inject 1% RMS noise into a unit-amplitude sine. Expected SNR = 40 dB.
	std::size_t N = 4096;
	mtl::vec::dense_vector<double> ref(N);
	mtl::vec::dense_vector<double> tst(N);
	double noise = 0.01;  // amplitude
	for (std::size_t i = 0; i < N; ++i) {
		double v = std::sin(2.0 * pi * 5.0 * static_cast<double>(i) / static_cast<double>(N));
		ref[i] = v;
		// Deterministic uniform-ish noise via a hash of i.
		double phase = std::sin(static_cast<double>(i) * 0.7853981633974483);
		tst[i] = v + noise * phase;
	}
	double s = snr_db(ref, tst);
	// Expected ~40 dB; allow generous slack for the deterministic-noise PDF.
	if (!(s > 30.0 && s < 50.0))
		throw std::runtime_error("test failed: noisy SNR = " + std::to_string(s) +
			" (expected ~40 dB)");
	std::cout << "  snr_with_known_noise: " << s << " dB, passed\n";
}

void test_snr_size_mismatch_throws() {
	mtl::vec::dense_vector<double> a(10);
	mtl::vec::dense_vector<double> b(11);
	bool threw = false;
	try { snr_db(a, b); }
	catch (const std::invalid_argument&) { threw = true; }
	if (!threw)
		throw std::runtime_error("test failed: size mismatch should throw");
	std::cout << "  snr_size_mismatch_throws: passed\n";
}

// ============================================================================
// NCO SFDR
// ============================================================================

void test_nco_sfdr_double() {
	// A pure double NCO at a non-trivial frequency should have very high
	// SFDR, limited only by FFT leakage. Use a bin-aligned tone for
	// minimal leakage so the spurs stay well below the peak.
	const std::size_t N = 4096;
	const double fs = static_cast<double>(N);   // pick fs so f_tone is on a bin
	const double f_tone = 137.0;                // 137 cycles in 4096 samples
	NCO<double> nco(f_tone, fs);
	double sfdr = measure_nco_sfdr_db(nco, N);
	if (!(sfdr > 80.0))
		throw std::runtime_error("test failed: double NCO SFDR = " +
			std::to_string(sfdr) + " dB (expected > 80)");
	std::cout << "  nco_sfdr_double: " << sfdr << " dB, passed\n";
}

void test_nco_sfdr_zero_size_throws() {
	NCO<double> nco(1.0, 1024.0);
	bool threw = false;
	try { measure_nco_sfdr_db(nco, 0); }
	catch (const std::invalid_argument&) { threw = true; }
	if (!threw)
		throw std::runtime_error("test failed: fft_size=0 should throw");
	std::cout << "  nco_sfdr_zero_size_throws: passed\n";
}

void test_nco_sfdr_huge_size_throws_overflow() {
	// Lock in the round-2 fix: a fft_size near size_t max would have
	// previously wrapped N to 0 and spun forever.
	NCO<double> nco(1.0, 1024.0);
	bool threw = false;
	try {
		measure_nco_sfdr_db(nco, std::numeric_limits<std::size_t>::max());
	}
	catch (const std::overflow_error&) { threw = true; }
	if (!threw)
		throw std::runtime_error("test failed: huge fft_size should throw overflow_error");
	std::cout << "  nco_sfdr_huge_size_throws_overflow: passed\n";
}

void test_nco_sfdr_low_bin_circular_guard() {
	// Tone at bin 1 — wrap-adjacent bin N-1 must be excluded by the
	// circular guard. With the previous linear-distance code, bin N-1
	// would slip through and the SFDR floor would collapse.
	const std::size_t N = 4096;
	const double fs = static_cast<double>(N);
	const double f_tone = 1.0;  // exactly bin 1
	NCO<double> nco(f_tone, fs);
	double sfdr = measure_nco_sfdr_db(nco, N, /*guard_bins=*/2);
	if (!(sfdr > 80.0))
		throw std::runtime_error("test failed: low-bin SFDR = " +
			std::to_string(sfdr) +
			" dB (expected > 80; circular guard not working?)");
	std::cout << "  nco_sfdr_low_bin_circular_guard: " << sfdr << " dB, passed\n";
}

void test_nco_sfdr_posit() {
	using posit_t = sw::universal::posit<32, 2>;
	const std::size_t N = 4096;
	const posit_t fs(static_cast<double>(N));
	const posit_t f_tone(137.0);
	NCO<posit_t> nco(f_tone, fs);
	double sfdr = measure_nco_sfdr_db(nco, N);
	// Posit<32,2> tapered precision near unity is ~28 bits; theoretical
	// SFDR of ~6.02*28 = ~168 dB. FFT spectral leakage and posit rounding
	// cap it lower in practice; 60 dB is a defensive lower bound.
	if (!(sfdr > 60.0))
		throw std::runtime_error("test failed: posit NCO SFDR = " +
			std::to_string(sfdr) + " dB (expected > 60)");
	std::cout << "  nco_sfdr_posit: " << sfdr << " dB, passed\n";
}

// ============================================================================
// CIC bit-growth verification
// ============================================================================

void test_cic_bit_growth_dc() {
	// All-ones DC input is the worst case. After settling, the CIC output
	// converges to (R*D)^M = 64 for R=4, M=3, D=1. So observed_bits should
	// match theoretical = 3 * ceil(log2(4)) = 6 bits.
	const int R = 4, M = 3;
	CICDecimator<double> cic(R, M);
	std::vector<double> input(256, 1.0);
	auto report = check_cic_bit_growth(cic,
		std::span<const double>(input.data(), input.size()));

	const int expected_theoretical = M * static_cast<int>(std::ceil(std::log2(double(R))));
	if (report.theoretical_bits != expected_theoretical)
		throw std::runtime_error("test failed: theoretical_bits = " +
			std::to_string(report.theoretical_bits) +
			", expected " + std::to_string(expected_theoretical));
	if (!report.within_theory)
		throw std::runtime_error("test failed: observed " +
			std::to_string(report.observed_bits) +
			" exceeds theoretical " + std::to_string(report.theoretical_bits));
	// Max output for DC=1 should be exactly (R*D)^M = 64.
	const double expected_peak = std::pow(double(R), double(M));
	if (std::abs(report.max_abs_output - expected_peak) > 1e-9)
		throw std::runtime_error("test failed: peak " +
			std::to_string(report.max_abs_output) +
			", expected " + std::to_string(expected_peak));
	std::cout << "  cic_bit_growth_dc: theoretical=" << report.theoretical_bits
	          << " bits, observed=" << report.observed_bits
	          << " bits, peak=" << report.max_abs_output
	          << ", passed\n";
}

// ============================================================================
// Per-stage noise budget on a DecimationChain
// ============================================================================

void test_per_stage_noise_via_snr() {
	// Build two chains with identical structure (CIC-4 -> half-band -> FIR-2)
	// and identical taps, but one uses double end-to-end and the other uses
	// posit<32,2>. Compare outputs via snr_db; the cross-precision SNR
	// quantifies the posit chain's deviation from the ideal.
	using posit_t = sw::universal::posit<32, 2>;

	const double fs = 1'000'000.0;
	const std::size_t R_fir = 2;

	auto build_double_chain = [&]() {
		CICDecimator<double> cic(4, 2);
		auto hb_taps = design_halfband<double>(15, 0.15);
		HalfBandFilter<double> hb(hb_taps);
		auto win = hamming_window<double>(17);
		auto fir_taps = design_fir_lowpass<double>(17, 0.2, win);
		PolyphaseDecimator<double> pf(fir_taps, R_fir);
		return DecimationChain<double, CICDecimator<double>,
		                                 HalfBandFilter<double>,
		                                 PolyphaseDecimator<double>>(fs, cic, hb, pf);
	};

	auto build_posit_chain = [&]() {
		CICDecimator<posit_t> cic(4, 2);
		auto hb_d = design_halfband<double>(15, 0.15);
		mtl::vec::dense_vector<posit_t> hb_p(hb_d.size());
		for (std::size_t i = 0; i < hb_d.size(); ++i) hb_p[i] = posit_t(hb_d[i]);
		HalfBandFilter<posit_t> hb(hb_p);
		auto win = hamming_window<double>(17);
		auto fir_d = design_fir_lowpass<double>(17, 0.2, win);
		mtl::vec::dense_vector<posit_t> fir_p(fir_d.size());
		for (std::size_t i = 0; i < fir_d.size(); ++i) fir_p[i] = posit_t(fir_d[i]);
		PolyphaseDecimator<posit_t> pf(fir_p, R_fir);
		return DecimationChain<posit_t, CICDecimator<posit_t>,
		                                  HalfBandFilter<posit_t>,
		                                  PolyphaseDecimator<posit_t>>(posit_t(fs), cic, hb, pf);
	};

	auto chain_d = build_double_chain();
	auto chain_p = build_posit_chain();

	const std::size_t N = 1024;
	mtl::vec::dense_vector<double> in_d(N);
	mtl::vec::dense_vector<posit_t> in_p(N);
	for (std::size_t n = 0; n < N; ++n) {
		double v = std::cos(2.0 * pi * 1000.0 * static_cast<double>(n) / fs);
		in_d[n] = v;
		in_p[n] = posit_t(v);
	}
	auto out_d = chain_d.process_block(in_d);
	auto out_p = chain_p.process_block(in_p);

	// Cast posit output to double for the SNR comparison.
	mtl::vec::dense_vector<double> out_p_as_d(out_p.size());
	for (std::size_t i = 0; i < out_p.size(); ++i) out_p_as_d[i] = static_cast<double>(out_p[i]);

	if (out_d.size() != out_p_as_d.size())
		throw std::runtime_error("test failed: chain output size mismatch");

	double s = snr_db(out_d, out_p_as_d);
	double enob = enob_from_snr_db(s);
	// posit<32,2> through this 3-stage chain measures ~98 dB SNR (~16 ENOB).
	// CIC accumulates to gain 64 = (R*D)^M for R=4,M=2,D=1, where posit's
	// tapered precision is somewhat lower than at unity. 90 dB / ~14.7 ENOB
	// gives modest headroom for cross-toolchain libm variance while still
	// asserting "this is a high-quality posit pipeline".
	if (!(s > 90.0))
		throw std::runtime_error("test failed: posit chain SNR = " +
			std::to_string(s) + " (expected > 90)");
	std::cout << "  per_stage_noise_via_snr: SNR=" << s
	          << " dB, ENOB=" << enob << ", passed\n";
}

// ============================================================================
// CSV writer schema check
// ============================================================================

void test_csv_writer_schema() {
	std::vector<AcquisitionPrecisionRow> rows;
	{
		AcquisitionPrecisionRow r;
		r.pipeline = "ddc";
		r.config_name = "uniform_double";
		r.coeff_type = "double";
		r.state_type = "double";
		r.sample_type = "double";
		r.total_bits = 192;
		r.output_snr_db = 250.0;
		r.output_enob = 41.3;
		rows.push_back(r);
	}
	{
		AcquisitionPrecisionRow r;
		r.pipeline = "decim_chain";
		r.config_name = "posit32_uniform";
		r.coeff_type = "posit<32,2>";
		r.state_type = "posit<32,2>";
		r.sample_type = "posit<32,2>";
		r.total_bits = 96;
		r.output_snr_db = 110.0;
		r.output_enob = 17.97;
		r.cic_overflow_margin_bits = 12.0;
		rows.push_back(r);
	}

	// std::filesystem::temp_directory_path() is portable across Linux/macOS/
	// Windows; the steady_clock suffix prevents collisions when ctest runs
	// tests in parallel or when the test process is repeated.
	const auto unique = std::to_string(
		std::chrono::steady_clock::now().time_since_epoch().count());
	const std::string path = (std::filesystem::temp_directory_path() /
		("test_acquisition_precision_" + unique + ".csv")).string();
	write_acquisition_csv(path, rows);

	std::ifstream in(path);
	if (!in)
		throw std::runtime_error("test failed: cannot read written CSV");
	std::string header;
	std::getline(in, header);
	std::string expected_header =
		"pipeline,config_name,coeff_type,state_type,sample_type,"
		"total_bits,output_snr_db,output_enob,nco_sfdr_db,"
		"cic_overflow_margin_bits";
	if (header != expected_header)
		throw std::runtime_error("test failed: CSV header mismatch:\n  got: " +
			header + "\n  exp: " + expected_header);

	int line_count = 0;
	std::string line;
	while (std::getline(in, line)) {
		if (!line.empty()) ++line_count;
	}
	if (line_count != 2)
		throw std::runtime_error("test failed: expected 2 data rows, got " +
			std::to_string(line_count));

	// Verify the second row has the quoted "posit<32,2>" preserved (no
	// special chars to quote there, but the schema must be intact).
	std::ifstream in2(path);
	std::string h, r1, r2;
	std::getline(in2, h);
	std::getline(in2, r1);
	std::getline(in2, r2);
	if (r2.find("posit<32,2>") == std::string::npos)
		throw std::runtime_error("test failed: posit type string not preserved");

	// Close all ifstream handles before unlinking the file. On Linux this
	// isn't strictly necessary (you can unlink open files), but on Windows
	// std::filesystem::remove fails for files that are still held open.
	in.close();
	in2.close();
	std::error_code ec;
	std::filesystem::remove(path, ec);
	if (ec)
		throw std::runtime_error("test failed: temp CSV cleanup failed: " +
			ec.message());
	std::cout << "  csv_writer_schema: passed\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
	try {
		std::cout << "Acquisition Precision Analysis Tests\n";

		test_enob_formula();
		test_snr_identity();
		test_snr_with_known_noise();
		test_snr_size_mismatch_throws();
		test_nco_sfdr_double();
		test_nco_sfdr_zero_size_throws();
		test_nco_sfdr_huge_size_throws_overflow();
		test_nco_sfdr_low_bin_circular_guard();
		test_nco_sfdr_posit();
		test_cic_bit_growth_dc();
		test_per_stage_noise_via_snr();
		test_csv_writer_schema();

		std::cout << "All acquisition precision tests passed.\n";
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << "\n";
		return 1;
	}
	return 0;
}
