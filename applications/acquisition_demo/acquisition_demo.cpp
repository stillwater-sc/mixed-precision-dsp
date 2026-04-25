// acquisition_demo.cpp: end-to-end acquisition pipeline mixed-precision sweep
//
// Simulates an IF-sampling receiver:
//   - real ADC stream at f_s (configurable bit depth)
//   - Digital Down-Converter (NCO + complex mixer + polyphase decimation)
//   - quality measured via the analysis/acquisition_precision.hpp primitives
//
// Sweeps across number-system configurations and ADC bit depths, writing
// per-configuration quality metrics to stdout (Pareto-style summary table)
// and to acquisition_demo.csv (schema-compatible with precision_sweep.csv
// and acquisition_precision.csv at the identifier columns).
//
// Capstone for the High-Rate Data Acquisition Pipeline epic (#84).
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/acquisition/ddc.hpp>
#include <sw/dsp/analysis/acquisition_precision.hpp>
#include <sw/dsp/filter/fir/fir_design.hpp>
#include <sw/dsp/filter/fir/polyphase.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/windows/hamming.hpp>

#if __has_include(<bit>)
#include <bit>
#endif
#include <sw/universal/number/posit/posit.hpp>
#include <sw/universal/number/cfloat/cfloat.hpp>
#include <sw/universal/number/fixpnt/fixpnt.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace sw::dsp;
using sw::dsp::analysis::AcquisitionPrecisionRow;
using sw::dsp::analysis::write_acquisition_csv;
using sw::dsp::analysis::snr_db;
using sw::dsp::analysis::enob_from_snr_db;

// ============================================================================
// Type aliases for the sweep
// ============================================================================

using p32  = sw::universal::posit<32, 2>;
using p16  = sw::universal::posit<16, 2>;
using cf32 = sw::universal::cfloat<32, 8, std::uint32_t, true, false, false>;
// Q4.28: 4 integer bits (signal range ±8 — ample headroom for the
// post-mixer/FIR baseband) and 28 fractional bits (~28 ENOB ceiling).
// fixpnt<32,16> would give Q16.16 — vastly more integer headroom than the
// signal needs and only 16 fractional bits, which collapses to ~0 ENOB.
using fx32 = sw::universal::fixpnt<32, 28>;

// ============================================================================
// Pipeline parameters — fixed for all configurations
// ============================================================================

namespace params {
constexpr double      kSampleRateHz   = 1e6;     // simulated ADC rate (1 MHz)
constexpr double      kIfFrequencyHz  = 100e3;   // tone we want to capture
constexpr double      kSignalAmp      = 0.5;     // peak signal amplitude
constexpr double      kNoiseRms       = 0.005;   // additive AWGN
constexpr std::size_t kNumSamples     = 4096;    // input length
constexpr std::size_t kDecimation     = 8;       // DDC output rate = fs / 8
constexpr std::size_t kFirTaps        = 65;      // anti-alias FIR length
}  // namespace params

// ============================================================================
// ADC simulation: a real cosine + AWGN, quantized to N bits
// ============================================================================

mtl::vec::dense_vector<double> simulate_adc(int adc_bits, unsigned seed = 0xACDC) {
	mtl::vec::dense_vector<double> samples(params::kNumSamples);
	std::mt19937 rng(seed);
	std::normal_distribution<double> noise(0.0, params::kNoiseRms);

	const double full_scale = 1.0;
	const double levels = static_cast<double>(1 << (adc_bits - 1));
	const double q_step = full_scale / levels;

	for (std::size_t n = 0; n < params::kNumSamples; ++n) {
		const double t = static_cast<double>(n) / params::kSampleRateHz;
		const double clean = params::kSignalAmp *
		                      std::cos(2.0 * pi * params::kIfFrequencyHz * t);
		double noisy = clean + noise(rng);
		// Clamp to [-1, 1] full-scale, then quantize
		noisy = std::clamp(noisy, -full_scale, full_scale);
		const double quantized = std::round(noisy / q_step) * q_step;
		samples[n] = quantized;
	}
	return samples;
}

// ============================================================================
// Run the DDC pipeline at a given (CoeffScalar, StateScalar, SampleScalar)
// configuration. Returns the complex baseband output cast back to double for
// quality comparison against the reference.
// ============================================================================

template <class CoeffScalar, class StateScalar, class SampleScalar>
std::vector<std::complex<double>>
run_ddc_pipeline(const mtl::vec::dense_vector<double>& adc_in_double) {
	using DDC_t = DDC<CoeffScalar, StateScalar, SampleScalar,
	                   PolyphaseDecimator<CoeffScalar, StateScalar, SampleScalar>>;

	// Design taps in double, project to CoeffScalar.
	auto win = hamming_window<double>(params::kFirTaps);
	auto taps_d = design_fir_lowpass<double>(
		params::kFirTaps,
		0.45 / static_cast<double>(params::kDecimation),
		win);
	mtl::vec::dense_vector<CoeffScalar> taps(taps_d.size());
	for (std::size_t i = 0; i < taps_d.size(); ++i)
		taps[i] = static_cast<CoeffScalar>(taps_d[i]);

	PolyphaseDecimator<CoeffScalar, StateScalar, SampleScalar> decim(
		taps, params::kDecimation);
	// DDC only needs the IF / fs ratio. Pass normalized rates so that
	// narrow-integer-range StateScalars (e.g., fixpnt<32,28> with only
	// ±8 representable) don't saturate trying to hold MHz-scale values.
	const double f_norm = params::kIfFrequencyHz / params::kSampleRateHz;
	DDC_t ddc(static_cast<StateScalar>(f_norm),
	          static_cast<StateScalar>(1.0),
	          decim);

	// Project ADC samples into SampleScalar for streaming.
	mtl::vec::dense_vector<SampleScalar> adc_in(adc_in_double.size());
	for (std::size_t i = 0; i < adc_in_double.size(); ++i)
		adc_in[i] = static_cast<SampleScalar>(adc_in_double[i]);

	auto out = ddc.process_block(adc_in);

	// Cast back to double for cross-precision comparison.
	std::vector<std::complex<double>> out_d(out.size());
	for (std::size_t i = 0; i < out.size(); ++i) {
		out_d[i] = std::complex<double>(static_cast<double>(out[i].real()),
		                                 static_cast<double>(out[i].imag()));
	}
	return out_d;
}

// ============================================================================
// Measurement: run reference (uniform double) and a test configuration; compute
// SNR/ENOB on the magnitude streams.
// ============================================================================

template <class CoeffScalar, class StateScalar, class SampleScalar>
AcquisitionPrecisionRow measure_config(
		const std::string& config_name,
		const std::string& coeff_label,
		const std::string& state_label,
		const std::string& sample_label,
		int total_bits,
		const mtl::vec::dense_vector<double>& adc_in,
		const std::vector<std::complex<double>>& reference_out) {
	AcquisitionPrecisionRow row;
	row.pipeline    = "ddc_acquisition";
	row.config_name = config_name;
	row.coeff_type  = coeff_label;
	row.state_type  = state_label;
	row.sample_type = sample_label;
	row.total_bits  = total_bits;
	row.nco_sfdr_db = -1.0;                  // not measured per config here
	row.cic_overflow_margin_bits = -1.0;     // CIC not used in this pipeline

	auto test_out = run_ddc_pipeline<CoeffScalar, StateScalar, SampleScalar>(adc_in);

	// Compare magnitude envelopes — they're the meaningful baseband output of a
	// real-IF DDC. Skip the first few samples to let the polyphase delay line
	// fill (roughly num_taps / decimation = 65/8 ≈ 8 samples).
	const std::size_t skip = 16;
	if (test_out.size() != reference_out.size() || test_out.size() <= skip) {
		row.output_snr_db = 0.0;
		row.output_enob   = 0.0;
		return row;
	}
	std::vector<double> ref_mag(test_out.size() - skip);
	std::vector<double> tst_mag(test_out.size() - skip);
	for (std::size_t i = skip; i < test_out.size(); ++i) {
		ref_mag[i - skip] = std::abs(reference_out[i]);
		tst_mag[i - skip] = std::abs(test_out[i]);
	}
	const double s = snr_db(std::span<const double>(ref_mag.data(), ref_mag.size()),
	                         std::span<const double>(tst_mag.data(), tst_mag.size()));
	row.output_snr_db = s;
	row.output_enob   = enob_from_snr_db(s);
	return row;
}

// ============================================================================
// Sweep across configurations
// ============================================================================

std::vector<AcquisitionPrecisionRow> sweep_configurations(
		const mtl::vec::dense_vector<double>& adc_in) {
	std::vector<AcquisitionPrecisionRow> rows;

	// Reference (uniform double) — also the comparison baseline. Result is
	// trivially +inf SNR but gives the row a place in the table.
	auto reference_out = run_ddc_pipeline<double, double, double>(adc_in);

	// Helper bit-width totals
	auto total = [](int a, int b, int c) { return a + b + c; };

	// Uniform configurations
	rows.push_back(measure_config<double, double, double>(
		"uniform_double", "double", "double", "double",
		total(64, 64, 64), adc_in, reference_out));
	rows.push_back(measure_config<float, float, float>(
		"uniform_float", "float", "float", "float",
		total(32, 32, 32), adc_in, reference_out));
	rows.push_back(measure_config<p32, p32, p32>(
		"uniform_posit32", "posit<32,2>", "posit<32,2>", "posit<32,2>",
		total(32, 32, 32), adc_in, reference_out));
	rows.push_back(measure_config<p16, p16, p16>(
		"uniform_posit16", "posit<16,2>", "posit<16,2>", "posit<16,2>",
		total(16, 16, 16), adc_in, reference_out));
	rows.push_back(measure_config<cf32, cf32, cf32>(
		"uniform_cfloat32", "cfloat<32,8>", "cfloat<32,8>", "cfloat<32,8>",
		total(32, 32, 32), adc_in, reference_out));
	rows.push_back(measure_config<fx32, fx32, fx32>(
		"uniform_fixpnt32", "fixpnt<32,28>", "fixpnt<32,28>", "fixpnt<32,28>",
		total(32, 32, 32), adc_in, reference_out));

	// Mixed-precision: high-precision design / coefficients, narrower runtime
	rows.push_back(measure_config<double, p32, p16>(
		"mixed_double_p32_p16", "double", "posit<32,2>", "posit<16,2>",
		total(64, 32, 16), adc_in, reference_out));
	rows.push_back(measure_config<double, double, float>(
		"mixed_double_double_float", "double", "double", "float",
		total(64, 64, 32), adc_in, reference_out));
	rows.push_back(measure_config<double, p32, float>(
		"mixed_double_p32_float", "double", "posit<32,2>", "float",
		total(64, 32, 32), adc_in, reference_out));
	rows.push_back(measure_config<double, float, float>(
		"mixed_double_float_float", "double", "float", "float",
		total(64, 32, 32), adc_in, reference_out));
	rows.push_back(measure_config<double, fx32, fx32>(
		"mixed_double_fx32_fx32", "double", "fixpnt<32,28>", "fixpnt<32,28>",
		total(64, 32, 32), adc_in, reference_out));

	return rows;
}

// ============================================================================
// Console summary table
// ============================================================================

void print_sweep_table(const std::vector<AcquisitionPrecisionRow>& rows) {
	std::cout << std::left  << std::setw(28) << "Configuration"
	          << std::right << std::setw(8)  << "Bits"
	          << std::right << std::setw(11) << "SNR(dB)"
	          << std::right << std::setw(8)  << "ENOB"
	          << "\n";
	std::cout << std::string(28 + 8 + 11 + 8, '-') << "\n";
	for (const auto& r : rows) {
		std::cout << std::left  << std::setw(28) << r.config_name
		          << std::right << std::setw(8)  << r.total_bits;
		if (r.output_snr_db >= 299.0) {
			std::cout << std::right << std::setw(11) << "inf"
			          << std::right << std::setw(8)  << "ref";
		} else {
			std::cout << std::right << std::setw(11) << std::fixed
			          << std::setprecision(2) << r.output_snr_db
			          << std::right << std::setw(8)  << std::fixed
			          << std::setprecision(2) << r.output_enob;
		}
		std::cout << "\n";
	}
	std::cout << std::flush;
}

// ============================================================================
// ADC bit-depth scan: sanity check that quantization noise dominates correctly
// ============================================================================

void print_adc_scan_header() {
	std::cout << "\n=== ADC bit-depth scan (uniform-double pipeline) ===\n";
	std::cout << std::left  << std::setw(12) << "ADC bits"
	          << std::right << std::setw(11) << "SNR(dB)"
	          << std::right << std::setw(8)  << "ENOB"
	          << "\n";
	std::cout << std::string(12 + 11 + 8, '-') << "\n";
}

void scan_adc_bit_depths(std::vector<AcquisitionPrecisionRow>& rows) {
	// For each ADC bit depth, compare the quantized ADC pipeline output
	// against an unquantized (full-precision) reference. Demonstrates the
	// 6.02*N + 1.76 SNR ceiling from the ADC stage propagating end-to-end.
	auto ideal_in = simulate_adc(64);  // effectively unquantized (q ~ 1e-19)
	auto reference_out = run_ddc_pipeline<double, double, double>(ideal_in);

	print_adc_scan_header();
	for (int bits : {8, 12, 14, 16}) {
		auto adc_in = simulate_adc(bits);
		auto out_d = run_ddc_pipeline<double, double, double>(adc_in);
		const std::size_t skip = 16;
		if (out_d.size() != reference_out.size() || out_d.size() <= skip)
			continue;
		std::vector<double> ref_mag(out_d.size() - skip);
		std::vector<double> tst_mag(out_d.size() - skip);
		for (std::size_t i = skip; i < out_d.size(); ++i) {
			ref_mag[i - skip] = std::abs(reference_out[i]);
			tst_mag[i - skip] = std::abs(out_d[i]);
		}
		double s = snr_db(std::span<const double>(ref_mag.data(), ref_mag.size()),
		                   std::span<const double>(tst_mag.data(), tst_mag.size()));

		AcquisitionPrecisionRow row;
		row.pipeline = "ddc_adc_scan";
		row.config_name = "adc_" + std::to_string(bits) + "bit";
		row.coeff_type = "double";
		row.state_type = "double";
		row.sample_type = "double";
		row.total_bits = bits;
		row.output_snr_db = s;
		row.output_enob = enob_from_snr_db(s);
		row.nco_sfdr_db = -1.0;
		row.cic_overflow_margin_bits = -1.0;
		rows.push_back(row);

		std::cout << std::left  << std::setw(12) << (std::to_string(bits) + "-bit")
		          << std::right << std::setw(11) << std::fixed
		          << std::setprecision(2) << s
		          << std::right << std::setw(8)  << std::fixed
		          << std::setprecision(2) << row.output_enob
		          << "\n";
	}
	std::cout << std::flush;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
	std::cout << "=== Acquisition Pipeline Demo (Issue #93) ===\n";
	std::cout << "Pipeline: real ADC -> DDC(NCO + mixer + polyphase decim x"
	          << params::kDecimation << ")\n";
	std::cout << "Sample rate:    " << params::kSampleRateHz / 1e6 << " MHz\n";
	std::cout << "IF frequency:   " << params::kIfFrequencyHz / 1e3 << " kHz\n";
	std::cout << "Output rate:    "
	          << params::kSampleRateHz / params::kDecimation / 1e3 << " kHz\n";
	std::cout << "Block length:   " << params::kNumSamples << " input samples\n";
	std::cout << "FIR taps:       " << params::kFirTaps << "\n\n";

	// Sweep across number-system configurations at 16-bit ADC
	std::cout << "=== Number-system sweep (16-bit ADC) ===\n";
	auto adc_in = simulate_adc(16);
	auto rows = sweep_configurations(adc_in);
	print_sweep_table(rows);

	// ADC bit-depth scan with the uniform-double pipeline
	scan_adc_bit_depths(rows);

	// CSV export
	const std::string csv_path = (argc > 1)
		? std::string(argv[1])
		: std::string("acquisition_demo.csv");
	write_acquisition_csv(csv_path, rows);
	std::cout << "\nCSV: " << csv_path << " (" << rows.size() << " rows)\n";

	return 0;
}
