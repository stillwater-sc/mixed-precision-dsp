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

#include <sw/dsp/acquisition/cic.hpp>
#include <sw/dsp/acquisition/ddc.hpp>
#include <sw/dsp/acquisition/decimation_chain.hpp>
#include <sw/dsp/acquisition/halfband.hpp>
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
#include <limits>
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

// Pipeline parameters are now mutable so they can be overridden from the
// command line. Defaults are 1 MHz sample rate, 100 kHz IF, total
// decimation 16, with the chain breaking down as DDC↓2 then a CIC↓2 →
// HalfBand↓2 → Polyphase↓2 chain on each I/Q stream — exercising all
// three multistage primitives the issue lists.
struct PipelineParams {
	double      sample_rate_hz     = 1e6;       // simulated ADC rate
	double      if_frequency_hz    = 100e3;     // tone we want to capture
	double      signal_amp         = 0.5;       // peak signal amplitude
	double      noise_rms          = 0.005;     // additive AWGN
	std::size_t num_samples        = 4096;      // input length
	std::size_t ddc_decimation     = 2;         // DDC's internal polyphase decim
	std::size_t cic_ratio          = 2;         // first stage of post-DDC chain
	int         cic_stages         = 2;
	std::size_t poly_decimation    = 2;         // last stage
	std::size_t fir_taps           = 65;        // DDC's anti-alias FIR length
	std::size_t hb_taps            = 11;        // half-band length (4K+3)
	std::size_t chain_fir_taps     = 17;        // post-DDC polyphase FIR length
	std::vector<int> adc_scan_bits = {8, 12, 14, 16};
};
inline PipelineParams params;

// ============================================================================
// ADC simulation: a real cosine + AWGN, quantized to N bits
// ============================================================================

mtl::vec::dense_vector<double> simulate_adc(int adc_bits, unsigned seed = 0xACDC) {
	mtl::vec::dense_vector<double> samples(params.num_samples);
	std::mt19937 rng(seed);
	std::normal_distribution<double> noise(0.0, params.noise_rms);

	// Standard 2's-complement N-bit ADC: codes range [-2^(N-1), 2^(N-1)-1],
	// giving exactly 2^N distinct codes (e.g., 65536 for a 16-bit ADC).
	// Earlier formulation used round-to-nearest with codes [-2^(N-1),
	// +2^(N-1)] which produced 2^N+1 codes (off by one).
	// std::ldexp(1.0, n) computes 2^n as a double for any non-negative n,
	// avoiding the UB of `1 << 63` when the ADC scan calls simulate_adc(64).
	const double half_levels = std::ldexp(1.0, adc_bits - 1);
	const double q_step      = 1.0 / half_levels;
	const double code_max    = half_levels - 1.0;
	const double code_min    = -half_levels;

	for (std::size_t n = 0; n < params.num_samples; ++n) {
		const double t = static_cast<double>(n) / params.sample_rate_hz;
		const double clean = params.signal_amp *
		                      std::cos(2.0 * pi * params.if_frequency_hz * t);
		const double noisy = clean + noise(rng);
		// Use floor (truncate-toward-negative) on (x / q_step) to mirror
		// real ADC behavior: the asymmetry [-2^(N-1), 2^(N-1)-1] is
		// preserved and we get exactly 2^N codes.
		double code = std::floor(noisy / q_step);
		code = std::clamp(code, code_min, code_max);
		samples[n] = code * q_step;
	}
	return samples;
}

// ============================================================================
// Run the DDC pipeline at a given (CoeffScalar, StateScalar, SampleScalar)
// configuration. Returns the complex baseband output cast back to double for
// quality comparison against the reference.
// ============================================================================

// Full multistage acquisition pipeline:
//   ADC (real)
//     -> DDC{NCO + complex mixer + polyphase ↓2}   produces I/Q
//     -> for each of I, Q stream:
//          DecimationChain<CIC ↓2 → HalfBand ↓2 → Polyphase ↓2>
//     -> baseband I/Q at fs / 16
//
// All filter design is done in double and projected to CoeffScalar
// (see the comment block in Stage 1 below). This deliberately
// deviates from the library's design-time-precision invariant so
// that cross-configuration SNR isolates streaming-arithmetic
// precision from filter-design variance.
template <class CoeffScalar, class StateScalar, class SampleScalar>
std::vector<std::complex<double>>
run_pipeline(const mtl::vec::dense_vector<double>& adc_in_double) {
	using DDC_t = DDC<CoeffScalar, StateScalar, SampleScalar,
	                   PolyphaseDecimator<CoeffScalar, StateScalar, SampleScalar>>;

	// --- Stage 1: design DDC anti-alias FIR ---
	//
	// All three filter designs (DDC anti-alias, half-band, polyphase) are
	// done in double and projected to CoeffScalar. The library DOES support
	// design at T (see the T-parameterization audit, issues #111-#116) —
	// but for THIS demo we want the SNR measurement to isolate
	// streaming-arithmetic precision from filter-design precision. If we
	// designed taps at CoeffScalar, the test and the double reference would
	// run through different taps (Remez at posit converges to slightly
	// different values than at double), so SNR would conflate
	// filter-design variance with the arithmetic quality we're trying to
	// measure. fixpnt is a stronger example: design_halfband<fixpnt> trips
	// divide-by-zero in the Remez iteration because fixpnt doesn't have
	// the dynamic range Remez assumes. Design once in double, project for
	// each test config.
	const auto win_d = hamming_window<double>(params.fir_taps);
	const auto ddc_taps_d = design_fir_lowpass<double>(
		params.fir_taps,
		0.45 / static_cast<double>(params.ddc_decimation),
		win_d);
	mtl::vec::dense_vector<CoeffScalar> ddc_taps(ddc_taps_d.size());
	std::transform(ddc_taps_d.begin(), ddc_taps_d.end(), ddc_taps.begin(),
	               [](double d) { return static_cast<CoeffScalar>(d); });
	PolyphaseDecimator<CoeffScalar, StateScalar, SampleScalar> ddc_decim(
		ddc_taps, params.ddc_decimation);

	// DDC only needs the IF / fs ratio. Pass normalized rates so that
	// narrow-integer-range StateScalars (e.g., fixpnt<32,28> with only
	// ±8 representable) don't saturate trying to hold MHz-scale values.
	const double f_norm = params.if_frequency_hz / params.sample_rate_hz;
	DDC_t ddc(static_cast<StateScalar>(f_norm),
	          static_cast<StateScalar>(1.0),
	          ddc_decim);

	// Project ADC samples into SampleScalar and run the DDC.
	mtl::vec::dense_vector<SampleScalar> adc_in(adc_in_double.size());
	std::transform(adc_in_double.begin(), adc_in_double.end(), adc_in.begin(),
	               [](double d) { return static_cast<SampleScalar>(d); });
	const auto ddc_out = ddc.process_block(adc_in);

	// --- Stage 2: build CIC → HalfBand → Polyphase chain ---
	// Both I and Q streams need their own chain instance running in
	// lockstep. Build a "prototype" we can copy.
	using HalfBand_t  = HalfBandFilter<CoeffScalar, StateScalar, SampleScalar>;
	using Poly_t      = PolyphaseDecimator<CoeffScalar, StateScalar, SampleScalar>;
	using CIC_t       = CICDecimator<StateScalar, SampleScalar>;
	using Chain_t     = DecimationChain<SampleScalar, CIC_t, HalfBand_t, Poly_t>;

	const auto hb_taps_d = design_halfband<double>(params.hb_taps, 0.1);
	mtl::vec::dense_vector<CoeffScalar> hb_taps(hb_taps_d.size());
	std::transform(hb_taps_d.begin(), hb_taps_d.end(), hb_taps.begin(),
	               [](double d) { return static_cast<CoeffScalar>(d); });

	const auto poly_win_d  = hamming_window<double>(params.chain_fir_taps);
	const auto poly_taps_d = design_fir_lowpass<double>(
		params.chain_fir_taps,
		0.45 / static_cast<double>(params.poly_decimation),
		poly_win_d);
	mtl::vec::dense_vector<CoeffScalar> poly_taps(poly_taps_d.size());
	std::transform(poly_taps_d.begin(), poly_taps_d.end(), poly_taps.begin(),
	               [](double d) { return static_cast<CoeffScalar>(d); });

	auto build_chain = [&]() {
		CIC_t      cic(static_cast<int>(params.cic_ratio), params.cic_stages);
		HalfBand_t hb(hb_taps);
		Poly_t     poly(poly_taps, params.poly_decimation);
		return Chain_t(static_cast<SampleScalar>(1.0),
		               std::move(cic), std::move(hb), std::move(poly));
	};
	auto chain_i = build_chain();
	auto chain_q = build_chain();

	// --- Stage 3: split DDC output into I/Q streams, decimate each ---
	mtl::vec::dense_vector<SampleScalar> i_stream(ddc_out.size());
	mtl::vec::dense_vector<SampleScalar> q_stream(ddc_out.size());
	for (std::size_t n = 0; n < ddc_out.size(); ++n) {
		i_stream[n] = ddc_out[n].real();
		q_stream[n] = ddc_out[n].imag();
	}
	const auto i_out = chain_i.process_block(i_stream);
	const auto q_out = chain_q.process_block(q_stream);

	// --- Stage 4: cast back to std::complex<double> for SNR comparison ---
	const std::size_t n_out = std::min(i_out.size(), q_out.size());
	std::vector<std::complex<double>> out_d(n_out);
	for (std::size_t n = 0; n < n_out; ++n) {
		out_d[n] = std::complex<double>(static_cast<double>(i_out[n]),
		                                 static_cast<double>(q_out[n]));
	}
	return out_d;
}

// ============================================================================
// Measurement: run reference (uniform double) and a test configuration; compute
// SNR/ENOB on the magnitude streams.
// ============================================================================

// Compare two complex baseband streams' magnitude envelopes after skipping
// `skip` transient samples (the polyphase delay line takes ~num_taps/decim
// samples to fill — using 16 to be safe). Returns NaN if sizes mismatch or
// there aren't enough samples; NaN unambiguously says "no measurement",
// distinguishing it from a legitimate poor SNR (e.g., 0 dB or -3 dB).
double measure_baseband_snr_db(
		const std::vector<std::complex<double>>& reference_out,
		const std::vector<std::complex<double>>& test_out,
		std::size_t skip = 40) {  // multistage chain transient is ~30 outputs
	if (test_out.size() != reference_out.size() || test_out.size() <= skip)
		return std::numeric_limits<double>::quiet_NaN();
	std::vector<double> ref_mag(test_out.size() - skip);
	std::vector<double> tst_mag(test_out.size() - skip);
	for (std::size_t i = skip; i < test_out.size(); ++i) {
		ref_mag[i - skip] = std::abs(reference_out[i]);
		tst_mag[i - skip] = std::abs(test_out[i]);
	}
	return snr_db(std::span<const double>(ref_mag.data(), ref_mag.size()),
	              std::span<const double>(tst_mag.data(), tst_mag.size()));
}

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
	row.cic_overflow_margin_bits = -1.0;     // CIC metric not measured per config

	auto test_out = run_pipeline<CoeffScalar, StateScalar, SampleScalar>(adc_in);
	const double s = measure_baseband_snr_db(reference_out, test_out);
	row.output_snr_db = s;
	row.output_enob   = std::isnan(s)
		? std::numeric_limits<double>::quiet_NaN()
		: enob_from_snr_db(s);
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
	auto reference_out = run_pipeline<double, double, double>(adc_in);

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
		if (std::isnan(r.output_snr_db)) {
			std::cout << std::right << std::setw(11) << "FAIL"
			          << std::right << std::setw(8)  << "—";
		} else if (r.output_snr_db >= 299.0) {
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
	auto reference_out = run_pipeline<double, double, double>(ideal_in);

	print_adc_scan_header();
	for (int bits : params.adc_scan_bits) {
		auto adc_in = simulate_adc(bits);
		auto out_d = run_pipeline<double, double, double>(adc_in);
		const double s = measure_baseband_snr_db(reference_out, out_d);
		if (std::isnan(s)) continue;

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

// Parse `--key=value` style flags. Unknown flags are reported and ignored.
// Recognized: --if-freq=<Hz>, --sample-rate=<Hz>, --adc-bits=<comma list>,
// --num-samples=<N>, --csv=<path>. Anything else is treated as the CSV path
// (legacy positional argument) for backwards compatibility.
void print_usage(const char* prog) {
	std::cout
		<< "Usage: " << prog << " [OPTIONS] [csv_path]\n\n"
		<< "Options:\n"
		<< "  --if-freq=<Hz>          IF frequency (default: 100000)\n"
		<< "  --sample-rate=<Hz>      ADC sample rate (default: 1000000)\n"
		<< "  --adc-bits=8,12,14,16   Comma-separated bit depths for the ADC scan\n"
		<< "  --num-samples=<N>       Input block length (default: 4096)\n"
		<< "  --csv=<path>            Output CSV path (default: acquisition_demo.csv)\n"
		<< "  -h, --help              This message\n";
}

// Parse a comma-separated list of ints. Throws std::invalid_argument
// with a clear message identifying the offending token, instead of
// letting std::stoi's bare exceptions propagate up to terminate().
std::vector<int> parse_int_list(const std::string& s) {
	std::vector<int> out;
	std::size_t pos = 0;
	while (pos < s.size()) {
		const std::size_t comma = s.find(',', pos);
		const std::string tok = s.substr(pos, comma - pos);
		if (!tok.empty()) {
			try {
				out.push_back(std::stoi(tok));
			} catch (const std::exception&) {
				throw std::invalid_argument(
					"could not parse '" + tok + "' as an integer");
			}
		}
		if (comma == std::string::npos) break;
		pos = comma + 1;
	}
	return out;
}

int main(int argc, char** argv) {
	std::string csv_path = "acquisition_demo.csv";
	try {
		for (int i = 1; i < argc; ++i) {
			const std::string arg = argv[i];
			if (arg == "-h" || arg == "--help") {
				print_usage(argv[0]);
				return 0;
			}
			if (arg.rfind("--if-freq=", 0) == 0) {
				params.if_frequency_hz = std::stod(arg.substr(10));
			} else if (arg.rfind("--sample-rate=", 0) == 0) {
				params.sample_rate_hz = std::stod(arg.substr(14));
			} else if (arg.rfind("--adc-bits=", 0) == 0) {
				params.adc_scan_bits = parse_int_list(arg.substr(11));
			} else if (arg.rfind("--num-samples=", 0) == 0) {
				params.num_samples = static_cast<std::size_t>(std::stoull(arg.substr(14)));
			} else if (arg.rfind("--csv=", 0) == 0) {
				csv_path = arg.substr(6);
			} else if (arg.rfind("--", 0) == 0) {
				std::cerr << "Unknown flag: " << arg << "\n";
				print_usage(argv[0]);
				return 1;
			} else {
				// Positional CSV path (legacy)
				csv_path = arg;
			}
		}
	} catch (const std::exception& ex) {
		std::cerr << "Error parsing arguments: " << ex.what() << "\n";
		print_usage(argv[0]);
		return 1;
	}

	// Validate parameters before doing any pipeline math. Catches the
	// obvious failure modes (zero sample rate, IF outside Nyquist,
	// empty bit-scan list) early with a clear error.
	auto fail = [&](const std::string& msg) {
		std::cerr << "Invalid parameter: " << msg << "\n";
		print_usage(argv[0]);
		return 1;
	};
	if (!(params.sample_rate_hz > 0.0))
		return fail("--sample-rate must be positive");
	if (!(params.if_frequency_hz >= 0.0 &&
	      params.if_frequency_hz < params.sample_rate_hz / 2.0))
		return fail("--if-freq must satisfy 0 <= IF < sample_rate/2 (Nyquist)");
	if (params.num_samples == 0)
		return fail("--num-samples must be > 0");
	if (params.adc_scan_bits.empty())
		return fail("--adc-bits must list at least one bit depth");
	for (int b : params.adc_scan_bits) {
		if (b < 1 || b > 64)
			return fail("--adc-bits values must be in [1, 64]");
	}

	const std::size_t total_decim =
		params.ddc_decimation * params.cic_ratio * 2 * params.poly_decimation;
	if (total_decim == 0)
		return fail("internal: total decimation factor is 0 (check PipelineParams)");

	std::cout << "=== Acquisition Pipeline Demo (Issue #93) ===\n";
	std::cout << "Pipeline: ADC -> DDC(NCO + mixer + polyphase ↓"
	          << params.ddc_decimation << ") -> [I/Q parallel] CIC↓"
	          << params.cic_ratio << " -> HB↓2 -> Poly↓"
	          << params.poly_decimation << "  (total ↓" << total_decim << ")\n";
	std::cout << "Sample rate:    " << params.sample_rate_hz / 1e6 << " MHz\n";
	std::cout << "IF frequency:   " << params.if_frequency_hz / 1e3 << " kHz\n";
	std::cout << "Output rate:    "
	          << params.sample_rate_hz / static_cast<double>(total_decim) / 1e3
	          << " kHz\n";
	std::cout << "Block length:   " << params.num_samples << " input samples\n";
	std::cout << "DDC FIR taps:   " << params.fir_taps << "\n";
	std::cout << "Half-band taps: " << params.hb_taps << "\n";
	std::cout << "Chain FIR taps: " << params.chain_fir_taps << "\n\n";

	// Sweep across number-system configurations at 16-bit ADC
	std::cout << "=== Number-system sweep (16-bit ADC) ===\n";
	auto adc_in = simulate_adc(16);
	auto rows = sweep_configurations(adc_in);
	print_sweep_table(rows);

	// ADC bit-depth scan with the uniform-double pipeline
	scan_adc_bit_depths(rows);

	// CSV export
	write_acquisition_csv(csv_path, rows);
	std::cout << "\nCSV: " << csv_path << " (" << rows.size() << " rows)\n";

	return 0;
}
