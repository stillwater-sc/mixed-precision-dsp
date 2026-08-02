// fractional_delay.cpp: polyphase fractional-delay demo (issue #138).
//
// Exercises the L-phase polyphase FractionalDelay at 1/L-sample
// resolution across the standard six-config mixed-precision matrix.
//
// Test structure:
//
//   Test A - delay accuracy:
//     For each requested delay in {6.0, 6.25, 6.5, 6.75, 6.9, 7.0} samples,
//     run a 1 kHz sine through the fractional-delay line and measure the
//     actual implemented delay by fitting the phase of the FFT bin at
//     1 kHz. Compare measured vs. requested; report absolute error.
//     Acceptance: reference config within 1% of requested for all cases.
//
//   Test B - group-delay flatness:
//     For each test frequency in {100, 500, 1000, 2000, 5000, 10000} Hz,
//     run a sine at requested delay = 8.5 samples through the delay line
//     and measure the implemented group delay from the FFT bin phase.
//     Report the spread; verify < 0.5 sample variation across the
//     passband.
//     Acceptance: reference config group-delay spread < 0.5 samples.
//
//   Test C - precision sweep:
//     Repeat Test A across all six numeric configs (reference / float /
//     posit32 / posit16 / cfloat32 / fixpnt32). Report per-config delay
//     accuracy so cross-precision differences are visible.
//
// CSV schema (long format):
//   pipeline, test, config, tone_hz, requested_delay, measured_delay,
//   error_samples, gain_db
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/multirate/fractional_delay.hpp>
#include <sw/dsp/spectral/fft.hpp>

#include <mtl/vec/dense_vector.hpp>

#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/fixpnt/fixpnt.hpp>
#include <universal/number/posit/posit.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

using namespace sw::dsp;

// ============================================================================
// Type aliases + constants
// ============================================================================

using p32  = sw::universal::posit<32, 2>;
using p16  = sw::universal::posit<16, 2>;
using cf32 = sw::universal::cfloat<32, 8, uint32_t, true, false, false>;
// fixpnt<32,24>: Q8.24 - +/-128 range. Coefficients here are DC-normalized
// (max value approx 1) so 24 fractional bits are the useful precision axis.
using fx32 = sw::universal::fixpnt<32, 24>;

struct DemoParams {
	double      sample_rate_hz = 48000.0;
	std::size_t input_len      = 8192;

	// Polyphase bank sizing
	std::size_t L              = 64;    // 1/64 sample resolution
	std::size_t taps_per_phase = 15;    // odd, gives 7.0 sample group delay
	std::size_t max_int_delay  = 32;
	double      kaiser_beta    = 8.0;   // ~-58 dB stopband

	// Test A: fixed-frequency, swept-delay
	double              test_a_tone_hz    = 1000.0;
	std::vector<double> test_a_delays     = {7.0, 7.25, 7.5, 7.75, 7.9, 8.0};

	// Test B: swept-frequency, fixed-delay
	double              test_b_delay      = 8.5;
	std::vector<double> test_b_tones_hz   = {100.0, 500.0, 1000.0,
	                                          2000.0, 5000.0, 10000.0};

	// Analysis
	std::size_t transient_skip = 400;   // > L*taps_per_phase/2 + max delay
};
inline DemoParams params;

// ============================================================================
// Delay measurement via FFT phase
//
// For a pure tone x(t) = A * sin(2 pi f t) delayed by tau seconds:
//   y(t) = A * sin(2 pi f (t - tau)) = A * sin(2 pi f t - 2 pi f tau)
// The FFT of the delayed sequence has phase -2 pi f tau at the tone bin.
// Compared to the input phase, the difference gives -2 pi f tau, from
// which tau (in seconds, or in samples if we multiply by fs) can be
// recovered.
//
// This is much more accurate than cross-correlation peak-picking for
// sub-sample delays: the phase readout scales linearly with the delay,
// so measurement noise is bounded by the FFT SNR rather than by grid
// discretization.
// ============================================================================

// Complex spectrum value at a specific tone frequency. Applies no
// window - we assume the analysis captures a whole number of cycles
// (choose input_len to make the tone periodic in the window when
// possible) so leakage is small.
std::complex<double> fft_bin_at(const std::vector<double>& x,
                                 double tone_hz,
                                 double sample_rate_hz,
                                 std::size_t fft_size) {
	mtl::vec::dense_vector<std::complex<double>> buf(fft_size,
	                                                    std::complex<double>{});
	const std::size_t take = std::min(x.size(), fft_size);
	for (std::size_t i = 0; i < take; ++i) {
		buf[i] = std::complex<double>(x[i], 0.0);
	}
	sw::dsp::spectral::fft_forward<double>(buf);
	// Find nearest bin
	const std::size_t k = static_cast<std::size_t>(
		std::round(tone_hz * static_cast<double>(fft_size) / sample_rate_hz));
	return buf[k];
}

// Measure the delay in samples between `output` and `input` by comparing
// the phase of the FFT bin at `tone_hz` in each sequence.
//
// `hint_delay_samples` disambiguates phase wraparound at higher
// frequencies (where the raw phase-diff measurement only recovers the
// delay modulo one tone period). The returned delay is the multiple
// of tone_period closest to `hint_delay_samples`.
double measure_delay_samples(const std::vector<double>& input,
                              const std::vector<double>& output,
                              double tone_hz,
                              double sample_rate_hz,
                              double hint_delay_samples) {
	// Round FFT size to a power of two >= input length, for FFT
	// efficiency. (The spectral::fft_forward implementation handles
	// arbitrary sizes; power-of-two is just faster.)
	std::size_t fft_size = 1;
	while (fft_size < input.size()) fft_size <<= 1;

	const auto in_bin  = fft_bin_at(input,  tone_hz, sample_rate_hz, fft_size);
	const auto out_bin = fft_bin_at(output, tone_hz, sample_rate_hz, fft_size);
	if (std::abs(in_bin) < 1e-15 || std::abs(out_bin) < 1e-15)
		return std::numeric_limits<double>::quiet_NaN();

	// Phase difference in radians. Output lags input, so out_bin's phase
	// should be *less* than in_bin's phase - phase difference is negative
	// for a positive delay.
	double phase_diff = std::arg(out_bin) - std::arg(in_bin);
	// Wrap to (-pi, pi]
	const double two_pi = 2.0 * std::numbers::pi_v<double>;
	while (phase_diff >  std::numbers::pi_v<double>) phase_diff -= two_pi;
	while (phase_diff <= -std::numbers::pi_v<double>) phase_diff += two_pi;
	// Delay in seconds = -phase_diff / (2*pi*f); in samples = * fs.
	const double delay_sec = -phase_diff / (two_pi * tone_hz);
	double measured = delay_sec * sample_rate_hz;
	// Unwrap: pick the period multiple that lands closest to the hint.
	const double period_samples = sample_rate_hz / tone_hz;
	const double n_periods = std::round((hint_delay_samples - measured)
	                                     / period_samples);
	return measured + n_periods * period_samples;
}

// Magnitude ratio |out|/|in| at the tone bin, in dB. Should be very
// close to 0 for a well-designed fractional delay across the passband.
double measure_gain_db(const std::vector<double>& input,
                        const std::vector<double>& output,
                        double tone_hz,
                        double sample_rate_hz) {
	std::size_t fft_size = 1;
	while (fft_size < input.size()) fft_size <<= 1;
	const auto in_bin  = fft_bin_at(input,  tone_hz, sample_rate_hz, fft_size);
	const auto out_bin = fft_bin_at(output, tone_hz, sample_rate_hz, fft_size);
	if (std::abs(in_bin) < 1e-15) return std::numeric_limits<double>::quiet_NaN();
	return 20.0 * std::log10(std::abs(out_bin) / std::abs(in_bin));
}

// ============================================================================
// Test signal generation
// ============================================================================

std::vector<double> generate_tone(std::size_t n_samples,
                                    double sample_rate_hz,
                                    double tone_hz,
                                    double amplitude = 1.0) {
	std::vector<double> out(n_samples);
	const double two_pi = 2.0 * std::numbers::pi_v<double>;
	for (std::size_t n = 0; n < n_samples; ++n) {
		out[n] = amplitude * std::sin(two_pi * tone_hz
		                               * static_cast<double>(n)
		                               / sample_rate_hz);
	}
	return out;
}

// ============================================================================
// Run the FractionalDelay over `input` at a fixed requested delay.
// Returns the resulting output as doubles for measurement.
// ============================================================================

template <class T>
std::vector<double> run_delay_line(const std::vector<double>& input,
                                    double requested_delay) {
	multirate::FractionalDelay<double, T, T> fd(
		params.L, params.taps_per_phase,
		params.max_int_delay, params.kaiser_beta);
	std::vector<double> output;
	output.reserve(input.size());
	for (double x : input) {
		T y = fd.delay(static_cast<T>(x), requested_delay);
		output.push_back(static_cast<double>(y));
	}
	return output;
}

// ============================================================================
// Record + CSV
// ============================================================================

struct Row {
	std::string test;
	std::string config;
	std::string scalar_type;
	double      tone_hz;
	double      requested_delay;
	double      measured_delay;
	double      error_samples;
	double      gain_db;
};

void write_csv(const std::string& path, const std::vector<Row>& rows) {
	std::ofstream out(path);
	if (!out) throw std::runtime_error("write_csv: cannot open " + path);
	out << "pipeline,test,config,scalar_type,tone_hz,requested_delay,"
	    << "measured_delay,error_samples,gain_db\n";
	out << std::setprecision(9);
	for (const auto& r : rows) {
		out << "fractional_delay," << r.test << "," << r.config << ","
		    << "\"" << r.scalar_type << "\"," << r.tone_hz << ","
		    << r.requested_delay << "," << r.measured_delay << ","
		    << r.error_samples << "," << r.gain_db << "\n";
	}
}

// ============================================================================
// Test drivers
// ============================================================================

// Trim the transient region (approximately the intrinsic group delay plus
// the largest requested integer offset) before measurement.
std::vector<double> trim_transient(const std::vector<double>& x) {
	if (x.size() <= params.transient_skip) return {};
	return std::vector<double>(x.begin() + params.transient_skip, x.end());
}

// Test A: sweep requested delay at a fixed tone, one config.
template <class T>
void run_test_a(std::vector<Row>& rows,
                 const std::string& config, const std::string& type_str) {
	const auto input = generate_tone(params.input_len,
	                                   params.sample_rate_hz,
	                                   params.test_a_tone_hz);
	const auto in_trim = trim_transient(input);
	for (double req : params.test_a_delays) {
		auto out = run_delay_line<T>(input, req);
		const auto out_trim = trim_transient(out);
		const double meas = measure_delay_samples(in_trim, out_trim,
		                                            params.test_a_tone_hz,
		                                            params.sample_rate_hz,
		                                            req);
		const double gain = measure_gain_db(in_trim, out_trim,
		                                     params.test_a_tone_hz,
		                                     params.sample_rate_hz);
		rows.push_back({"delay_sweep", config, type_str,
		                params.test_a_tone_hz, req, meas,
		                meas - req, gain});
	}
}

// Test B: sweep tone frequency at a fixed requested delay, one config.
template <class T>
void run_test_b(std::vector<Row>& rows,
                 const std::string& config, const std::string& type_str) {
	for (double f : params.test_b_tones_hz) {
		const auto input = generate_tone(params.input_len,
		                                   params.sample_rate_hz, f);
		auto out = run_delay_line<T>(input, params.test_b_delay);
		const auto in_trim  = trim_transient(input);
		const auto out_trim = trim_transient(out);
		const double meas = measure_delay_samples(in_trim, out_trim,
		                                            f, params.sample_rate_hz,
		                                            params.test_b_delay);
		const double gain = measure_gain_db(in_trim, out_trim,
		                                     f, params.sample_rate_hz);
		rows.push_back({"freq_sweep", config, type_str, f,
		                params.test_b_delay, meas,
		                meas - params.test_b_delay, gain});
	}
}

// ============================================================================
// Console summary
// ============================================================================

void print_test_a_summary(const std::vector<Row>& rows) {
	std::cout << "\nTest A - delay accuracy at " << params.test_a_tone_hz
	          << " Hz (all configs):\n";
	std::cout << std::string(90, '-') << "\n";
	std::cout << std::left << std::setw(14) << "config"
	          << std::right << std::setw(14) << "requested"
	          << std::setw(14) << "measured"
	          << std::setw(14) << "error"
	          << std::setw(14) << "gain(dB)" << "\n";
	std::cout << std::string(90, '-') << "\n";
	for (const auto& r : rows) {
		if (r.test != "delay_sweep") continue;
		std::cout << std::left << std::setw(14) << r.config
		          << std::right << std::fixed << std::setprecision(4)
		          << std::setw(14) << r.requested_delay
		          << std::setw(14) << r.measured_delay
		          << std::setw(14) << r.error_samples
		          << std::setprecision(3)
		          << std::setw(14) << r.gain_db << "\n";
	}
}

void print_test_b_summary(const std::vector<Row>& rows) {
	std::cout << "\nTest B - group-delay flatness at requested "
	          << params.test_b_delay << " samples (all configs):\n";
	std::cout << std::string(90, '-') << "\n";
	std::cout << std::left << std::setw(14) << "config"
	          << std::right << std::setw(12) << "tone(Hz)"
	          << std::setw(14) << "measured"
	          << std::setw(14) << "error"
	          << std::setw(14) << "gain(dB)" << "\n";
	std::cout << std::string(90, '-') << "\n";
	for (const auto& r : rows) {
		if (r.test != "freq_sweep") continue;
		std::cout << std::left << std::setw(14) << r.config
		          << std::right << std::fixed << std::setprecision(2)
		          << std::setw(12) << r.tone_hz
		          << std::setprecision(4)
		          << std::setw(14) << r.measured_delay
		          << std::setw(14) << r.error_samples
		          << std::setprecision(3)
		          << std::setw(14) << r.gain_db << "\n";
	}
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv) try {
	std::string csv_path = "fractional_delay.csv";
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--csv=", 0) == 0)     csv_path = a.substr(6);
		else if (a == "-h" || a == "--help") {
			std::cout << "Usage: " << argv[0] << " [--csv=path]\n";
			return 0;
		}
	}

	std::cout << "fractional_delay: polyphase fractional-delay demo\n"
	          << "  sample rate:     " << params.sample_rate_hz << " Hz\n"
	          << "  L (phases):      " << params.L
	          << "  (resolution = 1/" << params.L << " samples)\n"
	          << "  taps per phase:  " << params.taps_per_phase << "\n"
	          << "  total taps:      " << (params.L * params.taps_per_phase)
	          << "\n"
	          << "  base group delay:" << (0.5 * (params.taps_per_phase - 1))
	          << " samples\n"
	          << "  input length:    " << params.input_len << " samples\n\n";

	std::vector<Row> rows;

	auto run_config = [&](auto tag, const std::string& name,
	                       const std::string& type_str) {
		using T = decltype(tag);
		std::cout << "  running " << name << " (" << type_str << ")...\n";
		run_test_a<T>(rows, name, type_str);
		run_test_b<T>(rows, name, type_str);
	};

	run_config(double{},   "reference", "double");
	run_config(float{},    "float",     "float");
	run_config(p32{},      "posit32",   "posit<32,2>");
	run_config(p16{},      "posit16",   "posit<16,2>");
	run_config(cf32{},     "cfloat32",  "cfloat<32,8>");
	run_config(fx32{},     "fixpnt32",  "fixpnt<32,24>");

	print_test_a_summary(rows);
	print_test_b_summary(rows);

	write_csv(csv_path, rows);
	std::cout << "\nCSV written: " << csv_path << "\n";

	// Acceptance criteria (issue #138):
	//   1. Delay accuracy within 1% at all test frequencies for double
	//      reference.
	//   2. Group delay flat to within 0.5 sample across the passband.
	bool acc_ok = true;
	double worst_a_err = 0.0;
	double min_group_delay = std::numeric_limits<double>::infinity();
	double max_group_delay = -std::numeric_limits<double>::infinity();
	for (const auto& r : rows) {
		if (r.config != "reference") continue;
		if (r.test == "delay_sweep") {
			const double pct = std::abs(r.error_samples) / r.requested_delay;
			if (pct > 0.01) acc_ok = false;
			worst_a_err = std::max(worst_a_err, pct);
		}
		if (r.test == "freq_sweep") {
			min_group_delay = std::min(min_group_delay, r.measured_delay);
			max_group_delay = std::max(max_group_delay, r.measured_delay);
		}
	}
	const double gd_spread = max_group_delay - min_group_delay;

	std::cout << "\nAcceptance:\n"
	          << "  A. reference worst delay error: "
	          << std::fixed << std::setprecision(4) << (worst_a_err * 100.0)
	          << "% (limit: 1%)  " << (acc_ok ? "[ok]" : "[FAIL]") << "\n"
	          << "  B. reference group-delay spread: "
	          << gd_spread << " samples (limit: 0.5)  "
	          << ((gd_spread < 0.5) ? "[ok]" : "[FAIL]") << "\n";
	if (!acc_ok || gd_spread >= 0.5) return 1;
	return 0;
} catch (const std::exception& ex) {
	std::cerr << "FATAL: " << ex.what() << "\n";
	return 1;
}
