// fft_tradeoff.cpp: FFT trade-off analysis with Pareto frontier
//
// Evaluates historically-grounded fixed-point FFT configurations and
// uniform-precision sweeps across five number systems. Measures quality
// (SNR, SFDR, spectral leakage, Parseval energy) vs cost (bit-width,
// estimated energy) to produce Pareto frontier data for visualization.
//
// Complements the IIR precision sweep (#69) — one covers filter design,
// this covers spectral analysis.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/spectral/fft.hpp>
#include <sw/dsp/math/constants.hpp>

#if __has_include(<bit>)
#include <bit>
#endif
#include <sw/universal/number/posit/posit.hpp>
#include <sw/universal/number/fixpnt/fixpnt.hpp>
#include <sw/universal/number/cfloat/cfloat.hpp>
#include <sw/universal/number/lns/lns.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace sw::dsp;
using namespace sw::universal;

// ============================================================================
// Type aliases
// ============================================================================

using p8   = posit<8,  2>;
using p16  = posit<16, 2>;
using p24  = posit<24, 2>;
using p32  = posit<32, 2>;

using cf16 = cfloat<16, 5, uint16_t, true, false, false>;
using cf32 = cfloat<32, 8, uint32_t, true, false, false>;

using q11    = fixpnt<12, 11>;
using q15    = fixpnt<16, 15>;
using q23    = fixpnt<24, 23>;
using q31    = fixpnt<32, 31>;
using q48_32 = fixpnt<48, 32>;
using fx16   = fixpnt<16, 8>;
using fx32   = fixpnt<32, 16>;

using lns16 = lns<16, 10>;

// ============================================================================
// Test parameters
// ============================================================================

static const int FFT_SIZES[] = {64, 256, 1024, 4096};
static const int NUM_SIZES = 4;
static constexpr double AMPLITUDE = 0.25;

// ============================================================================
// Result structure
// ============================================================================

struct FftRow {
	std::string config_name;
	std::string number_system;
	int         sample_bits;
	int         total_bits;
	double      energy_proxy;
	int         fft_size;
	std::string signal_type;
	double      snr_db;
	double      sqnr_db;
	double      sfdr_db;
	double      leakage_error;
	double      parseval_error;
};

// ============================================================================
// Signal generation (all in double)
// ============================================================================

struct RefData {
	std::vector<double> signal;
	std::vector<std::complex<double>> spectrum;
	int peak_bin;
};

using complex_d = std::complex<double>;
using cvec_d    = mtl::vec::dense_vector<complex_for_t<double>>;

cvec_d run_ref_fft(const std::vector<double>& sig) {
	std::size_t N = sig.size();
	cvec_d data(N, complex_d{});
	for (std::size_t i = 0; i < N; ++i)
		data[i] = complex_d(sig[i], 0.0);
	spectral::fft_forward<double>(data);
	return data;
}

RefData gen_single_tone(int N) {
	RefData rd;
	rd.peak_bin = N / 8;
	rd.signal.resize(static_cast<std::size_t>(N));
	for (int n = 0; n < N; ++n)
		rd.signal[static_cast<std::size_t>(n)] =
			AMPLITUDE * std::sin(two_pi * rd.peak_bin * n / static_cast<double>(N));
	auto fft_out = run_ref_fft(rd.signal);
	rd.spectrum.resize(static_cast<std::size_t>(N));
	for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i)
		rd.spectrum[i] = fft_out[i];
	return rd;
}

RefData gen_two_tone(int N) {
	RefData rd;
	int k1 = N / 8, k2 = N / 4;
	rd.peak_bin = k1;
	rd.signal.resize(static_cast<std::size_t>(N));
	for (int n = 0; n < N; ++n)
		rd.signal[static_cast<std::size_t>(n)] =
			0.5 * AMPLITUDE * (std::sin(two_pi * k1 * n / static_cast<double>(N)) +
			                   std::sin(two_pi * k2 * n / static_cast<double>(N)));
	auto fft_out = run_ref_fft(rd.signal);
	rd.spectrum.resize(static_cast<std::size_t>(N));
	for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i)
		rd.spectrum[i] = fft_out[i];
	return rd;
}

RefData gen_noise(int N, unsigned seed = 42) {
	RefData rd;
	rd.peak_bin = -1;
	rd.signal.resize(static_cast<std::size_t>(N));
	std::mt19937 rng(seed);
	std::normal_distribution<double> dist(0.0, AMPLITUDE * 0.3);
	for (int n = 0; n < N; ++n)
		rd.signal[static_cast<std::size_t>(n)] = dist(rng);
	auto fft_out = run_ref_fft(rd.signal);
	rd.spectrum.resize(static_cast<std::size_t>(N));
	for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i)
		rd.spectrum[i] = fft_out[i];
	return rd;
}

RefData gen_chirp(int N) {
	RefData rd;
	rd.peak_bin = -1;
	rd.signal.resize(static_cast<std::size_t>(N));
	double f0 = 0.0, f1 = 0.5 * N;
	for (int n = 0; n < N; ++n) {
		double t = static_cast<double>(n) / static_cast<double>(N);
		double phase = two_pi * (f0 * t + 0.5 * (f1 - f0) * t * t);
		rd.signal[static_cast<std::size_t>(n)] =
			AMPLITUDE * std::sin(phase);
	}
	auto fft_out = run_ref_fft(rd.signal);
	rd.spectrum.resize(static_cast<std::size_t>(N));
	for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i)
		rd.spectrum[i] = fft_out[i];
	return rd;
}

// ============================================================================
// Quality metric computation
// ============================================================================

double compute_snr(const std::vector<complex_d>& ref,
                   const std::vector<complex_d>& test) {
	double sig_power = 0.0, noise_power = 0.0;
	for (std::size_t i = 0; i < ref.size(); ++i) {
		sig_power += std::norm(ref[i]);
		noise_power += std::norm(ref[i] - test[i]);
	}
	if (noise_power < 1e-300) return 300.0;
	return 10.0 * std::log10(sig_power / noise_power);
}

double compute_sfdr(const std::vector<complex_d>& X, int peak_bin) {
	if (peak_bin < 0) return 0.0;
	int N = static_cast<int>(X.size());
	double peak = std::norm(X[static_cast<std::size_t>(peak_bin)]);
	double max_spur = 0.0;
	for (int i = 0; i < N; ++i) {
		if (std::abs(i - peak_bin) <= 2) continue;
		if (std::abs(i - (N - peak_bin)) <= 2) continue;
		double val = std::norm(X[static_cast<std::size_t>(i)]);
		max_spur = std::max(max_spur, val);
	}
	if (max_spur < 1e-300) return 300.0;
	if (peak < 1e-300) return 0.0;
	return 10.0 * std::log10(peak / max_spur);
}

double compute_leakage(const std::vector<complex_d>& X, int peak_bin) {
	if (peak_bin < 0) return 0.0;
	int N = static_cast<int>(X.size());
	double total = 0.0, main_lobe = 0.0;
	for (int i = 0; i < N; ++i) {
		double val = std::norm(X[static_cast<std::size_t>(i)]);
		total += val;
		if (std::abs(i - peak_bin) <= 1 || std::abs(i - (N - peak_bin)) <= 1)
			main_lobe += val;
	}
	if (total < 1e-300) return 0.0;
	return (total - main_lobe) / total;
}

double compute_parseval_error(const std::vector<double>& x,
                              const std::vector<complex_d>& X) {
	double time_energy = 0.0;
	for (auto v : x) time_energy += v * v;
	double freq_energy = 0.0;
	for (const auto& v : X) freq_energy += std::norm(v);
	freq_energy /= static_cast<double>(X.size());
	if (time_energy < 1e-300) return 0.0;
	return std::abs(time_energy - freq_energy) / time_energy;
}

// ============================================================================
// Generic FFT measurement
// ============================================================================

// Run FFT in type StateT with input quantized through SampleT.
// When SampleT == StateT, this is a uniform-precision config.
template <typename StateT, typename SampleT = StateT>
FftRow measure_fft(const std::string& config_name,
                   const std::string& number_system,
                   int sample_bits, int state_bits, int twiddle_bits,
                   const RefData& ref, const std::string& signal_type) {
	using complex_t = complex_for_t<StateT>;
	int N = static_cast<int>(ref.signal.size());

	// Quantize signal: double → SampleT → StateT
	mtl::vec::dense_vector<complex_t> data(static_cast<std::size_t>(N), complex_t{});
	std::vector<double> quantized_signal(static_cast<std::size_t>(N));
	for (int n = 0; n < N; ++n) {
		SampleT sample = static_cast<SampleT>(ref.signal[static_cast<std::size_t>(n)]);
		StateT  state  = static_cast<StateT>(static_cast<double>(sample));
		data[static_cast<std::size_t>(n)] = complex_t(state, StateT{});
		quantized_signal[static_cast<std::size_t>(n)] = static_cast<double>(sample);
	}

	// Run FFT in StateT precision (twiddle factors also use StateT;
	// the library does not yet support a separate twiddle type)
	spectral::fft_forward<StateT>(data);

	// Convert result to std::complex<double> for comparison
	std::vector<complex_d> test_spectrum(static_cast<std::size_t>(N));
	for (int i = 0; i < N; ++i) {
		test_spectrum[static_cast<std::size_t>(i)] = complex_d(
			static_cast<double>(data[static_cast<std::size_t>(i)].real()),
			static_cast<double>(data[static_cast<std::size_t>(i)].imag()));
	}

	// Also compute reference FFT of the quantized input for fair SNR
	auto ref_of_quantized = run_ref_fft(quantized_signal);
	std::vector<complex_d> ref_q(static_cast<std::size_t>(N));
	for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i)
		ref_q[i] = ref_of_quantized[i];

	double snr  = compute_snr(ref.spectrum, test_spectrum);
	double sqnr = compute_snr(ref_q, test_spectrum);
	double sfdr = compute_sfdr(test_spectrum, ref.peak_bin);
	double leak = compute_leakage(test_spectrum, ref.peak_bin);
	double pars = compute_parseval_error(quantized_signal, test_spectrum);

	int total_bits = sample_bits + state_bits + twiddle_bits;
	double energy  = std::pow(static_cast<double>(state_bits) / 64.0, 1.5);

	return {config_name, number_system, sample_bits, total_bits, energy,
	        N, signal_type, snr, sqnr, sfdr, leak, pars};
}

// ============================================================================
// Configuration sweeps
// ============================================================================

void run_config(std::vector<FftRow>& rows,
                const std::string& config_name,
                const std::string& number_system,
                int sample_bits, int state_bits, int twiddle_bits,
                auto measure_fn) {
	const char* sig_names[] = {"single_tone", "two_tone", "noise", "chirp"};
	for (int si = 0; si < NUM_SIZES; ++si) {
		int N = FFT_SIZES[si];
		RefData signals[] = {gen_single_tone(N), gen_two_tone(N),
		                     gen_noise(N), gen_chirp(N)};
		for (int t = 0; t < 4; ++t) {
			rows.push_back(measure_fn(config_name, number_system,
			                          sample_bits, state_bits, twiddle_bits,
			                          signals[t], sig_names[t]));
		}
	}
}

template <typename T>
void add_uniform(std::vector<FftRow>& rows, const std::string& name,
                 const std::string& system, int bits) {
	run_config(rows, name, system, bits, bits, bits,
	           measure_fft<T, T>);
}

template <typename StateT, typename SampleT>
void add_mixed(std::vector<FftRow>& rows, const std::string& name,
               const std::string& system,
               int sample_bits, int state_bits, int twiddle_bits) {
	run_config(rows, name, system, sample_bits, state_bits, twiddle_bits,
	           measure_fft<StateT, SampleT>);
}

std::vector<FftRow> sweep_uniform() {
	std::vector<FftRow> rows;
	std::cout << "  Uniform configs..." << std::flush;

	add_uniform<double>(rows, "double",         "ieee",  64);
	add_uniform<float>(rows,  "float",          "ieee",  32);
	add_uniform<p32>(rows,    "posit<32,2>",    "posit", 32);
	add_uniform<p16>(rows,    "posit<16,2>",    "posit", 16);
	add_uniform<p8>(rows,     "posit<8,2>",     "posit",  8);
	add_uniform<cf32>(rows,   "cfloat<32,8>",   "cfloat",32);
	add_uniform<cf16>(rows,   "cfloat<16,5>",   "cfloat",16);
	add_uniform<fx32>(rows,   "fixpnt<32,16>",  "fixpnt",32);
	add_uniform<fx16>(rows,   "fixpnt<16,8>",   "fixpnt",16);
	add_uniform<lns16>(rows,  "lns<16,10>",     "lns",   16);

	std::cout << " done (" << rows.size() << " rows)\n";
	return rows;
}

std::vector<FftRow> sweep_named() {
	std::vector<FftRow> rows;
	std::cout << "  Named configs..." << std::flush;

	add_mixed<q31, q15>(rows,
		"TMS320 classic", "fixpnt", 16, 32, 16);

	add_mixed<q48_32, q23>(rows,
		"DSP56000 style", "fixpnt", 24, 48, 24);

	add_mixed<q31, q11>(rows,
		"Radar 12-bit", "fixpnt", 12, 32, 16);

	add_mixed<p32, p16>(rows,
		"Posit pipeline", "posit", 16, 32, 32);

	add_mixed<p24, p8>(rows,
		"Posit narrow", "posit", 8, 24, 24);

	add_mixed<p32, float>(rows,
		"Cross-system", "mixed", 32, 32, 32);

	add_mixed<p32, lns16>(rows,
		"LNS input", "mixed", 16, 32, 32);

	std::cout << " done (" << rows.size() << " rows)\n";
	return rows;
}

// ============================================================================
// Output
// ============================================================================

std::string csv_quote(const std::string& s) {
	if (s.find_first_of(",\"\n") == std::string::npos) return s;
	std::string quoted = "\"";
	for (char c : s) {
		if (c == '"') quoted += "\"\"";
		else quoted += c;
	}
	quoted += '"';
	return quoted;
}

void print_summary(const std::vector<FftRow>& rows) {
	std::cout << "\n" << std::string(110, '=') << "\n";
	std::cout << "  Results Summary (single-tone, N=256)\n";
	std::cout << std::string(110, '=') << "\n\n";

	std::cout << std::left  << std::setw(22) << "Configuration"
	          << std::right << std::setw(8)  << "Bits"
	          << std::right << std::setw(10) << "SNR(dB)"
	          << std::right << std::setw(10) << "SFDR(dB)"
	          << std::right << std::setw(14) << "Leakage"
	          << std::right << std::setw(14) << "Parseval Err"
	          << std::right << std::setw(10) << "Energy"
	          << "\n";
	std::cout << std::string(88, '-') << "\n";

	for (const auto& r : rows) {
		if (r.fft_size != 256 || r.signal_type != "single_tone") continue;

		std::cout << std::left << std::setw(22) << r.config_name;
		std::cout << std::right << std::setw(8) << r.total_bits;

		if (r.snr_db > 290.0)
			std::cout << std::right << std::setw(10) << "inf";
		else
			std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(1) << r.snr_db;

		if (r.sfdr_db > 290.0)
			std::cout << std::right << std::setw(10) << "inf";
		else
			std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(1) << r.sfdr_db;

		std::cout << std::right << std::setw(14) << std::scientific << std::setprecision(2) << r.leakage_error;
		std::cout << std::right << std::setw(14) << std::scientific << std::setprecision(2) << r.parseval_error;
		std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(4) << r.energy_proxy;
		std::cout << "\n";
	}
}

void write_csv(const std::string& path, const std::vector<FftRow>& rows) {
	std::ofstream ofs(path);
	if (!ofs) throw std::runtime_error("cannot open output file: " + path);
	ofs << "config_name,number_system,sample_bits,total_bits,energy_proxy,"
	    << "fft_size,signal_type,snr_db,sqnr_db,sfdr_db,leakage_error,parseval_error\n";
	ofs << std::setprecision(15);
	for (const auto& r : rows) {
		ofs << csv_quote(r.config_name) << ","
		    << csv_quote(r.number_system) << ","
		    << r.sample_bits << ","
		    << r.total_bits << ","
		    << r.energy_proxy << ","
		    << r.fft_size << ","
		    << csv_quote(r.signal_type) << ","
		    << r.snr_db << ","
		    << r.sqnr_db << ","
		    << r.sfdr_db << ","
		    << r.leakage_error << ","
		    << r.parseval_error << "\n";
	}
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
	try {
		std::string outdir = ".";
		if (argc > 1) outdir = argv[1];
		if (!outdir.empty() && outdir.back() == '/') outdir.pop_back();

		std::cout << std::string(110, '=') << "\n";
		std::cout << "  FFT Trade-Off Analysis — Pareto Frontier\n";
		std::cout << "  5 number systems, 17 configurations\n";
		std::cout << "  Signals: single tone, two-tone, noise, chirp\n";
		std::cout << "  FFT sizes: 64, 256, 1024, 4096\n";
		std::cout << "  Metrics: SNR, SFDR, spectral leakage, Parseval energy\n";
		std::cout << std::string(110, '=') << "\n\n";

		std::vector<FftRow> all_rows;

		auto uniform = sweep_uniform();
		all_rows.insert(all_rows.end(), uniform.begin(), uniform.end());

		auto named = sweep_named();
		all_rows.insert(all_rows.end(), named.begin(), named.end());

		print_summary(all_rows);

		std::string csv_path = outdir + "/fft_tradeoff.csv";
		write_csv(csv_path, all_rows);

		std::cout << "\n" << std::string(110, '=') << "\n";
		std::cout << "  Total: " << all_rows.size() << " measurements\n";
		std::cout << "  CSV:   " << csv_path << "\n";
		std::cout << std::string(110, '=') << "\n";

		return 0;
	} catch (const std::exception& e) {
		std::cerr << "ERROR: " << e.what() << '\n';
		return 1;
	}
}
