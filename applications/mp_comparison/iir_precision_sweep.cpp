// iir_precision_sweep.cpp: mixed-precision IIR filter comparison
//
// Sweeps 6 IIR filter families across 8 arithmetic types, measuring:
//   - Impulse response error (max absolute and relative)
//   - SQNR when filtering a test signal
//   - Pole displacement from double reference
//   - Stability margin
//
// Outputs console tables and CSV files for Python visualization.
//
// Test conditions:
//   4th order, 44100 Hz sample rate, 2000 Hz cutoff
//   Coefficients designed in double, state+samples in target type
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/filter/iir/butterworth.hpp>
#include <sw/dsp/filter/iir/chebyshev1.hpp>
#include <sw/dsp/filter/iir/chebyshev2.hpp>
#include <sw/dsp/filter/iir/elliptic.hpp>
#include <sw/dsp/filter/iir/bessel.hpp>
#include <sw/dsp/filter/iir/legendre.hpp>
#include <sw/dsp/filter/filter.hpp>
#include <sw/dsp/signals/generators.hpp>
#include <sw/dsp/analysis/stability.hpp>
#include <sw/dsp/analysis/sensitivity.hpp>
#include <sw/dsp/types/projection.hpp>

#if __has_include(<bit>)
#include <bit>
#endif
#include <sw/universal/number/fixpnt/fixpnt.hpp>
#include <sw/universal/number/cfloat/cfloat.hpp>
#include <sw/universal/number/posit/posit.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace sw::dsp;
using namespace sw::universal;

// ============================================================================
// Test parameters
// ============================================================================

constexpr int    ORDER           = 4;
constexpr double SAMPLE_RATE     = 44100.0;
constexpr double CUTOFF          = 2000.0;
constexpr double RIPPLE_DB       = 1.0;
constexpr double STOPBAND_DB     = 40.0;
// Elliptic uses a DSPFilters-style selectivity parameter, not stopband dB.
// Valid range is [0.1, 5.0]; see elliptic.hpp.
constexpr double ELLIPTIC_ROLLOFF = 1.0;
constexpr int    IMPULSE_LEN     = 200;
constexpr int    SIGNAL_LEN      = 2000;

// ============================================================================
// Result structure
// ============================================================================

struct MetricRow {
	std::string filter_family;
	std::string arith_type;
	int         bits;
	double      max_abs_error;
	double      max_rel_error;
	double      sqnr_db;
	double      pole_displacement;
	double      stability_margin_val;
};

struct FreqResponseRow {
	std::string filter_family;
	std::string arith_type;
	double      freq_hz;
	double      magnitude_db;
	double      phase_deg;
	double      ref_magnitude_db;
	double      ref_phase_deg;
};

struct PoleRow {
	std::string filter_family;
	std::string arith_type;
	int         pole_index;
	double      real_part;
	double      imag_part;
	double      ref_real;
	double      ref_imag;
	double      displacement;
};

// Quote a CSV field if it contains commas, quotes, or newlines.
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

// Global collectors for CSV export
std::vector<FreqResponseRow> g_freq_rows;
std::vector<PoleRow> g_pole_rows;

// ============================================================================
// Measurement functions
// ============================================================================

template <typename FilterType>
std::vector<double> impulse_response(FilterType& filter, int length) {
	using sample_t = typename FilterType::sample_scalar;
	filter.reset();
	std::vector<double> result(static_cast<std::size_t>(length));
	for (int n = 0; n < length; ++n) {
		sample_t x = (n == 0) ? static_cast<sample_t>(1) : static_cast<sample_t>(0);
		result[static_cast<std::size_t>(n)] = static_cast<double>(filter.process(x));
	}
	return result;
}

template <typename FilterType>
std::vector<double> filter_test_signal(FilterType& filter) {
	using sample_t = typename FilterType::sample_scalar;
	filter.reset();
	std::vector<double> result(SIGNAL_LEN);
	for (int n = 0; n < SIGNAL_LEN; ++n) {
		double t = static_cast<double>(n) / SAMPLE_RATE;
		double x_d = std::sin(2.0 * 3.14159265358979 * 500.0 * t)
		           + std::sin(2.0 * 3.14159265358979 * 5000.0 * t);
		sample_t x = static_cast<sample_t>(x_d);
		result[static_cast<std::size_t>(n)] = static_cast<double>(filter.process(x));
	}
	return result;
}

double compute_sqnr(const std::vector<double>& ref, const std::vector<double>& test) {
	double signal_power = 0, noise_power = 0;
	for (std::size_t i = 0; i < ref.size(); ++i) {
		signal_power += ref[i] * ref[i];
		double err = ref[i] - test[i];
		noise_power += err * err;
	}
	if (noise_power < 1e-300) return 300.0;
	return 10.0 * std::log10(signal_power / noise_power);
}

template <typename RefFilter, typename MixedFilter>
MetricRow compare_filter(const std::string& family, const std::string& type_name,
                         int bits, RefFilter& ref_filter, MixedFilter& mixed_filter) {
	auto ir_ref = impulse_response(ref_filter, IMPULSE_LEN);
	auto ir_mix = impulse_response(mixed_filter, IMPULSE_LEN);

	double max_abs = 0, max_ref = 0;
	for (std::size_t i = 0; i < ir_ref.size(); ++i) {
		max_abs = std::max(max_abs, std::abs(ir_ref[i] - ir_mix[i]));
		max_ref = std::max(max_ref, std::abs(ir_ref[i]));
	}
	double max_rel = (max_ref > 0) ? max_abs / max_ref : 0.0;

	auto sig_ref = filter_test_signal(ref_filter);
	auto sig_mix = filter_test_signal(mixed_filter);
	double sqnr = compute_sqnr(sig_ref, sig_mix);

	double disp = pole_displacement(ref_filter.cascade(), mixed_filter.cascade());
	double margin = sw::dsp::stability_margin(mixed_filter.cascade());

	// Collect frequency response data for CSV (mixed and reference)
	constexpr int NUM_FREQS = 200;
	for (int k = 0; k < NUM_FREQS; ++k) {
		double f = static_cast<double>(k) / NUM_FREQS * 0.5;
		double freq_hz = f * SAMPLE_RATE;

		auto resp_mix = mixed_filter.cascade().response(f);
		double mag_mix = std::abs(std::complex<double>(
			static_cast<double>(resp_mix.real()), static_cast<double>(resp_mix.imag())));
		double mag_mix_db = (mag_mix > 1e-20) ? 20.0 * std::log10(mag_mix) : -400.0;
		double phase_mix = std::atan2(static_cast<double>(resp_mix.imag()),
		                              static_cast<double>(resp_mix.real()));
		double phase_mix_deg = phase_mix * 180.0 / 3.14159265358979;

		auto resp_ref = ref_filter.cascade().response(f);
		double mag_ref = std::abs(std::complex<double>(
			static_cast<double>(resp_ref.real()), static_cast<double>(resp_ref.imag())));
		double mag_ref_db = (mag_ref > 1e-20) ? 20.0 * std::log10(mag_ref) : -400.0;
		double phase_ref = std::atan2(static_cast<double>(resp_ref.imag()),
		                              static_cast<double>(resp_ref.real()));
		double phase_ref_deg = phase_ref * 180.0 / 3.14159265358979;

		g_freq_rows.push_back({ family, type_name, freq_hz,
		                        mag_mix_db, phase_mix_deg, mag_ref_db, phase_ref_deg });
	}

	// Collect pole positions with reference comparison
	auto poles_mix = all_poles(mixed_filter.cascade());
	auto poles_ref = all_poles(ref_filter.cascade());
	for (std::size_t i = 0; i < poles_mix.size(); ++i) {
		double ref_r = (i < poles_ref.size()) ? poles_ref[i].real() : 0.0;
		double ref_i = (i < poles_ref.size()) ? poles_ref[i].imag() : 0.0;
		double d = (i < poles_ref.size()) ? std::abs(poles_mix[i] - poles_ref[i]) : 0.0;
		g_pole_rows.push_back({ family, type_name, static_cast<int>(i),
		                        poles_mix[i].real(), poles_mix[i].imag(),
		                        ref_r, ref_i, d });
	}

	return { family, type_name, bits, max_abs, max_rel, sqnr, disp, margin };
}

// ============================================================================
// Per-type comparison macro-free template dispatch
// ============================================================================

// Run one filter type and collect the metric row.
// setup_fn is a generic lambda that calls the appropriate setup() on the filter.
template <typename FilterDesign, typename RefFilter, typename SetupFn>
void sweep_type(const std::string& family, const std::string& type_name,
                int bits, RefFilter& ref, std::vector<MetricRow>& rows,
                SetupFn setup_fn) {
	SimpleFilter<FilterDesign> f;
	setup_fn(f);
	rows.push_back(compare_filter(family, type_name, bits, ref, f));
}

// ============================================================================
// Per-family sweep functions
// ============================================================================

using cf24  = cfloat<24, 5, uint32_t, true, false, false>;
using half_ = cfloat<16, 5, uint16_t, true, false, false>;
// NOTE: avoid the names fp32/fp16 here. `using namespace sw::universal`
// above pulls in sw::universal::fp32 (== single) and sw::universal::fp16
// (== half) from cfloat.hpp. A local `using fp32 = ...` would collide
// and make the name ambiguous inside template-argument contexts, which
// GCC reports as the cryptic "template argument is invalid" (issue #51).
using fxp32 = fixpnt<32, 16>;
using fxp16 = fixpnt<16, 8>;
using p32   = posit<32, 2>;
using p16   = posit<16, 1>;

// Helper: generate sweep for a lowpass filter (Butterworth, Bessel, Legendre)
template <template <int, typename, typename, typename> class LP>
std::vector<MetricRow> sweep_lp(const std::string& fam) {
	std::vector<MetricRow> rows;
	SimpleFilter<LP<ORDER, double, double, double>> ref;
	ref.setup(ORDER, SAMPLE_RATE, CUTOFF);
	auto s = [](auto& f) { f.setup(ORDER, SAMPLE_RATE, CUTOFF); };

	sweep_type<LP<ORDER, double, double, double>>(fam, "double",         64, ref, rows, s);
	sweep_type<LP<ORDER, double, float,  float>> (fam, "float",          32, ref, rows, s);
	sweep_type<LP<ORDER, double, cf24,   cf24>>  (fam, "cfloat<24,5>",   24, ref, rows, s);
	sweep_type<LP<ORDER, double, half_,  half_>> (fam, "half",           16, ref, rows, s);
	sweep_type<LP<ORDER, double, p32,    p32>>   (fam, "posit<32,2>",    32, ref, rows, s);
	sweep_type<LP<ORDER, double, p16,    p16>>   (fam, "posit<16,1>",    16, ref, rows, s);
	sweep_type<LP<ORDER, double, fxp32,  fxp32>> (fam, "fixpnt<32,16>",  32, ref, rows, s);
	sweep_type<LP<ORDER, double, fxp16,  fxp16>> (fam, "fixpnt<16,8>",   16, ref, rows, s);
	return rows;
}

std::vector<MetricRow> sweep_butterworth() {
	return sweep_lp<iir::ButterworthLowPass>("Butterworth");
}

// Helper: generate sweep for Chebyshev I (ripple parameter)
template <template <int, typename, typename, typename> class LP>
std::vector<MetricRow> sweep_cheby1(const std::string& fam) {
	std::vector<MetricRow> rows;
	SimpleFilter<LP<ORDER, double, double, double>> ref;
	ref.setup(ORDER, SAMPLE_RATE, CUTOFF, RIPPLE_DB);
	auto s = [](auto& f) { f.setup(ORDER, SAMPLE_RATE, CUTOFF, RIPPLE_DB); };

	sweep_type<LP<ORDER, double, double, double>>(fam, "double",         64, ref, rows, s);
	sweep_type<LP<ORDER, double, float,  float>> (fam, "float",          32, ref, rows, s);
	sweep_type<LP<ORDER, double, cf24,   cf24>>  (fam, "cfloat<24,5>",   24, ref, rows, s);
	sweep_type<LP<ORDER, double, half_,  half_>> (fam, "half",           16, ref, rows, s);
	sweep_type<LP<ORDER, double, p32,    p32>>   (fam, "posit<32,2>",    32, ref, rows, s);
	sweep_type<LP<ORDER, double, p16,    p16>>   (fam, "posit<16,1>",    16, ref, rows, s);
	sweep_type<LP<ORDER, double, fxp32,  fxp32>> (fam, "fixpnt<32,16>",  32, ref, rows, s);
	sweep_type<LP<ORDER, double, fxp16,  fxp16>> (fam, "fixpnt<16,8>",   16, ref, rows, s);
	return rows;
}

// Helper: generate sweep for Chebyshev II (stopband parameter)
template <template <int, typename, typename, typename> class LP>
std::vector<MetricRow> sweep_cheby2(const std::string& fam) {
	std::vector<MetricRow> rows;
	SimpleFilter<LP<ORDER, double, double, double>> ref;
	ref.setup(ORDER, SAMPLE_RATE, CUTOFF, STOPBAND_DB);
	auto s = [](auto& f) { f.setup(ORDER, SAMPLE_RATE, CUTOFF, STOPBAND_DB); };

	sweep_type<LP<ORDER, double, double, double>>(fam, "double",         64, ref, rows, s);
	sweep_type<LP<ORDER, double, float,  float>> (fam, "float",          32, ref, rows, s);
	sweep_type<LP<ORDER, double, cf24,   cf24>>  (fam, "cfloat<24,5>",   24, ref, rows, s);
	sweep_type<LP<ORDER, double, half_,  half_>> (fam, "half",           16, ref, rows, s);
	sweep_type<LP<ORDER, double, p32,    p32>>   (fam, "posit<32,2>",    32, ref, rows, s);
	sweep_type<LP<ORDER, double, p16,    p16>>   (fam, "posit<16,1>",    16, ref, rows, s);
	sweep_type<LP<ORDER, double, fxp32,  fxp32>> (fam, "fixpnt<32,16>",  32, ref, rows, s);
	sweep_type<LP<ORDER, double, fxp16,  fxp16>> (fam, "fixpnt<16,8>",   16, ref, rows, s);
	return rows;
}

// Helper: generate sweep for Elliptic (ripple + rolloff selectivity)
template <template <int, typename, typename, typename> class LP>
std::vector<MetricRow> sweep_elliptic_family(const std::string& fam) {
	std::vector<MetricRow> rows;
	SimpleFilter<LP<ORDER, double, double, double>> ref;
	ref.setup(ORDER, SAMPLE_RATE, CUTOFF, RIPPLE_DB, ELLIPTIC_ROLLOFF);
	auto s = [](auto& f) { f.setup(ORDER, SAMPLE_RATE, CUTOFF, RIPPLE_DB, ELLIPTIC_ROLLOFF); };

	sweep_type<LP<ORDER, double, double, double>>(fam, "double",         64, ref, rows, s);
	sweep_type<LP<ORDER, double, float,  float>> (fam, "float",          32, ref, rows, s);
	sweep_type<LP<ORDER, double, cf24,   cf24>>  (fam, "cfloat<24,5>",   24, ref, rows, s);
	sweep_type<LP<ORDER, double, half_,  half_>> (fam, "half",           16, ref, rows, s);
	sweep_type<LP<ORDER, double, p32,    p32>>   (fam, "posit<32,2>",    32, ref, rows, s);
	sweep_type<LP<ORDER, double, p16,    p16>>   (fam, "posit<16,1>",    16, ref, rows, s);
	sweep_type<LP<ORDER, double, fxp32,  fxp32>> (fam, "fixpnt<32,16>",  32, ref, rows, s);
	sweep_type<LP<ORDER, double, fxp16,  fxp16>> (fam, "fixpnt<16,8>",   16, ref, rows, s);
	return rows;
}

std::vector<MetricRow> sweep_chebyshev1() { return sweep_cheby1<iir::ChebyshevILowPass>("Chebyshev I"); }
std::vector<MetricRow> sweep_chebyshev2() { return sweep_cheby2<iir::ChebyshevIILowPass>("Chebyshev II"); }
std::vector<MetricRow> sweep_elliptic()   { return sweep_elliptic_family<iir::EllipticLowPass>("Elliptic"); }
std::vector<MetricRow> sweep_bessel()     { return sweep_lp<iir::BesselLowPass>("Bessel"); }
std::vector<MetricRow> sweep_legendre()   { return sweep_lp<iir::LegendreLowPass>("Legendre"); }

// ============================================================================
// Output formatting
// ============================================================================

void print_table(const std::string& family, const std::vector<MetricRow>& rows) {
	std::cout << "\n" << std::string(100, '=') << "\n";
	std::cout << "  " << family << " (order=" << ORDER
	          << ", fs=" << SAMPLE_RATE << ", fc=" << CUTOFF << ")\n";
	std::cout << std::string(100, '=') << "\n\n";

	std::cout << std::left  << std::setw(16) << "Type"
	          << std::right << std::setw(6)  << "Bits"
	          << std::right << std::setw(13) << "Abs Error"
	          << std::right << std::setw(13) << "Rel Error"
	          << std::right << std::setw(11) << "SQNR(dB)"
	          << std::right << std::setw(13) << "Pole Disp"
	          << std::right << std::setw(12) << "Margin"
	          << "\n";
	std::cout << std::string(84, '-') << "\n";

	for (const auto& r : rows) {
		std::cout << std::left << std::setw(16) << r.arith_type
		          << std::right << std::setw(6) << r.bits;

		auto fmt_sci = [](double v, int w) {
			if (v < 1e-15) { std::cout << std::right << std::setw(w) << "0"; return; }
			std::cout << std::right << std::setw(w) << std::scientific << std::setprecision(2) << v;
		};

		fmt_sci(r.max_abs_error, 13);
		fmt_sci(r.max_rel_error, 13);

		if (r.sqnr_db > 290.0)
			std::cout << std::right << std::setw(11) << "inf";
		else
			std::cout << std::right << std::setw(11) << std::fixed << std::setprecision(1) << r.sqnr_db;

		fmt_sci(r.pole_displacement, 13);
		std::cout << std::right << std::setw(12) << std::fixed << std::setprecision(6) << r.stability_margin_val;
		std::cout << "\n";
	}
}

void write_freq_csv(const std::string& path) {
	std::ofstream ofs(path);
	if (!ofs) { std::cerr << "WARNING: cannot open " << path << "\n"; return; }
	ofs << "filter_family,arith_type,freq_hz,magnitude_db,phase_deg,"
	    << "ref_magnitude_db,ref_phase_deg\n";
	ofs << std::setprecision(10);
	for (const auto& r : g_freq_rows) {
		ofs << csv_quote(r.filter_family) << "," << csv_quote(r.arith_type) << ","
		    << r.freq_hz << "," << r.magnitude_db << "," << r.phase_deg << ","
		    << r.ref_magnitude_db << "," << r.ref_phase_deg << "\n";
	}
}

void write_pole_csv(const std::string& path) {
	std::ofstream ofs(path);
	if (!ofs) { std::cerr << "WARNING: cannot open " << path << "\n"; return; }
	ofs << "filter_family,arith_type,pole_index,real,imag,"
	    << "ref_real,ref_imag,displacement\n";
	ofs << std::setprecision(15);
	for (const auto& r : g_pole_rows) {
		ofs << csv_quote(r.filter_family) << "," << csv_quote(r.arith_type) << ","
		    << r.pole_index << "," << r.real_part << "," << r.imag_part << ","
		    << r.ref_real << "," << r.ref_imag << "," << r.displacement << "\n";
	}
}

void write_csv(const std::string& path, const std::vector<MetricRow>& all_rows) {
	std::ofstream ofs(path);
	if (!ofs) {
		std::cerr << "WARNING: cannot open " << path << " for CSV output\n";
		return;
	}

	ofs << "filter_family,arith_type,bits,max_abs_error,max_rel_error,"
	    << "sqnr_db,pole_displacement,stability_margin\n";
	ofs << std::setprecision(15);

	for (const auto& r : all_rows) {
		ofs << csv_quote(r.filter_family) << ","
		    << csv_quote(r.arith_type) << ","
		    << r.bits << ","
		    << r.max_abs_error << ","
		    << r.max_rel_error << ","
		    << r.sqnr_db << ","
		    << r.pole_displacement << ","
		    << r.stability_margin_val << "\n";
	}
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
  try {
	// Optional output directory argument
	std::string outdir = ".";
	if (argc > 1) outdir = argv[1];
	std::cout << std::string(100, '=') << "\n";
	std::cout << "  Mixed-Precision IIR Filter Comparison\n";
	std::cout << "  6 filter families x 8 arithmetic types\n";
	std::cout << "  Order=" << ORDER << ", fs=" << SAMPLE_RATE
	          << " Hz, fc=" << CUTOFF << " Hz\n";
	std::cout << std::string(100, '=') << "\n";

	std::vector<MetricRow> all_rows;

	auto run = [&](const std::string& name, auto sweep_fn) {
		auto rows = sweep_fn();
		print_table(name, rows);
		all_rows.insert(all_rows.end(), rows.begin(), rows.end());
	};

	run("Butterworth",            sweep_butterworth);
	run("Chebyshev I (1dB)",      sweep_chebyshev1);
	run("Chebyshev II (40dB)",    sweep_chebyshev2);
	run("Elliptic (1dB, rolloff=1.0)", sweep_elliptic);
	run("Bessel",                 sweep_bessel);
	run("Legendre",               sweep_legendre);

	// Write CSV files
	std::string sep = "/";
	write_csv(outdir + sep + "iir_precision_sweep.csv", all_rows);
	write_freq_csv(outdir + sep + "frequency_response.csv");
	write_pole_csv(outdir + sep + "pole_positions.csv");

	std::cout << "\n" << std::string(100, '=') << "\n";
	std::cout << "  Summary: " << all_rows.size() << " measurements ("
	          << "6 families x 8 types)\n";
	std::cout << "  CSV files in: " << outdir << "/\n";
	std::cout << "    iir_precision_sweep.csv  (" << all_rows.size() << " rows)\n";
	std::cout << "    frequency_response.csv   (" << g_freq_rows.size() << " rows)\n";
	std::cout << "    pole_positions.csv       (" << g_pole_rows.size() << " rows)\n";
	std::cout << std::string(100, '=') << "\n";

	return 0;
  } catch (const std::exception& e) {
	std::cerr << "ERROR: " << e.what() << '\n';
	return 1;
  }
}
