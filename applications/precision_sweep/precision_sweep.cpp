// precision_sweep.cpp: flagship mixed-precision pipeline comparison
//
// Sweeps 6 pipeline configurations across 5 number systems (IEEE float,
// posit, fixed-point, LNS, custom float), measuring 7 quality metrics
// per configuration. Demonstrates the mixed-precision value proposition
// with three-scalar parameterization: CoeffScalar / StateScalar / SampleScalar.
//
// Pipeline configurations:
//   1. Uniform       — same type at all three positions
//   2. Classic mixed — double coefficients, varied state and samples
//   3. Posit         — p32 coeff → p24 state → p{16,12,8} samples
//   4. Fixed-point   — Q-format coefficient/state/sample combos
//   5. Cross-system  — double → posit state → fixpnt samples
//   6. LNS           — double → lns state → various samples
//
// Test filter: 4th-order Butterworth lowpass, 44100 Hz, 2000 Hz cutoff
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/filter/iir/butterworth.hpp>
#include <sw/dsp/filter/filter.hpp>
#include <sw/dsp/signals/generators.hpp>
#include <sw/dsp/analysis/stability.hpp>
#include <sw/dsp/analysis/sensitivity.hpp>
#include <sw/dsp/analysis/condition.hpp>
#include <sw/dsp/types/projection.hpp>
#include <sw/dsp/math/constants.hpp>

#if __has_include(<bit>)
#include <bit>
#endif
#include <sw/universal/number/fixpnt/fixpnt.hpp>
#include <sw/universal/number/cfloat/cfloat.hpp>
#include <sw/universal/number/posit/posit.hpp>
#include <sw/universal/number/lns/lns.hpp>

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

constexpr int    ORDER       = 4;
constexpr double SAMPLE_RATE = 44100.0;
constexpr double CUTOFF      = 2000.0;
constexpr int    SIGNAL_LEN  = 2000;

// Passband: [0, CUTOFF/SAMPLE_RATE] = [0, 0.0454]
// Stopband: [CUTOFF*1.5/SAMPLE_RATE, 0.5] ≈ [0.068, 0.5]
constexpr double PASSBAND_EDGE = CUTOFF / SAMPLE_RATE;
constexpr double STOPBAND_EDGE = CUTOFF * 1.5 / SAMPLE_RATE;

// ============================================================================
// Type aliases — 5 number systems
// ============================================================================

// Custom floats
using cf24  = cfloat<24, 5, uint32_t, true, false, false>;
using half_ = cfloat<16, 5, uint16_t, true, false, false>;

// Posits (es=2: standard DSP, es=1: balance, es=0: high fraction)
using p32   = posit<32, 2>;
using p24   = posit<24, 2>;
using p16   = posit<16, 2>;
using p12   = posit<12, 2>;
using p8    = posit<8,  2>;
using p16e1 = posit<16, 1>;
using p8e1  = posit<8,  1>;

// Fixed-point (Q-format)
using q31    = fixpnt<32, 31>;   // Q31   — accumulator/state
using q9_31  = fixpnt<40, 31>;   // Q9.31 — wide accumulator
using q15    = fixpnt<16, 15>;   // Q15   — sample precision
using q11    = fixpnt<12, 11>;   // Q11   — sample precision
using q7     = fixpnt<8,  7>;    // Q7    — sample precision
using fxp32  = fixpnt<32, 16>;
using fxp16  = fixpnt<16, 8>;

// LNS (logarithmic number system)
using lns32  = lns<32, 22>;
using lns16  = lns<16, 10>;

// ============================================================================
// Result structure
// ============================================================================

struct MetricRow {
	std::string pipeline;
	std::string config_name;
	std::string coeff_type;
	std::string state_type;
	std::string sample_type;
	double      sqnr_db;
	double      max_coeff_error;
	double      pole_displacement;
	double      stability_margin_val;
	double      passband_ripple_db;
	double      stopband_atten_db;
	double      condition_number;
};

// ============================================================================
// Measurement functions
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

template <typename FilterType>
std::vector<double> filter_test_signal(FilterType& filter) {
	using sample_t = typename FilterType::sample_scalar;
	filter.reset();
	std::vector<double> result(SIGNAL_LEN);
	for (int n = 0; n < SIGNAL_LEN; ++n) {
		double t = static_cast<double>(n) / SAMPLE_RATE;
		// Peak amplitude ±1.0: half-scale to stay within Q-format [-1,1) range
		double x_d = 0.5 * (std::sin(two_pi * 500.0 * t)
		                   + std::sin(two_pi * 5000.0 * t));
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

template <typename T, int MaxStages>
double max_coefficient_error(const Cascade<double, MaxStages>& ref,
                             const Cascade<T, MaxStages>& test) {
	double max_err = 0.0;
	for (int i = 0; i < ref.num_stages(); ++i) {
		auto& r = ref.stage(i);
		auto& t = test.stage(i);
		max_err = std::max(max_err, std::abs(static_cast<double>(r.b0) - static_cast<double>(t.b0)));
		max_err = std::max(max_err, std::abs(static_cast<double>(r.b1) - static_cast<double>(t.b1)));
		max_err = std::max(max_err, std::abs(static_cast<double>(r.b2) - static_cast<double>(t.b2)));
		max_err = std::max(max_err, std::abs(static_cast<double>(r.a1) - static_cast<double>(t.a1)));
		max_err = std::max(max_err, std::abs(static_cast<double>(r.a2) - static_cast<double>(t.a2)));
	}
	return max_err;
}

template <typename T, int MaxStages>
double measure_passband_ripple(const Cascade<T, MaxStages>& cascade) {
	double max_db = -1e20, min_db = 1e20;
	constexpr int NUM_PTS = 100;
	for (int k = 1; k <= NUM_PTS; ++k) {
		double f = PASSBAND_EDGE * static_cast<double>(k) / NUM_PTS;
		auto resp = cascade.response(f);
		double mag = std::abs(std::complex<double>(
			static_cast<double>(resp.real()), static_cast<double>(resp.imag())));
		double db = (mag > 1e-20) ? 20.0 * std::log10(mag) : -400.0;
		max_db = std::max(max_db, db);
		min_db = std::min(min_db, db);
	}
	return max_db - min_db;
}

template <typename T, int MaxStages>
double measure_stopband_attenuation(const Cascade<T, MaxStages>& cascade) {
	double worst_db = -400.0;
	constexpr int NUM_PTS = 200;
	for (int k = 0; k < NUM_PTS; ++k) {
		double f = STOPBAND_EDGE + (0.5 - STOPBAND_EDGE) * static_cast<double>(k) / NUM_PTS;
		auto resp = cascade.response(f);
		double mag = std::abs(std::complex<double>(
			static_cast<double>(resp.real()), static_cast<double>(resp.imag())));
		double db = (mag > 1e-20) ? 20.0 * std::log10(mag) : -400.0;
		worst_db = std::max(worst_db, db);
	}
	return worst_db;
}

// ============================================================================
// Generic sweep entry point
// ============================================================================

using RefFilter = SimpleFilter<iir::ButterworthLowPass<ORDER, double, double, double>>;

template <typename MixedFilter>
MetricRow measure_config(const std::string& pipeline, const std::string& config_name,
                         const std::string& coeff_name, const std::string& state_name,
                         const std::string& sample_name,
                         RefFilter& ref) {
	MixedFilter mixed;
	mixed.setup(ORDER, SAMPLE_RATE, CUTOFF);
	ref.setup(ORDER, SAMPLE_RATE, CUTOFF);

	auto sig_ref = filter_test_signal(ref);
	auto sig_mix = filter_test_signal(mixed);
	double sqnr = compute_sqnr(sig_ref, sig_mix);

	double coeff_err = max_coefficient_error(ref.cascade(), mixed.cascade());
	double disp = pole_displacement(ref.cascade(), mixed.cascade());
	double margin = sw::dsp::stability_margin(mixed.cascade());
	double pb_ripple = measure_passband_ripple(mixed.cascade());
	double sb_atten = measure_stopband_attenuation(mixed.cascade());
	double cond = cascade_condition_number(mixed.cascade(), 128);

	return { pipeline, config_name, coeff_name, state_name, sample_name,
	         sqnr, coeff_err, disp, margin, pb_ripple, sb_atten, cond };
}

// ============================================================================
// Pipeline sweep functions
// ============================================================================

// Coefficient design always uses double for numerical accuracy (constexpr
// constants like pi_v<T> require std::frexp which is non-constexpr for
// Universal types). "Uniform" here means state and sample types are identical
// (state == sample), while coefficients remain double.
std::vector<MetricRow> sweep_uniform(RefFilter& ref) {
	std::vector<MetricRow> rows;
	auto m = [&](auto tag, const std::string& name,
	             const std::string& cn, const std::string& sn) {
		using F = decltype(tag);
		rows.push_back(measure_config<SimpleFilter<F>>(
			"Uniform", name, cn, sn, sn, ref));
	};

	m(iir::ButterworthLowPass<ORDER, double, double, double>{}, "double",        "double",        "double");
	m(iir::ButterworthLowPass<ORDER, double, float,  float>{},  "float",         "double",        "float");
	m(iir::ButterworthLowPass<ORDER, double, cf24,   cf24>{},   "cfloat<24,5>",  "double",        "cfloat<24,5>");
	m(iir::ButterworthLowPass<ORDER, double, p32,    p32>{},    "posit<32,2>",   "double",        "posit<32,2>");
	m(iir::ButterworthLowPass<ORDER, double, p16,    p16>{},    "posit<16,2>",   "double",        "posit<16,2>");
	m(iir::ButterworthLowPass<ORDER, double, fxp32,  fxp32>{},  "fixpnt<32,16>", "double",        "fixpnt<32,16>");
	m(iir::ButterworthLowPass<ORDER, double, fxp16,  fxp16>{},  "fixpnt<16,8>",  "double",        "fixpnt<16,8>");

	return rows;
}

std::vector<MetricRow> sweep_classic_mixed(RefFilter& ref) {
	std::vector<MetricRow> rows;
	auto m = [&](auto tag, const std::string& st, const std::string& sa) {
		using F = decltype(tag);
		rows.push_back(measure_config<SimpleFilter<F>>(
			"Classic mixed", "dbl/" + st + "/" + sa, "double", st, sa, ref));
	};

	m(iir::ButterworthLowPass<ORDER, double, float,  float>{},  "float",          "float");
	m(iir::ButterworthLowPass<ORDER, double, cf24,   cf24>{},   "cfloat<24,5>",   "cfloat<24,5>");
	m(iir::ButterworthLowPass<ORDER, double, half_,  half_>{},  "half",           "half");
	m(iir::ButterworthLowPass<ORDER, double, p32,    p32>{},    "posit<32,2>",    "posit<32,2>");
	m(iir::ButterworthLowPass<ORDER, double, p16,    p16>{},    "posit<16,2>",    "posit<16,2>");
	m(iir::ButterworthLowPass<ORDER, double, p16e1,  p16e1>{},  "posit<16,1>",    "posit<16,1>");
	m(iir::ButterworthLowPass<ORDER, double, fxp32,  fxp32>{},  "fixpnt<32,16>",  "fixpnt<32,16>");
	m(iir::ButterworthLowPass<ORDER, double, fxp16,  fxp16>{},  "fixpnt<16,8>",   "fixpnt<16,8>");

	return rows;
}

std::vector<MetricRow> sweep_posit_pipeline(RefFilter& ref) {
	std::vector<MetricRow> rows;
	auto m = [&](auto tag, const std::string& name,
	             const std::string& s, const std::string& sa) {
		using F = decltype(tag);
		rows.push_back(measure_config<SimpleFilter<F>>(
			"Posit pipeline", name, "double", s, sa, ref));
	};

	m(iir::ButterworthLowPass<ORDER, double, p32, p16>{},  "dbl/p32/p16", "posit<32,2>", "posit<16,2>");
	m(iir::ButterworthLowPass<ORDER, double, p24, p16>{},  "dbl/p24/p16", "posit<24,2>", "posit<16,2>");
	m(iir::ButterworthLowPass<ORDER, double, p24, p12>{},  "dbl/p24/p12", "posit<24,2>", "posit<12,2>");
	m(iir::ButterworthLowPass<ORDER, double, p24, p8>{},   "dbl/p24/p8",  "posit<24,2>", "posit<8,2>");
	m(iir::ButterworthLowPass<ORDER, double, p16, p8>{},   "dbl/p16/p8",  "posit<16,2>", "posit<8,2>");
	m(iir::ButterworthLowPass<ORDER, double, p16e1,p8e1>{}, "dbl/p16e1/p8e1", "posit<16,1>", "posit<8,1>");

	return rows;
}

std::vector<MetricRow> sweep_fixedpoint_pipeline(RefFilter& ref) {
	std::vector<MetricRow> rows;
	auto m = [&](auto tag, const std::string& name,
	             const std::string& s, const std::string& sa) {
		using F = decltype(tag);
		rows.push_back(measure_config<SimpleFilter<F>>(
			"Fixed-point", name, "double", s, sa, ref));
	};

	m(iir::ButterworthLowPass<ORDER, double, q31,   q15>{}, "dbl/Q31/Q15",   "fixpnt<32,31>", "fixpnt<16,15>");
	m(iir::ButterworthLowPass<ORDER, double, q31,   q7>{},  "dbl/Q31/Q7",    "fixpnt<32,31>", "fixpnt<8,7>");
	m(iir::ButterworthLowPass<ORDER, double, q9_31, q15>{}, "dbl/Q9.31/Q15", "fixpnt<40,31>", "fixpnt<16,15>");
	m(iir::ButterworthLowPass<ORDER, double, q9_31, q11>{}, "dbl/Q9.31/Q11", "fixpnt<40,31>", "fixpnt<12,11>");
	m(iir::ButterworthLowPass<ORDER, double, fxp32, q15>{}, "dbl/fx32/Q15",  "fixpnt<32,16>", "fixpnt<16,15>");

	return rows;
}

// Cross-system: mix number families via native-type samples.
// Direct Universal cross-family casts (posit↔fixpnt) are not yet supported,
// so we use float as the sample type to bridge between families.
std::vector<MetricRow> sweep_cross_system(RefFilter& ref) {
	std::vector<MetricRow> rows;
	auto m = [&](auto tag, const std::string& name,
	             const std::string& c, const std::string& s, const std::string& sa) {
		using F = decltype(tag);
		rows.push_back(measure_config<SimpleFilter<F>>(
			"Cross-system", name, c, s, sa, ref));
	};

	m(iir::ButterworthLowPass<ORDER, double, p32,   float>{}, "dbl/p32/flt",   "double", "posit<32,2>",  "float");
	m(iir::ButterworthLowPass<ORDER, double, p16,   float>{}, "dbl/p16/flt",   "double", "posit<16,2>",  "float");
	m(iir::ButterworthLowPass<ORDER, double, fxp32, float>{}, "dbl/fx32/flt",  "double", "fixpnt<32,16>","float");
	m(iir::ButterworthLowPass<ORDER, double, q31,   float>{}, "dbl/Q31/flt",   "double", "fixpnt<32,31>","float");

	return rows;
}

std::vector<MetricRow> sweep_lns_experiment(RefFilter& ref) {
	std::vector<MetricRow> rows;
	auto m = [&](auto tag, const std::string& name,
	             const std::string& c, const std::string& s, const std::string& sa) {
		using F = decltype(tag);
		rows.push_back(measure_config<SimpleFilter<F>>(
			"LNS experiment", name, c, s, sa, ref));
	};

	m(iir::ButterworthLowPass<ORDER, double, lns32, float>{},  "dbl/lns32/flt",  "double", "lns<32,22>", "float");
	m(iir::ButterworthLowPass<ORDER, double, lns16, float>{},  "dbl/lns16/flt",  "double", "lns<16,10>", "float");

	return rows;
}

// ============================================================================
// Output formatting
// ============================================================================

void print_pipeline(const std::string& name, const std::vector<MetricRow>& rows) {
	std::cout << "\n" << std::string(120, '=') << "\n";
	std::cout << "  Pipeline: " << name << "\n";
	std::cout << std::string(120, '=') << "\n\n";

	std::cout << std::left  << std::setw(28) << "Configuration"
	          << std::right << std::setw(11) << "SQNR(dB)"
	          << std::right << std::setw(13) << "Coeff Err"
	          << std::right << std::setw(13) << "Pole Disp"
	          << std::right << std::setw(10) << "Margin"
	          << std::right << std::setw(11) << "PB Rip(dB)"
	          << std::right << std::setw(11) << "SB Att(dB)"
	          << std::right << std::setw(12) << "Cond #"
	          << "\n";
	std::cout << std::string(109, '-') << "\n";

	auto fmt_sci = [](double v, int w) {
		if (v < 1e-15) { std::cout << std::right << std::setw(w) << "0"; return; }
		std::cout << std::right << std::setw(w) << std::scientific << std::setprecision(2) << v;
	};

	for (const auto& r : rows) {
		std::cout << std::left << std::setw(28) << r.config_name;

		if (r.sqnr_db > 290.0)
			std::cout << std::right << std::setw(11) << "inf";
		else
			std::cout << std::right << std::setw(11) << std::fixed << std::setprecision(1) << r.sqnr_db;

		fmt_sci(r.max_coeff_error, 13);
		fmt_sci(r.pole_displacement, 13);
		std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(6) << r.stability_margin_val;
		std::cout << std::right << std::setw(11) << std::fixed << std::setprecision(4) << r.passband_ripple_db;
		std::cout << std::right << std::setw(11) << std::fixed << std::setprecision(1) << r.stopband_atten_db;
		fmt_sci(r.condition_number, 12);
		std::cout << "\n";
	}
}

void write_sweep_csv(const std::string& path, const std::vector<MetricRow>& rows) {
	std::ofstream ofs(path);
	if (!ofs) { std::cerr << "WARNING: cannot open " << path << "\n"; return; }
	ofs << "pipeline,config,coeff_type,state_type,sample_type,"
	    << "sqnr_db,max_coeff_error,pole_displacement,stability_margin,"
	    << "passband_ripple_db,stopband_attenuation_db,condition_number\n";
	ofs << std::setprecision(15);
	for (const auto& r : rows) {
		ofs << csv_quote(r.pipeline) << ","
		    << csv_quote(r.config_name) << ","
		    << csv_quote(r.coeff_type) << ","
		    << csv_quote(r.state_type) << ","
		    << csv_quote(r.sample_type) << ","
		    << r.sqnr_db << ","
		    << r.max_coeff_error << ","
		    << r.pole_displacement << ","
		    << r.stability_margin_val << ","
		    << r.passband_ripple_db << ","
		    << r.stopband_atten_db << ","
		    << r.condition_number << "\n";
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

		std::cout << std::string(120, '=') << "\n";
		std::cout << "  Mixed-Precision Pipeline Sweep\n";
		std::cout << "  5 number systems x 6 pipeline configurations\n";
		std::cout << "  Filter: Butterworth lowpass, order=" << ORDER
		          << ", fs=" << SAMPLE_RATE << " Hz, fc=" << CUTOFF << " Hz\n";
		std::cout << "  Metrics: SQNR, coeff error, pole displacement, stability margin,\n";
		std::cout << "           passband ripple, stopband attenuation, condition number\n";
		std::cout << std::string(120, '=') << "\n";

		RefFilter ref;
		std::vector<MetricRow> all_rows;

		auto run = [&](const std::string& name, auto sweep_fn) {
			std::cout << "\n  Running " << name << "..." << std::flush;
			auto rows = sweep_fn(ref);
			print_pipeline(name, rows);
			all_rows.insert(all_rows.end(), rows.begin(), rows.end());
		};

		run("Uniform",        sweep_uniform);
		run("Classic mixed",  sweep_classic_mixed);
		run("Posit pipeline", sweep_posit_pipeline);
		run("Fixed-point",    sweep_fixedpoint_pipeline);
		run("Cross-system",   sweep_cross_system);
		run("LNS experiment", sweep_lns_experiment);

		std::string csv_path = outdir + "/precision_sweep.csv";
		write_sweep_csv(csv_path, all_rows);

		std::cout << "\n" << std::string(120, '=') << "\n";
		std::cout << "  Summary: " << all_rows.size() << " configurations measured\n";
		std::cout << "  CSV: " << csv_path << "\n";
		std::cout << std::string(120, '=') << "\n";

		return 0;
	} catch (const std::exception& e) {
		std::cerr << "ERROR: " << e.what() << '\n';
		return 1;
	}
}
