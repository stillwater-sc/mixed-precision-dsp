// audio_resampler.cpp: 44.1 kHz <-> 48 kHz mixed-precision multirate demo.
//
// Canonical rational sample-rate conversion. The natural ratio 48000/44100
// = 320/294 reduces by gcd(48000, 44100) = 300 to L/M = 160/147, showing
// up in every audio production toolchain that has to bridge the two rates.
//
// The demo exercises `RationalResampler` at that ratio across six numeric
// types from the standard mp-comparison matrix (double / float / posit32 /
// posit16 / cfloat32 / fixpnt32). Filter design is fixed at double —
// see the `run_rational_resampler` comment below for rationale — so the
// variable under test is the streaming multiply-accumulate arithmetic.
//
// Per config, per test tone we measure:
//
//   Tone level (dB)          - output amplitude at each expected tone
//   Passband ripple (dB)     - max(tone_level) - min(tone_level)
//   Stopband floor (dB)      - max magnitude above 24 kHz (out-of-band
//                              alias / imaging leakage)
//   In-band SNR (dB)         - signal-to-non-tone-bins in [0, 20 kHz)
//
// The test signal is a 4-tone multitone at [100 Hz, 1 kHz, 10 kHz, 19 kHz]
// with each component at amp 0.25 (sum peak amplitude = 1.0). Tones cover
// both passband edges (100 Hz near DC, 19 kHz near the 20 kHz audio
// ceiling).
//
// Acceptance criteria from issue #136:
//   * Builds on gcc + clang.
//   * Runs end-to-end, produces CSV.
//   * SNR > 80 dB for the double-precision reference config (checked at exit).
//   * All six precision configs measured.
//   * Companion docs page in docs-site/.../multirate/audio-resampler.md.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/conditioning/src.hpp>             // RationalResampler
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/spectral/fft.hpp>
#include <sw/dsp/windows/kaiser.hpp>

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
#include <span>
#include <string>
#include <vector>

using namespace sw::dsp;

// ============================================================================
// Type aliases + constants
// ============================================================================

using p32  = sw::universal::posit<32, 2>;
using p16  = sw::universal::posit<16, 2>;
// cfloat<32,8>: IEEE-754-like 32-bit float layout with the (subnormals=true,
// supernormals=false, saturating=false) semantics used across the demos.
using cf32 = sw::universal::cfloat<32, 8, uint32_t, true, false, false>;
// fixpnt<32,20>: Q12.20 - 12 integer bits (range +/-2048) accommodate the
// L=160 coefficient-scale factor that RationalResampler applies for unity
// passband gain (Q8.24's +/-128 range overflows on the center tap).
// 20 fractional bits give ~120 dB dynamic range.
using fx32 = sw::universal::fixpnt<32, 20>;

struct DemoParams {
	double      input_rate_hz  = 44100.0;
	double      output_rate_hz = 48000.0;
	// L/M = 160/147 (48000/44100 reduced by gcd = 300).
	std::size_t L              = 160;
	std::size_t M              = 147;
	// Input length trades measurement quality (longer = tighter FFT bin
	// spacing) against per-config runtime. The manual polyphase path
	// scales as O(input_len * L * taps_per_branch); for L=160 the low-
	// precision configs (posit16, fixpnt<32,24>) dominate wall-clock, so
	// 12000 samples (~0.27 s of audio) keeps the full 12-run sweep under
	// ~2 minutes while still giving < 6 Hz FFT bin resolution.
	std::size_t input_len      = 12000;

	// Test multitone
	std::vector<double> tones = {100.0, 1000.0, 10000.0, 19000.0};
	double tone_amp           = 0.25;    // per-tone amplitude
	                                     // (sum peak = 4 * 0.25 = 1.0)

	// Resampler design - default RationalResampler(10, 5) gives only
	// ~-42 dB filter stopband (Kaiser beta=5), which limits the achieveable
	// SNR to around 70 dB. Bumping to (20, 12) gives ~-115 dB filter
	// stopband (Kaiser beta=12) so numerical noise from the resampler's
	// own arithmetic dominates, not filter design headroom.
	std::size_t filter_half_length = 20;
	double      kaiser_beta        = 12.0;

	// FFT analysis - 8192 gives 48000/8192 ~ 5.86 Hz bin resolution, enough
	// to isolate each test tone. Must be <= steady-state slice length
	// after trimming filter transients (~2000 samples at both ends of the
	// ~13000-sample output).
	std::size_t fft_size = 8192;
	// Kaiser beta=18 gives ~-180 dB sidelobes so the SNR measurement is
	// bounded by the resampler's arithmetic noise floor, not the analysis
	// window leakage floor. Main-lobe half-width ~ beta/(pi/2) bins = 12
	// bins ~ 70 Hz for our fft/rate, so the guard band below must exceed
	// that to safely isolate main-lobe energy from stopband bins.
	double      window_beta = 18.0;

	// Frequency bands (Hz)
	double passband_upper_hz = 20000.0;       // audio ceiling
	double stopband_lower_hz = 24000.0;       // above Nyquist / 2

	// Tone-peak search window (Hz) - bins within this range around each
	// tone are checked for the tone's peak amplitude.
	double tone_peak_hz = 20.0;
	// Guard band (Hz) around each tone excluded from noise measurement -
	// captures full Kaiser beta=18 main lobe plus first few side lobes so
	// no window leakage leaks into the noise sum.
	double guard_hz = 150.0;
};
inline DemoParams params;

// ============================================================================
// Test signal — deterministic multitone at the configured frequencies.
// ============================================================================

std::vector<double> generate_multitone(std::size_t n_samples,
                                        double sample_rate_hz,
                                        const std::vector<double>& tones_hz,
                                        double amp_per_tone) {
	std::vector<double> out(n_samples);
	const double two_pi = 2.0 * std::numbers::pi_v<double>;
	for (std::size_t n = 0; n < n_samples; ++n) {
		double v = 0.0;
		const double t = static_cast<double>(n) / sample_rate_hz;
		for (double f : tones_hz) {
			v += amp_per_tone * std::sin(two_pi * f * t);
		}
		out[n] = v;
	}
	return out;
}

// ============================================================================
// Method A: library RationalResampler
//
// Follows the established mixed-precision demo pattern (see
// acquisition_demo): CoeffScalar is fixed at double so the internal
// Kaiser-window / sinc-lowpass design routines don't have to run at the
// target streaming precision. StateScalar and SampleScalar carry T so the
// streaming multiply-accumulate loop is the isolated variable under test.
//
// Rationale: kaiser_window<T> and design_fir_lowpass<T> for fixpnt<32,24>
// hit divide-by-zero in the modified Bessel iteration (fixpnt lacks the
// dynamic range Kaiser needs). Designing in double and projecting into T
// per-multiply keeps the sweep isolated to what we actually care about:
// how each numeric type behaves in the streaming SRC arithmetic.
// ============================================================================

template <class T>
std::vector<double> run_rational_resampler(const std::vector<double>& in_d) {
	RationalResampler<double, T, T> rr(params.L, params.M,
	                                    params.filter_half_length,
	                                    params.kaiser_beta);
	mtl::vec::dense_vector<T> in_T(in_d.size());
	for (std::size_t i = 0; i < in_d.size(); ++i)
		in_T[i] = static_cast<T>(in_d[i]);
	auto out_T = rr.process(in_T);
	std::vector<double> out_d(out_T.size());
	for (std::size_t i = 0; i < out_T.size(); ++i)
		out_d[i] = static_cast<double>(out_T[i]);
	return out_d;
}

// ============================================================================
// FFT-based measurements
// ============================================================================

// One-sided magnitude spectrum in dBFS. Zero-pads to `fft_size` and
// applies a Kaiser window to reduce spectral leakage.
std::vector<double> magnitude_spectrum_dbfs(const std::vector<double>& x,
                                              std::size_t fft_size,
                                              double kaiser_beta_win) {
	if (fft_size < 2) return {};
	auto win = kaiser_window<double>(x.size(), kaiser_beta_win);
	// Coherent gain of the window (used to normalize peak level so a
	// full-scale tone reads 0 dBFS).
	double win_sum = 0.0;
	for (std::size_t i = 0; i < win.size(); ++i) win_sum += win[i];
	if (win_sum <= 0.0) win_sum = 1.0;

	mtl::vec::dense_vector<std::complex<double>> buf(fft_size,
	                                                    std::complex<double>{});
	const std::size_t take = std::min(x.size(), fft_size);
	for (std::size_t i = 0; i < take; ++i) {
		buf[i] = std::complex<double>(x[i] * win[i], 0.0);
	}
	sw::dsp::spectral::fft_forward<double>(buf);

	const std::size_t half = fft_size / 2 + 1;
	std::vector<double> mag_db(half);
	for (std::size_t k = 0; k < half; ++k) {
		const double m = std::abs(buf[k]) / win_sum;
		// Peak-normalized dBFS: a full-scale sinusoid reads ~ -6 dBFS
		// after coherent-gain normalization; here we're comparing tones
		// to each other, so the absolute dBFS reference doesn't matter,
		// only the tone-to-noise-floor ratio.
		//
		// -300 dBFS floor keeps log10 finite when a bin is empty.
		mag_db[k] = (m > 1e-15) ? 20.0 * std::log10(m) : -300.0;
	}
	return mag_db;
}

// Report structure for a single config run.
struct ResamplerMetrics {
	std::string config_name;   // "reference", "float", "posit32", ...
	std::string scalar_type;   // string repr of T

	std::size_t         output_length = 0;
	std::vector<double> tone_levels_db;      // one per params.tones
	double              passband_ripple_db =
		std::numeric_limits<double>::quiet_NaN();
	double              stopband_floor_db  =
		std::numeric_limits<double>::quiet_NaN();
	double              in_band_snr_db     =
		std::numeric_limits<double>::quiet_NaN();
};

ResamplerMetrics measure(const std::vector<double>& output,
                          const std::string& config_name,
                          const std::string& scalar_type) {
	ResamplerMetrics r;
	r.config_name   = config_name;
	r.scalar_type   = scalar_type;
	r.output_length = output.size();

	if (output.empty()) {
		r.tone_levels_db.assign(params.tones.size(),
		                        std::numeric_limits<double>::quiet_NaN());
		return r;
	}

	// Trim the FIR transient at head + tail (roughly filter_half_length *
	// max(L, M)). Use a generous 2000-sample skip both ends.
	constexpr std::size_t trim = 2000;
	if (output.size() < 2 * trim + 128) {
		// Not enough steady-state samples — leave metrics NaN.
		return r;
	}
	std::vector<double> steady(output.begin() + trim, output.end() - trim);

	auto mag_db = magnitude_spectrum_dbfs(steady, params.fft_size,
	                                       params.window_beta);
	if (mag_db.empty()) return r;

	const double bin_hz = params.output_rate_hz /
	                       static_cast<double>(params.fft_size);
	const double nyquist_hz = params.output_rate_hz / 2.0;

	// Peak-bin search: returns tone-index if this bin is within
	// tone_peak_hz of a tone.
	auto peak_index_for_bin = [&](std::size_t k) -> int {
		const double f = static_cast<double>(k) * bin_hz;
		for (std::size_t i = 0; i < params.tones.size(); ++i) {
			if (std::abs(f - params.tones[i]) <= params.tone_peak_hz) {
				return static_cast<int>(i);
			}
		}
		return -1;
	};
	// Guard classifier: true if this bin is within guard_hz of ANY tone
	// (i.e., in a tone's main-lobe + immediate side-lobe zone that must
	// be excluded from the noise floor).
	auto in_guard_band = [&](std::size_t k) {
		const double f = static_cast<double>(k) * bin_hz;
		for (double t : params.tones) {
			if (std::abs(f - t) <= params.guard_hz) return true;
		}
		return false;
	};

	// Tone levels: peak within +/- tone_peak_hz of each tone.
	r.tone_levels_db.assign(params.tones.size(), -300.0);
	for (std::size_t k = 1; k < mag_db.size(); ++k) {
		const int ti = peak_index_for_bin(k);
		if (ti >= 0 && mag_db[k] > r.tone_levels_db[ti]) {
			r.tone_levels_db[ti] = mag_db[k];
		}
	}
	// Passband ripple = spread of tone levels (dB).
	const double max_tone = *std::max_element(r.tone_levels_db.begin(),
	                                           r.tone_levels_db.end());
	const double min_tone = *std::min_element(r.tone_levels_db.begin(),
	                                           r.tone_levels_db.end());
	r.passband_ripple_db = max_tone - min_tone;

	// Stopband floor: max magnitude above `stopband_lower_hz` (only bins
	// below Nyquist matter — anything past is not part of the output
	// spectrum).
	double stopband_max = -300.0;
	for (std::size_t k = 1; k < mag_db.size(); ++k) {
		const double f = static_cast<double>(k) * bin_hz;
		if (f >= params.stopband_lower_hz && f < nyquist_hz) {
			stopband_max = std::max(stopband_max, mag_db[k]);
		}
	}
	r.stopband_floor_db = stopband_max;

	// In-band SNR: linear-power sum of guard-band bins (signal + window
	// leakage) vs. linear-power sum of NON-guard bins in [0,
	// passband_upper_hz] (arithmetic noise + distortion products). Skips
	// DC (k=0) since window coherent-DC reads high and isn't a
	// distortion component.
	double signal_pow = 0.0;
	double noise_pow  = 0.0;
	for (std::size_t k = 1; k < mag_db.size(); ++k) {
		const double f = static_cast<double>(k) * bin_hz;
		if (f > params.passband_upper_hz) break;
		const double lin = std::pow(10.0, mag_db[k] / 20.0);
		const double p   = lin * lin;
		if (in_guard_band(k)) signal_pow += p;
		else                  noise_pow  += p;
	}
	if (signal_pow > 0.0 && noise_pow > 0.0) {
		r.in_band_snr_db = 10.0 * std::log10(signal_pow / noise_pow);
	}

	return r;
}

// ============================================================================
// CSV writer
//
// Schema (long format - one row per (config, metric-or-tone)):
//   pipeline, config_name, scalar_type, metric_kind, tone_hz, value_db
//
// metric_kind is one of:
//   "tone"                 - value_db is the level at frequency tone_hz
//   "passband_ripple_db"   - value_db populated, tone_hz = NaN
//   "stopband_floor_db"    - value_db populated, tone_hz = NaN
//   "in_band_snr_db"       - value_db populated, tone_hz = NaN
//   "output_length"        - value_db populated (as an integer sample count),
//                             tone_hz = NaN
// ============================================================================

void write_csv(const std::string& path,
                const std::vector<ResamplerMetrics>& results) {
	std::ofstream out(path);
	if (!out) throw std::runtime_error("write_csv: cannot open " + path);
	out << "pipeline,config_name,scalar_type,metric_kind,tone_hz,value_db\n";
	auto emit = [&](const ResamplerMetrics& r, const std::string& kind,
	                 double tone_hz, double value_db) {
		out << "audio_resampler," << r.config_name << ","
		    << "\"" << r.scalar_type << "\"," << kind << ",";
		if (std::isnan(tone_hz)) out << "";
		else                     out << tone_hz;
		out << "," << value_db << "\n";
	};
	for (const auto& r : results) {
		for (std::size_t i = 0; i < r.tone_levels_db.size(); ++i) {
			emit(r, "tone", params.tones[i], r.tone_levels_db[i]);
		}
		emit(r, "passband_ripple_db",
		     std::numeric_limits<double>::quiet_NaN(), r.passband_ripple_db);
		emit(r, "stopband_floor_db",
		     std::numeric_limits<double>::quiet_NaN(), r.stopband_floor_db);
		emit(r, "in_band_snr_db",
		     std::numeric_limits<double>::quiet_NaN(), r.in_band_snr_db);
		emit(r, "output_length",
		     std::numeric_limits<double>::quiet_NaN(),
		     static_cast<double>(r.output_length));
	}
}

// ============================================================================
// Console summary
// ============================================================================

void print_summary(const std::vector<ResamplerMetrics>& results) {
	std::cout << "\n" << std::string(110, '=') << "\n";
	std::cout << std::left << std::setw(16) << "config"
	          << std::setw(18) << "scalar type"
	          << std::right << std::setw(11) << "100Hz/dB"
	          << std::setw(11) << "1kHz/dB"
	          << std::setw(11) << "10kHz/dB"
	          << std::setw(11) << "19kHz/dB"
	          << std::setw(11) << "ripple/dB"
	          << std::setw(13) << "stopband/dB"
	          << std::setw(9) << "SNR/dB"
	          << "\n" << std::string(110, '-') << "\n";
	for (const auto& r : results) {
		std::cout << std::left << std::setw(16) << r.config_name
		          << std::setw(18) << r.scalar_type
		          << std::right << std::fixed << std::setprecision(2);
		for (double t : r.tone_levels_db) std::cout << std::setw(11) << t;
		std::cout << std::setw(11) << r.passband_ripple_db
		          << std::setw(13) << r.stopband_floor_db
		          << std::setw(9) << r.in_band_snr_db << "\n";
	}
	std::cout << std::string(110, '=') << "\n";
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv) try {
	std::string csv_path = "audio_resampler.csv";
	for (int i = 1; i < argc; ++i) {
		std::string a = argv[i];
		if (a.rfind("--csv=", 0) == 0)      csv_path = a.substr(6);
		else if (a == "-h" || a == "--help") {
			std::cout << "Usage: " << argv[0] << " [--csv=path]\n";
			return 0;
		}
	}

	std::cout << "audio_resampler: 44.1 kHz -> 48 kHz mixed-precision demo\n"
	          << "  input rate:   " << params.input_rate_hz << " Hz\n"
	          << "  output rate:  " << params.output_rate_hz << " Hz\n"
	          << "  L / M:        " << params.L << " / " << params.M
	          << "   (gcd-reduced from 48000/44100)\n"
	          << "  input length: " << params.input_len << " samples ("
	          << static_cast<double>(params.input_len)
	             / params.input_rate_hz << " s)\n"
	          << "  tones:        ";
	for (std::size_t i = 0; i < params.tones.size(); ++i)
		std::cout << params.tones[i] << (i + 1 < params.tones.size() ? " / " : "");
	std::cout << " Hz (each amp " << params.tone_amp << ")\n";

	const auto input = generate_multitone(params.input_len, params.input_rate_hz,
	                                       params.tones, params.tone_amp);

	std::vector<ResamplerMetrics> results;

	// Standard six-config mp-comparison matrix (order matches
	// acquisition_demo so cross-demo scripts can align rows).
	auto run_one = [&](auto make_output, const std::string& name,
	                    const std::string& type_str) {
		std::cout << "  running " << name << " (" << type_str << ")..."
		          << std::flush;
		auto out = make_output();
		auto m = measure(out, name, type_str);
		std::cout << " out=" << m.output_length
		          << "  SNR=" << std::fixed << std::setprecision(2)
		          << m.in_band_snr_db << " dB\n";
		results.push_back(std::move(m));
	};

	run_one([&]{ return run_rational_resampler<double>(input); },
	         "reference", "double");
	run_one([&]{ return run_rational_resampler<float>(input); },
	         "float",     "float");
	run_one([&]{ return run_rational_resampler<p32>(input); },
	         "posit32",   "posit<32,2>");
	run_one([&]{ return run_rational_resampler<p16>(input); },
	         "posit16",   "posit<16,2>");
	run_one([&]{ return run_rational_resampler<cf32>(input); },
	         "cfloat32",  "cfloat<32,8>");
	run_one([&]{ return run_rational_resampler<fx32>(input); },
	         "fixpnt32",  "fixpnt<32,20>");

	print_summary(results);
	write_csv(csv_path, results);
	std::cout << "\nCSV written: " << csv_path << "\n";

	// Acceptance criterion: reference (double) SNR > 80 dB.
	const auto& ref = results.front();
	if (!(ref.in_band_snr_db > 80.0)) {
		std::cerr << "\nFAIL: reference in-band SNR " << ref.in_band_snr_db
		          << " dB is not > 80 dB (issue #136 acceptance criterion)\n";
		return 1;
	}
	std::cout << "\nAcceptance check: reference SNR "
	          << std::fixed << std::setprecision(2) << ref.in_band_snr_db
	          << " dB > 80 dB  [ok]\n";
	return 0;
} catch (const std::exception& ex) {
	std::cerr << "FATAL: " << ex.what() << "\n";
	return 1;
}
