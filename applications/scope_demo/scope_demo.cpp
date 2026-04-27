// scope_demo.cpp: end-to-end digital-oscilloscope mixed-precision sweep.
//
// Synthesizes a realistic test waveform — 50 MHz square wave with a 5 ns
// narrow positive glitch buried in it plus low-level AWGN — and runs it
// through the full scope DSP pipeline at multiple precisions:
//
//   simulate_adc(N_bits, sample_rate)
//        |
//        v
//   EdgeTrigger + AutoTriggerWrapper       (instrument/trigger.hpp)
//        |
//        v
//   TriggerRingBuffer (pre/post capture)   (instrument/ring_buffer.hpp)
//        |
//        v
//   PeakDetectDecimator                    (instrument/peak_detect.hpp)
//        |
//        v
//   render_envelope (-> N pixels)          (instrument/display_envelope.hpp)
//        |
//        v
//   measurements (rise time, RMS, ...)     (instrument/measurements.hpp)
//        |
//        v
//   scope_demo.csv + console summary
//
// Capstone for the Digital Oscilloscope Demonstrator epic (#133). The
// six-config sweep (uniform_double, uniform_float, uniform_posit32,
// uniform_posit16, uniform_cfloat32, uniform_fixpnt) lets the user see
// how each pipeline stage's quality varies with the streaming arithmetic.
//
// The headline acceptance is glitch survival: does the 5 ns positive
// glitch's peak amplitude appear in the display-rate output trace at the
// expected location? Numbers narrow enough to quantize the glitch below
// the surrounding noise floor will fail this — that's the lesson.
//
// Per-stage timing instrumentation produces a 10 GSPS comparison
// (informational): real 10 GSPS scopes use ASIC pipelines, not general-
// purpose CPUs. The number reported here is for understanding the gap.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/instrument/trigger.hpp>
#include <sw/dsp/instrument/ring_buffer.hpp>
#include <sw/dsp/instrument/peak_detect.hpp>
#include <sw/dsp/instrument/display_envelope.hpp>
#include <sw/dsp/instrument/measurements.hpp>

#include <sw/universal/number/posit/posit.hpp>
#include <sw/universal/number/cfloat/cfloat.hpp>
#include <sw/universal/number/fixpnt/fixpnt.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <vector>

using namespace sw::dsp::instrument;
namespace chrono = std::chrono;

// ============================================================================
// Type aliases for the sweep
// ============================================================================

using p32  = sw::universal::posit<32, 2>;
using p16  = sw::universal::posit<16, 2>;
using cf32 = sw::universal::cfloat<32, 8, std::uint32_t, true, false, false>;
// Q2.30: 2 integer bits (signal amplitude is +-0.5, leaves 4x headroom)
// and 30 fractional bits (~30 ENOB ceiling). fixpnt<32,28> would be Q4.28
// — more integer headroom than this signal needs, fewer fractional bits.
using fx32 = sw::universal::fixpnt<32, 30>;

// ============================================================================
// Pipeline parameters
// ============================================================================

struct PipelineParams {
	// Signal: 50 MHz square wave, +-0.5 amplitude (well inside posit16 sweet
	// spot). Low-level AWGN keeps the trigger reliable but doesn't dominate.
	double      sample_rate_hz = 1e9;       // 1 GSPS sim (10 GSPS is the perf
	                                        // target, not the actual rate)
	double      signal_freq_hz = 50e6;      // 50 MHz square wave
	double      signal_amp     = 0.5;
	double      noise_rms      = 0.005;
	int         adc_bits       = 12;        // 12-bit ADC

	// Glitch: a 5 ns positive overshoot that overrides the carrier (a
	// scope probe getting hit by an EMI pulse). The amplitude is the
	// absolute glitch peak, not an addition — so the glitch is well-
	// defined regardless of what the carrier was doing underneath.
	//
	// Placed at 500 ns into the stream so it lands AFTER the analytical
	// measurement window (rise_time / frequency are computed on the
	// pre-glitch region; the full segment is used for envelope and
	// peak_to_peak).
	double      glitch_peak       = 0.95;   // absolute glitch peak amplitude
	                                        // (must be < 1.0 so the 12-bit
	                                        // ADC's [-1, +1) range doesn't
	                                        // clamp it)
	double      glitch_width_s    = 5e-9;   // 5 ns wide
	double      glitch_offset_s   = 5e-7;   // 500 ns into the stream
	std::size_t pre_glitch_window = 400;    // samples used for rise_time /
	                                        // period / frequency. Must end
	                                        // before glitch_offset_s.

	// Capture
	std::size_t pre_trigger      = 256;
	std::size_t post_trigger     = 768;
	double      trigger_level    = 0.0;
	double      trigger_hyst     = 0.05;    // suppress noise re-triggers
	std::size_t auto_trigger_to  = 4000;    // force-fire if no edge

	// Display reduction
	std::size_t peak_detect_R    = 2;       // first decimation
	std::size_t pixel_width      = 200;     // final display columns

	// Stream length
	std::size_t num_samples      = 1024 * 8;
};
inline PipelineParams params;

// ============================================================================
// ADC simulation: 50 MHz square wave + narrow positive glitch + AWGN, then
// quantized to N bits.
// ============================================================================

std::vector<double> simulate_adc(unsigned seed = 0xACDC) {
	std::vector<double> samples(params.num_samples);
	std::mt19937 rng(seed);
	std::normal_distribution<double> noise(0.0, params.noise_rms);

	const double half_levels = std::ldexp(1.0, params.adc_bits - 1);
	const double q_step      = 1.0 / half_levels;
	const double code_max    = half_levels - 1.0;
	const double code_min    = -half_levels;

	const double dt          = 1.0 / params.sample_rate_hz;
	const double glitch_t0   = params.glitch_offset_s;
	const double glitch_t1   = glitch_t0 + params.glitch_width_s;

	// Integer-phase square wave: avoid `sin(2*pi*f*t) >= 0`, which is
	// numerically unstable at sample boundaries where sin(k*pi) returns
	// tiny FP-noise values that flip sign unpredictably (e.g., sin(6*pi)
	// on x86 came out positive on this build, shortening one low half-
	// cycle by a sample and biasing period measurements).
	const std::size_t half_period_samples =
		static_cast<std::size_t>(std::round(
			0.5 * params.sample_rate_hz / params.signal_freq_hz));
	const std::size_t cycle_samples = 2 * half_period_samples;

	for (std::size_t n = 0; n < params.num_samples; ++n) {
		const double t = static_cast<double>(n) * dt;
		const std::size_t phase_n = n % cycle_samples;
		const double sq = (phase_n < half_period_samples)
		                   ? params.signal_amp : -params.signal_amp;
		// Glitch overrides the carrier during its window (well-defined peak
		// regardless of carrier state at glitch time).
		const double clean = (t >= glitch_t0 && t < glitch_t1)
		                      ? params.glitch_peak : sq;
		const double noisy = clean + noise(rng);

		double code = std::floor(noisy / q_step);
		code = std::clamp(code, code_min, code_max);
		samples[n] = code * q_step;
	}
	return samples;
}

// ============================================================================
// Stage timings for a single pipeline run
// ============================================================================

struct StageTimingsNs {
	double trigger_ring   = 0.0;
	double peak_detect    = 0.0;
	double envelope       = 0.0;
	double measurements   = 0.0;
	double total          = 0.0;
};

// ============================================================================
// Per-config result
// ============================================================================

struct ConfigResult {
	std::string config_name;
	std::string coeff_type;
	std::string state_type;
	std::string sample_type;

	// Headline metrics. Numeric fields default to NaN so a config that
	// fails to capture (no trigger fired in the input window) prints as
	// "not available" rather than as a column of zeros that look like
	// legitimate measurements.
	bool        glitch_survived      = false;
	double      glitch_peak_observed =
		std::numeric_limits<double>::quiet_NaN();
	double      rise_time_samples    =
		std::numeric_limits<double>::quiet_NaN();
	double      rise_time_expected   =
		std::numeric_limits<double>::quiet_NaN();
	double      rms                  =
		std::numeric_limits<double>::quiet_NaN();
	double      mean                 =
		std::numeric_limits<double>::quiet_NaN();
	double      period_samples       =
		std::numeric_limits<double>::quiet_NaN();
	double      frequency_hz         =
		std::numeric_limits<double>::quiet_NaN();
	double      output_snr_db        =
		std::numeric_limits<double>::quiet_NaN();
	std::size_t captured_length      = 0;

	StageTimingsNs timings;

	// Per-pixel envelope (mins, maxs) — used to write the CSV. Cast to
	// double regardless of SampleScalar precision so we have one schema.
	std::vector<double> envelope_min;
	std::vector<double> envelope_max;
};

// ============================================================================
// run_pipeline<SampleScalar>
//
// One scope pipeline run at a single precision config. Returns a
// ConfigResult with both the headline measurements and the rendered
// envelope. The trigger/ring/peak-detect/envelope chain is parameterized
// solely on SampleScalar — these primitives don't take CoeffScalar /
// StateScalar separately because they don't have filter coefficients.
// (The spec lists three scalars to keep the type signature uniform with
// the rest of the library; for scope_demo only Sample matters.)
// ============================================================================

template <class SampleScalar>
ConfigResult run_pipeline(const std::vector<double>& adc_in_double,
                          const std::string& config_name) {
	ConfigResult result;
	result.config_name = config_name;
	result.coeff_type  = config_name;   // uniform — keep the column for schema
	result.state_type  = config_name;
	result.sample_type = config_name;

	// Project ADC samples into SampleScalar.
	std::vector<SampleScalar> adc_in(adc_in_double.size());
	for (std::size_t n = 0; n < adc_in_double.size(); ++n)
		adc_in[n] = static_cast<SampleScalar>(adc_in_double[n]);

	// Stage 1: trigger + ring buffer in lockstep.
	auto t0 = chrono::high_resolution_clock::now();
	EdgeTrigger<SampleScalar> trig(static_cast<SampleScalar>(params.trigger_level),
	                                Slope::Rising,
	                                static_cast<SampleScalar>(params.trigger_hyst));
	AutoTriggerWrapper<EdgeTrigger<SampleScalar>>
		auto_trig(trig, params.auto_trigger_to);
	TriggerRingBuffer<SampleScalar> ring(params.pre_trigger, params.post_trigger);

	bool triggered = false;
	for (std::size_t n = 0; n < adc_in.size(); ++n) {
		const SampleScalar x = adc_in[n];
		// Only the FIRST fire goes to push_trigger. Subsequent edges of
		// the inner trigger are ignored — push_trigger() is a no-op in
		// Capturing state, and routing the sample there instead of via
		// push() would silently drop it from the captured segment
		// (one sample per re-trigger).
		const bool fire = auto_trig.process(x);
		if (!triggered && fire) {
			ring.push_trigger(x);
			triggered = true;
		} else {
			ring.push(x);
		}
		if (ring.capture_complete()) break;
	}
	auto t1 = chrono::high_resolution_clock::now();
	result.timings.trigger_ring =
		chrono::duration<double, std::nano>(t1 - t0).count();

	if (!ring.capture_complete()) {
		// No trigger fired — leave envelope empty and bail. The console
		// summary will show this as zero glitch survival, NaN rise time.
		return result;
	}
	auto segment = ring.captured_segment();
	result.captured_length = segment.size();

	// Stage 2: peak-detect decimation. R=peak_detect_R; R=1 is passthrough.
	t0 = chrono::high_resolution_clock::now();
	PeakDetectDecimator<SampleScalar> pd(params.peak_detect_R);
	auto pd_env = pd.process_block(segment);
	t1 = chrono::high_resolution_clock::now();
	result.timings.peak_detect =
		chrono::duration<double, std::nano>(t1 - t0).count();

	// Stage 3: render envelope to pixel_width columns. We render the MAX
	// stream from peak_detect (the stream that preserves positive glitches).
	// For a true scope display you'd render both mins and maxs as a vertical
	// line per pixel; for this demo we focus on the max stream because
	// that's where the positive glitch shows up.
	t0 = chrono::high_resolution_clock::now();
	std::span<const SampleScalar> max_span(pd_env.maxs.data(), pd_env.maxs.size());
	std::span<const SampleScalar> min_span(pd_env.mins.data(), pd_env.mins.size());
	auto disp_max = render_envelope<SampleScalar>(max_span, params.pixel_width);
	auto disp_min = render_envelope<SampleScalar>(min_span, params.pixel_width);
	t1 = chrono::high_resolution_clock::now();
	result.timings.envelope =
		chrono::duration<double, std::nano>(t1 - t0).count();

	// Stage 4: measurements.
	//
	// rms / mean / glitch-peak read the full segment — the glitch is part
	// of what the user wants to see in those numbers (RMS is elevated
	// slightly, glitch peak is the headline).
	//
	// rise_time / period / frequency read a glitch-free pre-window. The
	// glitch's leading edge would otherwise create an extra zero-crossing
	// (period bias) and push the 90% threshold above the carrier
	// amplitude (rise time measures from carrier to glitch). The
	// pre-glitch window contains several clean carrier cycles.
	t0 = chrono::high_resolution_clock::now();
	result.rms = rms<SampleScalar>(segment);
	result.mean = mean<SampleScalar>(segment);
	const std::size_t mw_len =
		std::min(segment.size(), params.pre_glitch_window);
	auto measurement_window = segment.subspan(0, mw_len);
	result.rise_time_samples =
		rise_time_samples<SampleScalar>(measurement_window, 0.1, 0.9);
	result.period_samples =
		period_samples<SampleScalar>(measurement_window,
		                              static_cast<SampleScalar>(0));
	result.frequency_hz =
		frequency_hz<SampleScalar>(measurement_window, params.sample_rate_hz,
		                            static_cast<SampleScalar>(0));
	t1 = chrono::high_resolution_clock::now();
	result.timings.measurements =
		chrono::duration<double, std::nano>(t1 - t0).count();

	result.timings.total = result.timings.trigger_ring
	                     + result.timings.peak_detect
	                     + result.timings.envelope
	                     + result.timings.measurements;

	// Glitch survival: scan the rendered MAX-envelope for a peak that
	// significantly exceeds the carrier amplitude. The carrier maxes at
	// +signal_amp; the glitch tops out at glitch_peak. We declare
	// "survived" when the observed peak >= midway between the two
	// (a generous tolerance — a faithful pipeline recovers the full
	// glitch_peak, but peak-detect + render_envelope can mildly attenuate
	// a sub-window glitch when the decimation factor is high).
	double peak = -1e9;
	for (std::size_t i = 0; i < disp_max.maxs.size(); ++i)
		peak = std::max(peak, static_cast<double>(disp_max.maxs[i]));
	result.glitch_peak_observed = peak;
	const double glitch_threshold =
		0.5 * (params.signal_amp + params.glitch_peak);
	result.glitch_survived = peak >= glitch_threshold;

	// Stash the envelope for CSV writing. Cast to double for a uniform
	// schema; the CSV is the comparison surface across configs.
	result.envelope_min.assign(disp_min.mins.size(), 0.0);
	result.envelope_max.assign(disp_max.maxs.size(), 0.0);
	for (std::size_t i = 0; i < disp_max.maxs.size(); ++i) {
		result.envelope_min[i] = static_cast<double>(disp_min.mins[i]);
		result.envelope_max[i] = static_cast<double>(disp_max.maxs[i]);
	}

	return result;
}

// ============================================================================
// Output trace SNR vs the uniform_double reference
// ============================================================================

double snr_db_against_reference(const ConfigResult& test,
                                 const ConfigResult& ref) {
	if (ref.envelope_max.size() != test.envelope_max.size()
	    || ref.envelope_max.empty())
		return std::numeric_limits<double>::quiet_NaN();
	double sig_pow = 0.0, err_pow = 0.0;
	for (std::size_t i = 0; i < ref.envelope_max.size(); ++i) {
		const double s = ref.envelope_max[i];
		const double e = ref.envelope_max[i] - test.envelope_max[i];
		sig_pow += s * s;
		err_pow += e * e;
	}
	if (err_pow <= 0.0) return std::numeric_limits<double>::infinity();
	return 10.0 * std::log10(sig_pow / err_pow);
}

// ============================================================================
// CSV output
// ============================================================================

void write_csv(const std::string& path,
               std::span<const ConfigResult> results) {
	std::ofstream out(path);
	if (!out) {
		std::cerr << "warn: could not open '" << path << "' for write\n";
		return;
	}
	// Schema: pipeline,config_name,coeff_type,state_type,sample_type,
	//         pixel_index,envelope_min,envelope_max,
	//         glitch_survived,glitch_peak,rise_time,rms,mean,output_snr_db
	out << "pipeline,config_name,coeff_type,state_type,sample_type,"
	       "pixel_index,envelope_min,envelope_max,"
	       "glitch_survived,glitch_peak,rise_time_samples,rms,mean,output_snr_db\n";
	for (const auto& r : results) {
		for (std::size_t i = 0; i < r.envelope_max.size(); ++i) {
			out << "scope_demo,"
			    << r.config_name << ','
			    << r.coeff_type  << ','
			    << r.state_type  << ','
			    << r.sample_type << ','
			    << i << ','
			    << r.envelope_min[i] << ','
			    << r.envelope_max[i] << ','
			    << (r.glitch_survived ? 1 : 0) << ','
			    << r.glitch_peak_observed << ','
			    << r.rise_time_samples << ','
			    << r.rms << ','
			    << r.mean << ','
			    << r.output_snr_db
			    << '\n';
		}
	}
}

// ============================================================================
// Console summary table
// ============================================================================

void print_summary_header() {
	std::cout << "\n=== Mixed-precision scope sweep ===\n";
	std::cout << std::left  << std::setw(18) << "config"
	          << std::right << std::setw(10) << "glitch?"
	          << std::right << std::setw(12) << "peak"
	          << std::right << std::setw(12) << "rise(samp)"
	          << std::right << std::setw(10) << "rms"
	          << std::right << std::setw(12) << "freq(MHz)"
	          << std::right << std::setw(12) << "SNR(dB)"
	          << "\n";
	std::cout << std::string(18 + 10 + 12 + 12 + 10 + 12 + 12, '-') << "\n";
}

void print_summary_row(const ConfigResult& r) {
	std::cout << std::left  << std::setw(18) << r.config_name
	          << std::right << std::setw(10) << (r.glitch_survived ? "PASS" : "fail")
	          << std::right << std::setw(12) << std::fixed;
	auto print_or_nan = [](double v, int prec, double scale = 1.0) {
		if (std::isnan(v)) std::cout << "NaN";
		else std::cout << std::setprecision(prec) << (v * scale);
	};
	print_or_nan(r.glitch_peak_observed, 3);
	std::cout << std::right << std::setw(12);
	print_or_nan(r.rise_time_samples, 2);
	std::cout << std::right << std::setw(10);
	print_or_nan(r.rms, 3);
	std::cout << std::right << std::setw(12);
	print_or_nan(r.frequency_hz, 3, 1.0 / 1e6);
	std::cout << std::right << std::setw(12);
	print_or_nan(r.output_snr_db, 2);
	std::cout << "\n";
}

// ============================================================================
// Per-stage timing report — 10 GSPS comparison
// ============================================================================

void print_timing_report(const ConfigResult& reference) {
	std::cout << "\n=== Per-stage timing (uniform_double reference) ===\n";
	const double samples = static_cast<double>(reference.captured_length);
	if (samples <= 0.0) {
		std::cout << "  (no capture — timing skipped)\n";
		return;
	}
	auto print = [&](const char* name, double ns_total) {
		const double ns_per_sample = ns_total / samples;
		std::cout << "  " << std::left << std::setw(18) << name
		          << std::right << std::setw(12) << std::fixed
		          << std::setprecision(2) << ns_total << " ns total"
		          << std::right << std::setw(12) << std::setprecision(3)
		          << ns_per_sample << " ns/sample"
		          << "\n";
	};
	print("trigger+ring",   reference.timings.trigger_ring);
	print("peak_detect",    reference.timings.peak_detect);
	print("render_envelope",reference.timings.envelope);
	print("measurements",   reference.timings.measurements);
	print("TOTAL",          reference.timings.total);

	// 10 GSPS = 0.1 ns / sample budget end-to-end.
	const double total_per_sample = reference.timings.total / samples;
	const double target_ns_per_sample = 0.1;   // 1e9 / 10e9
	std::cout << "\n  10 GSPS budget: 0.100 ns/sample\n";
	if (total_per_sample <= target_ns_per_sample) {
		std::cout << "  10 GSPS: ACHIEVED ("
		          << std::fixed << std::setprecision(3)
		          << total_per_sample << " ns/sample)\n";
	} else {
		const double speedup_needed = total_per_sample / target_ns_per_sample;
		std::cout << "  10 GSPS: NOT achievable on general-purpose CPU "
		             "(would need " << std::fixed << std::setprecision(1)
		          << speedup_needed << "x speedup)\n";
		std::cout << "  Real 10 GSPS scopes use ASIC pipelines — this is an "
		             "informational comparison.\n";
	}
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) try {
	std::string csv_path = "scope_demo.csv";
	for (int i = 1; i < argc; ++i) {
		std::string a = argv[i];
		if (a.rfind("--csv=", 0) == 0)
			csv_path = a.substr(6);
		else if (a == "-h" || a == "--help") {
			std::cout << "Usage: " << argv[0] << " [--csv=path]\n";
			return 0;
		}
	}

	std::cout << "scope_demo: digital-oscilloscope mixed-precision sweep\n"
	          << "  signal:  50 MHz square wave +- " << params.signal_amp
	          << " (5 ns +" << params.glitch_peak << " glitch at "
	          << params.glitch_offset_s * 1e9 << " ns)\n"
	          << "  ADC:     " << params.adc_bits << "-bit, "
	          << params.sample_rate_hz / 1e9 << " GSPS, "
	          << params.num_samples << " samples\n"
	          << "  capture: " << params.pre_trigger << " pre + 1 trigger + "
	          << params.post_trigger << " post\n"
	          << "  display: peak-detect R=" << params.peak_detect_R
	          << ", " << params.pixel_width << " pixels\n";

	// Compute analytical rise time for the carrier transition: a square
	// wave's rising edge is one sample wide at 1 GSPS / 50 MHz (it goes
	// from -amp to +amp in one sample), so the 10/90 rise time is 0.8
	// samples (80% of one sample step). The pipeline's rise time should
	// recover this within roughly half a sample.
	const double expected_rise_samples = 0.8;

	const auto adc = simulate_adc();

	// Run the six configurations.
	std::array<ConfigResult, 6> results{{
		run_pipeline<double>(adc, "uniform_double"),
		run_pipeline<float>(adc,  "uniform_float"),
		run_pipeline<p32>(adc,    "uniform_posit32"),
		run_pipeline<p16>(adc,    "uniform_posit16"),
		run_pipeline<cf32>(adc,   "uniform_cfloat32"),
		run_pipeline<fx32>(adc,   "uniform_fixpnt"),
	}};

	// SNR vs uniform_double reference (results[0]).
	for (auto& r : results) {
		r.rise_time_expected = expected_rise_samples;
		r.output_snr_db = snr_db_against_reference(r, results[0]);
	}

	print_summary_header();
	for (const auto& r : results) print_summary_row(r);

	print_timing_report(results[0]);

	write_csv(csv_path, results);
	std::cout << "\nCSV written: " << csv_path << "\n";

	return 0;
} catch (const std::exception& ex) {
	std::cerr << "FATAL: " << ex.what() << "\n";
	return 1;
}
