// spectrum_analyzer_demo.cpp: end-to-end spectrum-analyzer mixed-precision sweep.
//
// Capstone for the Spectrum Analyzer Demonstrator epic (#134). Wires the
// nine spectrum/* primitives shipped over the v0.7 milestone into both
// of the canonical analyzer architectures and runs each architecture at
// four precision plans:
//
//                                  +---------------------+
//   simulate_input(double[])  -->  | EqualizerFilter     |  -- the
//   (multi-tone + 2nd harmonic     | (FrontEndCorrector) |     precision-
//    + AWGN)                       +---------------------+     sensitive
//                                            |                  stage
//                                            v
//                       +-----------+----------------+
//                       |                            |
//                  FFT path                  Swept-tuned path
//             (RealtimeSpectrum)               (mixer + RBW + det)
//                       |                            |
//                       v                            v
//                  TraceAverager                VBWFilter
//                       |                            |
//                       v                            v
//                  WaterfallBuffer            trace[bin] memory
//                       |                            |
//                       v                            v
//                  find_peaks                   find_peaks
//                  harmonic_markers             harmonic_markers
//                       |                            |
//                       +-------------+--------------+
//                                     |
//                                     v
//                               headline metrics
//                               + per-stage timings
//                               + CSV row
//
// Mixed-precision design (the WHOLE point of this demo):
//
//   The digital arithmetic in BOTH analyzer paths happens in two places
//   with different sensitivity profiles:
//
//     1. EqualizerFilter (FIR multiply-accumulate): unconditional
//        precision sensitivity — every sample touches it.
//     2. FFT butterflies / mixer-RBW-detector cascade: precision
//        sensitivity stacks across the long pipeline. The FFT's twiddles
//        are the dominant precision driver for the FFT path; the swept-
//        tuned path's sensitivity is concentrated in the RBW biquad
//        cascade and the IIR VBW filter.
//
//   Trace memory, marker scoring, and waterfall storage are comparison-
//   only or copy-only — narrowing those wouldn't move the precision
//   needle but would drop memory bandwidth. Same dynamic as the scope
//   demo, but with the FFT and the RBW/VBW now in the precision path
//   instead of the EQ alone.
//
//   To keep the template-instantiation explosion bounded (4 plans x 2
//   paths = 8 instantiations), each pipeline is parameterized on a
//   single `ArithScalar` that drives every numerically-active stage in
//   that path. The "reference" plan uses double everywhere (the SNR
//   baseline); the other three plans pick float / posit32 / posit16 for
//   ArithScalar.
//
// Headline metrics (per plan, per path):
//   - Frequency error of the dominant 3-tone group, in ppm.
//   - Amplitude error of the dominant 3-tone group, in dB.
//   - Noise-floor recovered (mean trace in tone-free bins, dB).
//   - 2nd-harmonic detected at -80 dBc (yes/no + observed amplitude).
//   - Output trace SNR in dB vs. the reference (double) plan.
//
// Per-stage timing report runs against the reference plan for the
// 10 GSPS budget comparison (informational; same convention as
// scope_demo).
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/spectrum/front_end_corrector.hpp>
#include <sw/dsp/spectrum/realtime_spectrum.hpp>
#include <sw/dsp/spectrum/trace_averaging.hpp>
#include <sw/dsp/spectrum/waterfall_buffer.hpp>
#include <sw/dsp/spectrum/swept_lo.hpp>
#include <sw/dsp/spectrum/rbw_filter.hpp>
#include <sw/dsp/spectrum/vbw_filter.hpp>
#include <sw/dsp/spectrum/detectors.hpp>
#include <sw/dsp/spectrum/markers.hpp>
#include <sw/dsp/windows/hanning.hpp>

#include <sw/universal/number/posit/posit.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace chrono = std::chrono;
using namespace sw::dsp;

// ============================================================================
// Type aliases for the precision plans
// ============================================================================

using p32 = sw::universal::posit<32, 2>;
using p16 = sw::universal::posit<16, 2>;

// ============================================================================
// Pipeline parameters
// ============================================================================

struct PipelineParams {
	// Sample rate. 2 MHz keeps the scenario CPU-tractable while leaving
	// enough headroom (Nyquist = 1 MHz) for the 50/100/150 kHz tones plus
	// their 2nd harmonics inside the band.
	double      sample_rate_hz = 2.0e6;

	// Three-tone test signal — the analyzer's analytical headline. The
	// strongest tone (50 kHz) drives the noise-floor measurement; the
	// other two are at progressively lower amplitudes (-30, -60 dBc) so
	// the dynamic range exercise spans 60 dB.
	double      tone1_hz = 50.0e3;    double tone1_dbc = 0.0;     // reference
	double      tone2_hz = 100.0e3;   double tone2_dbc = -30.0;
	double      tone3_hz = 150.0e3;   double tone3_dbc = -60.0;

	// Low-level discrete spurious planted at -80 dBc — a deep
	// dynamic-range probe that distinguishes the reference (double)
	// plan from posit16. Placed at 175 kHz: not coincident with any
	// of the three planted tones, not a harmonic of any of them
	// (avoids hiding under a stronger feature). If the analyzer's
	// noise floor is too high (or its precision too narrow) the
	// spurious disappears.
	double      spurious_hz   = 175.0e3;
	double      spurious_dbc  = -80.0;

	// Noise floor of the synthetic source, in linear amplitude (RMS).
	// Sets the absolute SNR ceiling for any plan; should be lower than
	// the deepest planted feature so the harmonic stays visible.
	double      noise_rms = 1.0e-5;

	// Stream length. Long enough to fill the FFT ring multiple times
	// AND give the swept-tuned path enough dwell-per-bin time for the
	// RBW filter to settle. At 64 swept bins and 60k samples per
	// sweep, dwell_per_bin = ~470 us — about 2.3 RBW (5 kHz) time
	// constants, enough for the dominant tones to register cleanly
	// and the swept noise floor to drop below the planted -60 dBc
	// tone3 amplitude.
	std::size_t num_samples = 1u << 16;     // 65536 samples = ~33 ms

	// FFT path
	std::size_t fft_size = 4096;
	std::size_t hop_size = 2048;            // 50% overlap
	std::size_t avg_n    = 4;               // exponential averaging alpha = 1/avg_n
	std::size_t waterfall_frames = 16;

	// Swept-tuned path. The trace bin count is chosen so each LO
	// dwell window is long enough for the RBW filter to substantially
	// settle: dwell_per_bin = num_samples / swept_trace_bins / sweeps.
	// At 32k samples / 64 bins / ~2 sweeps that's ~250 samples per
	// dwell at fs = 2 MHz = 125 us, which is comparable to the RBW
	// (5 kHz) settling time of ~200 us. Not perfect, but enough that
	// the dominant tones light up cleanly. RBW=5 kHz comfortably
	// resolves the 50 kHz tone separation.
	double      swept_f_start_hz = 10.0e3;
	double      swept_f_stop_hz  = 240.0e3;     // covers tones + spurious at 175k
	double      swept_duration_s = 30.0e-3;     // 60k samples at 2 MHz
	double      rbw_hz           = 5.0e3;       // 5 kHz RBW (BW < tone separation)
	double      vbw_hz           = 500.0;       // 10:1 below RBW (smooth detector)
	std::size_t rbw_order        = 5;           // ~10x shape factor
	std::size_t swept_trace_bins = 64;          // memory for the swept trace

	// Equalizer (front-end correction)
	std::size_t eq_taps = 31;
};
inline PipelineParams params;

// ============================================================================
// simulate_input: 3-tone + 2nd-harmonic + AWGN, in double
// ============================================================================

std::vector<double> simulate_input(unsigned seed = 0xBEEF) {
	std::vector<double> x(params.num_samples);
	std::mt19937 rng(seed);
	std::normal_distribution<double> noise(0.0, params.noise_rms);

	auto dbc_to_lin = [](double dbc) { return std::pow(10.0, dbc / 20.0); };

	const double a1 = dbc_to_lin(params.tone1_dbc);
	const double a2 = dbc_to_lin(params.tone2_dbc);
	const double a3 = dbc_to_lin(params.tone3_dbc);
	const double as = dbc_to_lin(params.spurious_dbc);
	const double w1 = 2.0 * M_PI * params.tone1_hz     / params.sample_rate_hz;
	const double w2 = 2.0 * M_PI * params.tone2_hz     / params.sample_rate_hz;
	const double w3 = 2.0 * M_PI * params.tone3_hz     / params.sample_rate_hz;
	const double ws = 2.0 * M_PI * params.spurious_hz  / params.sample_rate_hz;

	for (std::size_t n = 0; n < x.size(); ++n) {
		const double t = static_cast<double>(n);
		x[n] = a1 * std::sin(w1 * t)
		     + a2 * std::sin(w2 * t)
		     + a3 * std::sin(w3 * t)
		     + as * std::sin(ws * t)
		     + noise(rng);
	}
	return x;
}

// ============================================================================
// Synthetic analog-front-end calibration profile.
//
// Same shape as the scope demo's: mild rolloff with a small phase
// signature across the analyzer's band of interest. The equalizer's
// inverse correction adds a few dB of high-band boost — enough to
// exercise the precision-sensitive FIR multiply-accumulate without
// overshooting beyond what the demo's measurements can interpret.
// ============================================================================

spectrum::CalibrationProfile make_test_profile() {
	std::vector<double> freqs    = {0.0,    50e3,  100e3, 250e3, 500e3, 1e6};
	std::vector<double> gains_dB = {0.0,   -0.2,   -0.5,  -1.0,  -2.0, -3.0};
	std::vector<double> phases   = {0.0,   -0.05,  -0.10, -0.20, -0.30, -0.40};
	return spectrum::CalibrationProfile(std::move(freqs),
	                                     std::move(gains_dB),
	                                     std::move(phases));
}

// ============================================================================
// Per-stage timings for one pipeline run
// ============================================================================

struct StageTimingsNs {
	double equalizer    = 0.0;
	double spectrum     = 0.0;   // FFT path: RealtimeSpectrum.push() incl. windowing
	                              // Swept path: mixer + RBW + detector
	double trace_avg    = 0.0;   // FFT path only (trace averaging)
	double vbw          = 0.0;   // Swept path only
	double waterfall    = 0.0;   // FFT path only
	double markers      = 0.0;
	double total        = 0.0;
};

// ============================================================================
// Per-config result
// ============================================================================

struct ConfigResult {
	std::string plan_name;          // e.g., "reference", "arith_posit16"
	std::string path;               // "fft" or "swept"
	std::string arith_type;         // double / float / posit32 / posit16
	std::size_t arith_bytes_per_sample = 0;

	// Headline metrics (NaN means the metric isn't available for this
	// path or this plan didn't capture).
	double tone1_freq_hz       = std::numeric_limits<double>::quiet_NaN();
	double tone1_amp_db        = std::numeric_limits<double>::quiet_NaN();
	double tone2_freq_hz       = std::numeric_limits<double>::quiet_NaN();
	double tone2_amp_db        = std::numeric_limits<double>::quiet_NaN();
	double tone3_freq_hz       = std::numeric_limits<double>::quiet_NaN();
	double tone3_amp_db        = std::numeric_limits<double>::quiet_NaN();
	double spurious_amp_db     = std::numeric_limits<double>::quiet_NaN();
	bool   spurious_detected   = false;
	double noise_floor_db      = std::numeric_limits<double>::quiet_NaN();
	double output_snr_db       = std::numeric_limits<double>::quiet_NaN();

	// Final trace (in dB, length = trace size for that path) — used to
	// compute SNR vs the reference plan AND to write the CSV.
	std::vector<double> trace_db;
	double              trace_freq_step_hz = 0.0;

	StageTimingsNs timings;
};

// ============================================================================
// Helpers for marker amplitude / freq error reporting
// ============================================================================

// Find the marker among `markers` whose frequency is closest to
// `target_hz` AND within `tolerance_hz` of it. Returns a NaN-Marker if
// no candidate qualifies — important for the swept path where wide
// dynamic-range gaps can cause find_peaks to miss a tone entirely;
// without the tolerance window, nearest_marker would mis-attribute
// some unrelated peak to the missing tone.
spectrum::Marker nearest_marker(std::span<const spectrum::Marker> markers,
                                 double target_hz,
                                 double tolerance_hz) {
	spectrum::Marker best;
	best.frequency_hz = std::numeric_limits<double>::quiet_NaN();
	best.amplitude    = std::numeric_limits<double>::quiet_NaN();
	double best_err   = std::numeric_limits<double>::infinity();
	for (const auto& m : markers) {
		const double err = std::abs(m.frequency_hz - target_hz);
		if (err < best_err) { best_err = err; best = m; }
	}
	if (best_err > tolerance_hz) {
		// No marker within tolerance — return NaN-marker so the headline
		// metric prints as NaN ("not detected") rather than as a bogus
		// frequency / amplitude.
		spectrum::Marker none;
		none.frequency_hz = std::numeric_limits<double>::quiet_NaN();
		none.amplitude    = std::numeric_limits<double>::quiet_NaN();
		return none;
	}
	return best;
}

// Mean trace value (in dB) over bins that are NOT within `guard_bins`
// of any of the planted features. Used as the noise-floor estimate.
double noise_floor_db(std::span<const double> trace_db,
                       double bin_freq_step_hz,
                       std::initializer_list<double> feature_freqs,
                       std::size_t guard_bins,
                       std::size_t bin_lo,
                       std::size_t bin_hi) {
	double sum = 0.0;
	std::size_t n = 0;
	for (std::size_t i = bin_lo; i < std::min(bin_hi, trace_db.size()); ++i) {
		const double f = static_cast<double>(i) * bin_freq_step_hz;
		bool near_feature = false;
		for (double ff : feature_freqs) {
			const double bins_off = std::abs(f - ff) / bin_freq_step_hz;
			if (bins_off < static_cast<double>(guard_bins)) {
				near_feature = true; break;
			}
		}
		if (near_feature) continue;
		sum += trace_db[i]; ++n;
	}
	return n > 0 ? sum / static_cast<double>(n)
	             : std::numeric_limits<double>::quiet_NaN();
}

// ============================================================================
// FFT-path pipeline: input -> EqualizerFilter -> RealtimeSpectrum ->
//                    TraceAverager (Exponential) -> WaterfallBuffer ->
//                    find_peaks / harmonic_markers
//
// One ArithScalar drives the equalizer and the FFT/window arithmetic.
// Trace storage and marker scoring stay in double for simplicity.
// ============================================================================

template <class ArithScalar>
ConfigResult run_fft_path(std::span<const double> input_double,
                           const std::string& plan_name,
                           const std::string& arith_tag,
                           std::size_t arith_bytes,
                           const spectrum::CalibrationProfile& profile) {
	ConfigResult result;
	result.plan_name              = plan_name;
	result.path                   = "fft";
	result.arith_type             = arith_tag;
	result.arith_bytes_per_sample = arith_bytes;

	// --- Stage 1: equalizer in ArithScalar ---
	auto t0 = chrono::high_resolution_clock::now();
	spectrum::FrontEndCorrector<ArithScalar> eq(
		profile, params.eq_taps, params.sample_rate_hz);
	std::vector<ArithScalar> equalized(input_double.size());
	for (std::size_t n = 0; n < input_double.size(); ++n) {
		const ArithScalar x = static_cast<ArithScalar>(input_double[n]);
		equalized[n] = eq.process(x);
	}
	auto t1 = chrono::high_resolution_clock::now();
	result.timings.equalizer =
		chrono::duration<double, std::nano>(t1 - t0).count();

	// --- Stage 2: streaming FFT in ArithScalar ---
	auto window = hanning_window<ArithScalar>(params.fft_size);
	std::span<const ArithScalar> window_span(window.data(), params.fft_size);
	t0 = chrono::high_resolution_clock::now();
	spectrum::RealtimeSpectrum<ArithScalar, ArithScalar, ArithScalar, ArithScalar>
		live(params.fft_size, params.hop_size, window_span);
	std::span<const ArithScalar> eq_span(equalized.data(), equalized.size());
	(void)live.push(eq_span);
	t1 = chrono::high_resolution_clock::now();
	result.timings.spectrum =
		chrono::duration<double, std::nano>(t1 - t0).count();

	if (live.total_ffts() == 0) {
		// Stream too short to even fill the FFT once — bail with NaN.
		return result;
	}

	// --- Stage 3: trace averaging across the produced FFTs ---
	//
	// RealtimeSpectrum only retains the most recent magnitude buffer; to
	// average across sweeps we re-run with smaller pushes and call
	// accept_sweep on each. Cheap because the FFT cost has already been
	// paid above; what we're timing here is the per-bin smoothing.
	//
	// The "input" to the averager is double-typed (RealtimeSpectrum's
	// magnitude_db is always double); the averaged trace stays in double.
	t0 = chrono::high_resolution_clock::now();
	spectrum::TraceAverager<double> averager(
		params.fft_size,
		spectrum::TraceAverager<double>::Mode::Exponential,
		1.0 / static_cast<double>(params.avg_n));
	{
		// Re-run streaming in chunks of hop_size to feed the averager.
		spectrum::RealtimeSpectrum<ArithScalar, ArithScalar, ArithScalar, ArithScalar>
			live2(params.fft_size, params.hop_size, window_span);
		for (std::size_t off = 0; off < equalized.size(); off += params.hop_size) {
			const std::size_t take =
				std::min(params.hop_size, equalized.size() - off);
			std::span<const ArithScalar> chunk(equalized.data() + off, take);
			const std::size_t produced = live2.push(chunk);
			if (produced > 0)
				averager.accept_sweep(live2.latest_magnitude_db());
		}
	}
	auto avg_trace = averager.current_trace();
	t1 = chrono::high_resolution_clock::now();
	result.timings.trace_avg =
		chrono::duration<double, std::nano>(t1 - t0).count();

	// --- Stage 4: waterfall buffer (pure storage; cheap) ---
	t0 = chrono::high_resolution_clock::now();
	spectrum::WaterfallBuffer<double> waterfall(
		params.fft_size, params.waterfall_frames);
	{
		spectrum::RealtimeSpectrum<ArithScalar, ArithScalar, ArithScalar, ArithScalar>
			live3(params.fft_size, params.hop_size, window_span);
		for (std::size_t off = 0; off < equalized.size(); off += params.hop_size) {
			const std::size_t take =
				std::min(params.hop_size, equalized.size() - off);
			std::span<const ArithScalar> chunk(equalized.data() + off, take);
			const std::size_t produced = live3.push(chunk);
			if (produced > 0)
				waterfall.push_frame(live3.latest_magnitude_db());
		}
	}
	t1 = chrono::high_resolution_clock::now();
	result.timings.waterfall =
		chrono::duration<double, std::nano>(t1 - t0).count();

	// --- Stage 5: markers (peak-find for the four planted features) ---
	//
	// Search the lower half-band only (positive frequencies up to
	// Nyquist). FFT bin spacing is fs / N. Top-N is generous so the
	// 4 planted features all have a candidate in the returned list.
	t0 = chrono::high_resolution_clock::now();
	const double bin_step_hz =
		params.sample_rate_hz / static_cast<double>(params.fft_size);
	const std::size_t half_bins = params.fft_size / 2 + 1;
	std::vector<double> half_trace(avg_trace.begin(),
	                                avg_trace.begin() + half_bins);
	std::span<const double> half_span(half_trace.data(), half_trace.size());
	auto peaks = spectrum::find_peaks(half_span, bin_step_hz, 8, 5);
	t1 = chrono::high_resolution_clock::now();
	result.timings.markers =
		chrono::duration<double, std::nano>(t1 - t0).count();

	// Stash the half-band averaged trace for SNR + CSV.
	result.trace_db = std::move(half_trace);
	result.trace_freq_step_hz = bin_step_hz;

	// Tolerance for "this peak is the planted tone we're looking for":
	// 5 bins. Anything farther means the tone wasn't recovered.
	const double tone_tol_hz = 5.0 * bin_step_hz;
	auto m1 = nearest_marker(peaks, params.tone1_hz,    tone_tol_hz);
	auto m2 = nearest_marker(peaks, params.tone2_hz,    tone_tol_hz);
	auto m3 = nearest_marker(peaks, params.tone3_hz,    tone_tol_hz);
	auto ms = nearest_marker(peaks, params.spurious_hz, tone_tol_hz);
	result.tone1_freq_hz   = m1.frequency_hz;
	result.tone1_amp_db    = m1.amplitude;
	result.tone2_freq_hz   = m2.frequency_hz;
	result.tone2_amp_db    = m2.amplitude;
	result.tone3_freq_hz   = m3.frequency_hz;
	result.tone3_amp_db    = m3.amplitude;
	result.spurious_amp_db = ms.amplitude;

	const std::size_t guard_bins = 5;
	const double trace_max_freq = bin_step_hz * static_cast<double>(half_bins - 1);
	const std::size_t bin_lo =
		static_cast<std::size_t>(std::ceil(2.0e3 / bin_step_hz));   // skip DC
	const std::size_t bin_hi =
		static_cast<std::size_t>(std::floor(
			std::min(300.0e3, trace_max_freq) / bin_step_hz));
	result.noise_floor_db = noise_floor_db(
		std::span<const double>(result.trace_db.data(), result.trace_db.size()),
		bin_step_hz,
		{params.tone1_hz, params.tone2_hz, params.tone3_hz, params.spurious_hz},
		guard_bins, bin_lo, bin_hi);
	// "Detected" iff a peak landed within the tolerance window AND it
	// stands at least 10 dB above the surrounding noise floor. The
	// 10 dB margin is the conventional analyzer "discrete signal vs.
	// noise" threshold.
	result.spurious_detected =
		std::isfinite(result.spurious_amp_db) &&
		std::isfinite(result.noise_floor_db) &&
		(result.spurious_amp_db - result.noise_floor_db) > 10.0;

	result.timings.total = result.timings.equalizer
	                     + result.timings.spectrum
	                     + result.timings.trace_avg
	                     + result.timings.waterfall
	                     + result.timings.markers;
	return result;
}

// ============================================================================
// Swept-tuned pipeline: input -> EqualizerFilter -> mixer * SweptLO ->
//                       RBWFilter -> square-law detector -> VBWFilter ->
//                       trace[bin] memory indexed by current LO frequency
//
// All numerically-active stages run in ArithScalar. The detector is
// applied per-LO-bin: each LO bin collects samples from a small dwell
// window, then `detect_rms` reduces them to a single trace value.
// ============================================================================

template <class ArithScalar>
ConfigResult run_swept_path(std::span<const double> input_double,
                             const std::string& plan_name,
                             const std::string& arith_tag,
                             std::size_t arith_bytes,
                             const spectrum::CalibrationProfile& profile) {
	ConfigResult result;
	result.plan_name              = plan_name;
	result.path                   = "swept";
	result.arith_type             = arith_tag;
	result.arith_bytes_per_sample = arith_bytes;

	// --- Stage 1: equalizer in ArithScalar ---
	auto t0 = chrono::high_resolution_clock::now();
	spectrum::FrontEndCorrector<ArithScalar> eq(
		profile, params.eq_taps, params.sample_rate_hz);
	std::vector<ArithScalar> equalized(input_double.size());
	for (std::size_t n = 0; n < input_double.size(); ++n) {
		const ArithScalar x = static_cast<ArithScalar>(input_double[n]);
		equalized[n] = eq.process(x);
	}
	auto t1 = chrono::high_resolution_clock::now();
	result.timings.equalizer =
		chrono::duration<double, std::nano>(t1 - t0).count();

	// --- Stage 2: mixer * SweptLO -> RBW filter -> square-law detector ---
	//
	// The LO is centered on the "instantaneous frequency of interest";
	// the mixer downconverts the input around that frequency to DC; the
	// RBW (centered slightly off DC to avoid 1/f issues from the
	// quasi-DC product, but here we use a near-DC sync-tuned cascade
	// for simplicity) selects a narrow window; the square-law detector
	// produces the band power. We bin the per-sample power by the LO
	// frequency and accumulate.
	//
	// To keep the RBW stable, the filter is centered at a fixed IF
	// (`if_hz`) and the LO is offset so the input frequency of interest
	// shifts to that IF. This is the classic "swept LO + fixed IF"
	// architecture. IF chosen ~1.6x RBW so the RBW skirt sits
	// comfortably above DC (avoiding 1/f issues from the quasi-DC
	// product) while staying smaller than swept_f_start_hz so the LO
	// frequency LO_freq = f_real - IF stays positive across the sweep.
	const double if_hz = 8.0e3;
	t0 = chrono::high_resolution_clock::now();
	spectrum::SweptLO<ArithScalar> lo(
		params.swept_f_start_hz - if_hz,
		params.swept_f_stop_hz  - if_hz,
		params.swept_duration_s,
		params.sample_rate_hz);
	spectrum::RBWFilter<ArithScalar> rbw(
		if_hz, params.rbw_hz, params.sample_rate_hz, params.rbw_order);

	// Bin layout: the swept-trace memory is `swept_trace_bins` bins
	// uniformly spanning [swept_f_start_hz, swept_f_stop_hz]. The
	// LO-driven mixer hits each bin proportional to the dwell time at
	// that LO frequency. For a linear sweep over N samples, the dwell
	// per bin is roughly N / swept_trace_bins samples.
	std::vector<double> bin_power_sum(params.swept_trace_bins, 0.0);
	std::vector<std::size_t> bin_count(params.swept_trace_bins, 0);

	const double swept_span =
		params.swept_f_stop_hz - params.swept_f_start_hz;
	const double inv_bin_width =
		static_cast<double>(params.swept_trace_bins) / swept_span;

	for (std::size_t n = 0; n < equalized.size(); ++n) {
		// Mixer: x * cos(LO).
		auto [c, s] = lo.process();
		(void)s;
		const ArithScalar mixed = equalized[n] * c;
		const ArithScalar bp    = rbw.process(mixed);
		const double      bp_d  = static_cast<double>(bp);
		const double      pwr   = bp_d * bp_d;

		// LO frequency at this sample is the LO's instantaneous freq
		// PLUS the IF offset (the "real" frequency the analyzer is
		// looking at). Skip if outside [f_start, f_stop].
		const double f_real = lo.current_frequency_hz() + if_hz;
		if (f_real < params.swept_f_start_hz || f_real > params.swept_f_stop_hz)
			continue;
		const double bin_d = (f_real - params.swept_f_start_hz) * inv_bin_width;
		const std::size_t bin = std::min<std::size_t>(
			static_cast<std::size_t>(std::floor(bin_d)),
			params.swept_trace_bins - 1);
		bin_power_sum[bin] += pwr;
		bin_count[bin]     += 1;
	}
	t1 = chrono::high_resolution_clock::now();
	result.timings.spectrum =
		chrono::duration<double, std::nano>(t1 - t0).count();

	// --- Stage 3: VBW smoothing on the per-bin RMS trace ---
	//
	// Convert bin-sum-of-squares to per-bin RMS amplitude (the
	// square-law detector's output magnitude), then run a single-pole
	// VBW LPF across the bins. Fs for the VBW step is the bin rate
	// (one "sample" per bin).
	t0 = chrono::high_resolution_clock::now();
	std::vector<ArithScalar> trace_amp(params.swept_trace_bins,
	                                     ArithScalar{});
	for (std::size_t i = 0; i < params.swept_trace_bins; ++i) {
		if (bin_count[i] == 0) continue;
		const double mean_pwr =
			bin_power_sum[i] / static_cast<double>(bin_count[i]);
		trace_amp[i] = static_cast<ArithScalar>(std::sqrt(mean_pwr));
	}
	// VBW across the swept trace. Treat the bin index as the time axis;
	// pick a cutoff that's a small fraction of the bin rate so the VBW
	// genuinely smooths. A Butterworth-style 1/8 cutoff is fine.
	{
		const double bin_rate_hz = static_cast<double>(params.swept_trace_bins);
		spectrum::VBWFilter<ArithScalar> vbw(bin_rate_hz / 16.0, bin_rate_hz);
		std::span<ArithScalar> trace_span(trace_amp.data(), trace_amp.size());
		vbw.process_block(trace_span);
	}
	t1 = chrono::high_resolution_clock::now();
	result.timings.vbw =
		chrono::duration<double, std::nano>(t1 - t0).count();

	// Convert trace to dB for marker / SNR scoring.
	std::vector<double> trace_db(params.swept_trace_bins);
	for (std::size_t i = 0; i < params.swept_trace_bins; ++i) {
		const double a = static_cast<double>(trace_amp[i]);
		const double a2 = std::max(a * a, 1e-20);
		trace_db[i] = 10.0 * std::log10(a2);
	}
	const double bin_step_hz = swept_span / static_cast<double>(params.swept_trace_bins);

	// --- Stage 4: markers ---
	t0 = chrono::high_resolution_clock::now();
	std::span<const double> trace_span(trace_db.data(), trace_db.size());
	auto peaks = spectrum::find_peaks(trace_span, bin_step_hz, 8, 2);
	t1 = chrono::high_resolution_clock::now();
	result.timings.markers =
		chrono::duration<double, std::nano>(t1 - t0).count();

	// Markers report frequency relative to bin 0 (= swept_f_start_hz),
	// not absolute. Re-bias by f_start so the peak frequencies match
	// the planted-tone frequencies in absolute Hz.
	auto offset_marker = [&](spectrum::Marker m) {
		m.frequency_hz += params.swept_f_start_hz;
		return m;
	};
	std::vector<spectrum::Marker> abs_peaks;
	abs_peaks.reserve(peaks.size());
	for (auto& m : peaks) abs_peaks.push_back(offset_marker(m));

	// Tolerance for tone identification: a generous 3 bins on the
	// swept trace, since the swept architecture's resolution is
	// dominated by the (RBW) settling time and dwell-per-bin trade,
	// not by the trace bin width. With 64 bins over 230 kHz, 3 bins
	// is ~10.8 kHz — wider than RBW (5 kHz) but narrower than the
	// minimum tone separation (50 kHz).
	const double tone_tol_hz = 3.0 * bin_step_hz;
	auto m1 = nearest_marker(abs_peaks, params.tone1_hz,    tone_tol_hz);
	auto m2 = nearest_marker(abs_peaks, params.tone2_hz,    tone_tol_hz);
	auto m3 = nearest_marker(abs_peaks, params.tone3_hz,    tone_tol_hz);
	auto ms = nearest_marker(abs_peaks, params.spurious_hz, tone_tol_hz);
	result.tone1_freq_hz   = m1.frequency_hz;
	result.tone1_amp_db    = m1.amplitude;
	result.tone2_freq_hz   = m2.frequency_hz;
	result.tone2_amp_db    = m2.amplitude;
	result.tone3_freq_hz   = m3.frequency_hz;
	result.tone3_amp_db    = m3.amplitude;
	result.spurious_amp_db = ms.amplitude;

	// Noise floor: average dB across bins not near any planted feature.
	// Trace's bin-0 frequency is f_start, not 0, so use the relative
	// span when computing distances.
	auto noise_floor_swept = [&]() -> double {
		const std::size_t guard_bins = 2;
		double sum = 0.0;
		std::size_t n = 0;
		for (std::size_t i = 0; i < trace_db.size(); ++i) {
			const double f = params.swept_f_start_hz +
			                  static_cast<double>(i) * bin_step_hz;
			bool near = false;
			for (double ff : {params.tone1_hz, params.tone2_hz,
			                   params.tone3_hz, params.spurious_hz}) {
				if (std::abs(f - ff) / bin_step_hz <
				    static_cast<double>(guard_bins)) {
					near = true; break;
				}
			}
			if (near) continue;
			sum += trace_db[i]; ++n;
		}
		return n > 0 ? sum / static_cast<double>(n)
		             : std::numeric_limits<double>::quiet_NaN();
	};
	result.noise_floor_db = noise_floor_swept();
	result.spurious_detected =
		std::isfinite(result.spurious_amp_db) &&
		std::isfinite(result.noise_floor_db) &&
		(result.spurious_amp_db - result.noise_floor_db) > 10.0;

	result.trace_db           = std::move(trace_db);
	result.trace_freq_step_hz = bin_step_hz;
	result.timings.total = result.timings.equalizer
	                     + result.timings.spectrum
	                     + result.timings.vbw
	                     + result.timings.markers;
	return result;
}

// ============================================================================
// Output trace SNR vs the reference
// ============================================================================

double snr_db_against_reference(const ConfigResult& test,
                                 const ConfigResult& ref) {
	if (ref.trace_db.size() != test.trace_db.size() || ref.trace_db.empty())
		return std::numeric_limits<double>::quiet_NaN();
	double sig_pow = 0.0, err_pow = 0.0;
	for (std::size_t i = 0; i < ref.trace_db.size(); ++i) {
		const double s = ref.trace_db[i];
		const double e = ref.trace_db[i] - test.trace_db[i];
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
	out << "pipeline,plan_name,path,arith_type,arith_bytes_per_sample,"
	       "bin_index,frequency_hz,amplitude_db,"
	       "tone1_freq,tone1_amp,tone2_freq,tone2_amp,tone3_freq,tone3_amp,"
	       "spurious_amp,spurious_detected,noise_floor_db,output_snr_db\n";
	for (const auto& r : results) {
		for (std::size_t i = 0; i < r.trace_db.size(); ++i) {
			const double freq = (r.path == "swept")
				? params.swept_f_start_hz +
				  static_cast<double>(i) * r.trace_freq_step_hz
				: static_cast<double>(i) * r.trace_freq_step_hz;
			out << "spectrum_analyzer_demo,"
			    << r.plan_name        << ','
			    << r.path             << ','
			    << r.arith_type       << ','
			    << r.arith_bytes_per_sample << ','
			    << i                  << ','
			    << freq               << ','
			    << r.trace_db[i]      << ','
			    << r.tone1_freq_hz    << ','
			    << r.tone1_amp_db     << ','
			    << r.tone2_freq_hz    << ','
			    << r.tone2_amp_db     << ','
			    << r.tone3_freq_hz    << ','
			    << r.tone3_amp_db     << ','
			    << r.spurious_amp_db << ','
			    << (r.spurious_detected ? 1 : 0) << ','
			    << r.noise_floor_db   << ','
			    << r.output_snr_db
			    << '\n';
		}
	}
}

// ============================================================================
// Console summary
// ============================================================================

void print_summary_header() {
	std::cout << "\n=== Mixed-precision spectrum-analyzer sweep ===\n";
	std::cout << std::left  << std::setw(28) << "plan (path / arith)"
	          << std::right << std::setw(8)  << "B/samp"
	          << std::right << std::setw(11) << "tone1(kHz)"
	          << std::right << std::setw(11) << "tone1(dB)"
	          << std::right << std::setw(11) << "tone3(dB)"
	          << std::right << std::setw(10) << "spur?"
	          << std::right << std::setw(11) << "floor(dB)"
	          << std::right << std::setw(10) << "SNR(dB)"
	          << "\n";
	std::cout << std::string(28 + 8 + 11 + 11 + 11 + 10 + 11 + 10, '-')
	          << "\n";
}

void print_summary_row(const ConfigResult& r) {
	std::string label = r.plan_name + " (" + r.path + "/" + r.arith_type + ")";
	if (label.size() > 27) label = label.substr(0, 27);
	auto print_or_nan = [](double v, int prec, double scale = 1.0) {
		if (std::isnan(v)) std::cout << "NaN";
		else std::cout << std::fixed << std::setprecision(prec) << (v * scale);
	};
	std::cout << std::left  << std::setw(28) << label
	          << std::right << std::setw(8)  << r.arith_bytes_per_sample;
	std::cout << std::right << std::setw(11);
	print_or_nan(r.tone1_freq_hz, 3, 1.0 / 1e3);
	std::cout << std::right << std::setw(11);
	print_or_nan(r.tone1_amp_db, 2);
	std::cout << std::right << std::setw(11);
	print_or_nan(r.tone3_amp_db, 2);
	std::cout << std::right << std::setw(10);
	std::cout << (r.spurious_detected ? "PASS" : "fail");
	std::cout << std::right << std::setw(11);
	print_or_nan(r.noise_floor_db, 2);
	std::cout << std::right << std::setw(10);
	print_or_nan(r.output_snr_db, 2);
	std::cout << "\n";
}

// ============================================================================
// Per-stage timing report against the reference plan
// ============================================================================

void print_timing_report(const ConfigResult& reference,
                          std::size_t input_samples) {
	std::cout << "\n=== Per-stage timing (" << reference.path
	          << " path, reference plan) ===\n";
	const double samples = static_cast<double>(input_samples);
	auto print = [&](const char* name, double ns_total) {
		if (ns_total <= 0.0) return;
		const double ns_per_sample = ns_total / samples;
		std::cout << "  " << std::left << std::setw(18) << name
		          << std::right << std::setw(12) << std::fixed
		          << std::setprecision(2) << ns_total << " ns total"
		          << std::right << std::setw(12) << std::setprecision(3)
		          << ns_per_sample << " ns/sample"
		          << "\n";
	};
	print("equalizer",  reference.timings.equalizer);
	print("spectrum",   reference.timings.spectrum);
	print("trace_avg",  reference.timings.trace_avg);
	print("vbw",        reference.timings.vbw);
	print("waterfall",  reference.timings.waterfall);
	print("markers",    reference.timings.markers);
	print("TOTAL",      reference.timings.total);

	const double total_per_sample = reference.timings.total / samples;
	const double target_ns_per_sample = 0.1;   // 10 GSPS budget
	std::cout << "\n  10 GSPS budget: 0.100 ns/sample\n";
	if (total_per_sample <= target_ns_per_sample) {
		std::cout << "  10 GSPS: ACHIEVED (" << std::fixed
		          << std::setprecision(3) << total_per_sample
		          << " ns/sample)\n";
	} else {
		const double speedup_needed = total_per_sample / target_ns_per_sample;
		std::cout << "  10 GSPS: NOT achievable on general-purpose CPU "
		             "(would need " << std::fixed << std::setprecision(1)
		          << speedup_needed << "x speedup)\n";
		std::cout << "  Real 10 GSPS analyzers use ASIC pipelines — this is "
		             "an informational comparison.\n";
	}
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) try {
	std::string csv_path = "spectrum_analyzer_demo.csv";
	for (int i = 1; i < argc; ++i) {
		std::string a = argv[i];
		if (a.rfind("--csv=", 0) == 0) csv_path = a.substr(6);
		else if (a == "-h" || a == "--help") {
			std::cout << "Usage: " << argv[0] << " [--csv=path]\n";
			return 0;
		}
	}

	std::cout << "spectrum_analyzer_demo: mixed-precision spectrum-analyzer sweep\n"
	          << "  signal:    3-tone (50/100/150 kHz at 0/-30/-60 dBc), "
	             "spurious at " << params.spurious_hz / 1e3 << " kHz "
	          << params.spurious_dbc << " dBc, AWGN "
	          << params.noise_rms << "\n"
	          << "  fs:        " << params.sample_rate_hz / 1e6
	          << " MHz, " << params.num_samples << " samples\n"
	          << "  FFT path:  size=" << params.fft_size
	          << ", hop=" << params.hop_size
	          << ", exp avg alpha=1/" << params.avg_n
	          << ", waterfall=" << params.waterfall_frames << " frames\n"
	          << "  Swept:     ["
	          << params.swept_f_start_hz / 1e3 << "k.."
	          << params.swept_f_stop_hz  / 1e3 << "k] Hz, "
	          << params.swept_duration_s * 1e3 << " ms, "
	          << "RBW=" << params.rbw_hz << " Hz (order "
	          << params.rbw_order << "), VBW=" << params.vbw_hz << " Hz, "
	          << params.swept_trace_bins << " bins\n"
	          << "  EQ:        " << params.eq_taps
	          << "-tap FIR, mild rolloff to -3 dB at 1 MHz\n";

	const auto profile = make_test_profile();
	const auto input   = simulate_input();
	std::span<const double> in_span(input.data(), input.size());

	// =========================================================================
	// Precision plans (Option A: both paths x 4 ArithScalar plans)
	//
	//   reference     - double everywhere (SNR baseline)
	//   arith_float   - float drives EQ + FFT/RBW/VBW
	//   arith_posit32 - posit<32,2> across the precision-active stages
	//   arith_posit16 - posit<16,2> across the precision-active stages
	// =========================================================================
	std::array<ConfigResult, 8> results{{
		// FFT path
		run_fft_path<double>(in_span, "reference",
			"double",      sizeof(double), profile),
		run_fft_path<float >(in_span, "arith_float",
			"float",       sizeof(float),  profile),
		run_fft_path<p32   >(in_span, "arith_posit32",
			"posit<32,2>", sizeof(p32),    profile),
		run_fft_path<p16   >(in_span, "arith_posit16",
			"posit<16,2>", sizeof(p16),    profile),
		// Swept-tuned path
		run_swept_path<double>(in_span, "reference",
			"double",      sizeof(double), profile),
		run_swept_path<float >(in_span, "arith_float",
			"float",       sizeof(float),  profile),
		run_swept_path<p32   >(in_span, "arith_posit32",
			"posit<32,2>", sizeof(p32),    profile),
		run_swept_path<p16   >(in_span, "arith_posit16",
			"posit<16,2>", sizeof(p16),    profile),
	}};

	// SNR vs the reference of each path. results[0] = FFT reference,
	// results[4] = swept reference.
	const ConfigResult& fft_ref   = results[0];
	const ConfigResult& swept_ref = results[4];
	for (auto& r : results) {
		const ConfigResult& ref = (r.path == "fft") ? fft_ref : swept_ref;
		r.output_snr_db = snr_db_against_reference(r, ref);
	}

	print_summary_header();
	for (const auto& r : results) print_summary_row(r);

	print_timing_report(fft_ref,   params.num_samples);
	print_timing_report(swept_ref, params.num_samples);

	write_csv(csv_path, results);
	std::cout << "\nCSV written: " << csv_path << "\n";

	return 0;
} catch (const std::exception& ex) {
	std::cerr << "FATAL: " << ex.what() << "\n";
	return 1;
}
