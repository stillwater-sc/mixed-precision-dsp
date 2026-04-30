// scope_demo.cpp: end-to-end digital-oscilloscope mixed-precision sweep.
//
// Synthesizes a realistic test waveform — 50 MHz square wave with a 5 ns
// narrow positive glitch buried in it plus low-level AWGN — and runs it
// through the full scope DSP pipeline:
//
//   simulate_adc(N_bits, sample_rate)
//        |
//        v
//   EqualizerFilter<EqCoeff,EqState,EqSample>   (instrument/calibration.hpp)
//        |   <-- the precision-sensitive stage
//        v
//   EdgeTrigger + AutoTriggerWrapper            (instrument/trigger.hpp)
//        |
//        v
//   TriggerRingBuffer (pre/post capture)        (instrument/ring_buffer.hpp)
//        |
//        v
//   PeakDetectDecimator                         (instrument/peak_detect.hpp)
//        |
//        v
//   render_envelope (-> N pixels)               (instrument/display_envelope.hpp)
//        |
//        v
//   measurements (rise time, RMS, ...)          (instrument/measurements.hpp)
//        |
//        v
//   scope_demo.csv + console summary
//
// Mixed-precision design (the WHOLE point of this demo):
//
//   The trigger -> ring -> peak-detect -> envelope chain is comparison-
//   only and copy-only — every operation preserves bit-exact ordering
//   regardless of the storage type. Storage precision is therefore a
//   memory-bandwidth knob: pick the narrowest type that doesn't lose
//   information from the ADC.
//
//   The EqualizerFilter is the one place where the streaming path does
//   actual arithmetic (FIR multiply-accumulate). That's where precision
//   matters: narrowing the equalizer's coefficient/state/sample types
//   trades streaming-arithmetic accuracy for energy and bits.
//
//   Measurements always run in double internally regardless of input
//   type — they're the analytical reporting layer, not part of the
//   streaming arithmetic.
//
// A "precision plan" is the per-stage tuple of types — NOT a single
// "uniform_T" choice. Each row of the sweep table picks each stage's
// precision independently. SNR is measured against the all-double
// reference plan, so a row with a narrow equalizer shows real SNR
// degradation while one with a narrow storage type but a high-precision
// equalizer shows none.
//
// Per-stage timing instrumentation produces a 10 GSPS comparison
// (informational): real 10 GSPS scopes use ASIC pipelines, not general-
// purpose CPUs. The number reported here is for understanding the gap.
//
// Capstone for the Digital Oscilloscope Demonstrator epic (#133).
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/instrument/calibration.hpp>
#include <sw/dsp/instrument/trigger.hpp>
#include <sw/dsp/instrument/ring_buffer.hpp>
#include <sw/dsp/instrument/peak_detect.hpp>
#include <sw/dsp/instrument/display_envelope.hpp>
#include <sw/dsp/instrument/measurements.hpp>

#include <sw/universal/number/posit/posit.hpp>
#include <sw/universal/number/cfloat/cfloat.hpp>
#include <sw/universal/number/fixpnt/fixpnt.hpp>

#include <mtl/vec/dense_vector.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <span>
#include <string>
#include <vector>

using namespace sw::dsp::instrument;
namespace chrono = std::chrono;

// ============================================================================
// Type aliases used in the precision plans
// ============================================================================

using p32  = sw::universal::posit<32, 2>;
using p16  = sw::universal::posit<16, 2>;
// Q4.12 (16 bits): 4 integer bits give the equalizer some headroom for
// transient overshoots, 12 fractional bits match the ADC's 12-bit output.
// "ADC-native" storage — narrowing storage to this size cuts ring-buffer
// memory bandwidth 4x vs double without dropping any ADC information.
using fx16_storage = sw::universal::fixpnt<16, 12>;

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

	// Equalizer (calibration / front-end correction).
	std::size_t eq_taps          = 31;      // FIR length
};
inline PipelineParams params;

// ============================================================================
// Forward-profile FIR design (analog front-end model)
//
// Mirrors EqualizerFilter::design_taps() but WITHOUT the inversion step:
// where the equalizer samples 1 / |H(f)| (so the streaming filter cancels
// the front-end response), this function samples |H(f)| directly so the
// streaming filter REPRODUCES it. The two helpers are sibling FIR designs
// from the same CalibrationProfile object — running the source signal
// through this filter gives a profile-distorted signal, which is then what
// the equalizer un-distorts on the digital path.
//
// Algorithm (frequency-sampling design with linear-phase shift + Hamming):
//   1. Sample H(f) at N uniformly-spaced frequency bins, applying both the
//      profile's magnitude (in dB → linear) AND its phase.
//   2. Force DC and Nyquist bins to be real (a real impulse response
//      requires this; non-zero phase at those bins is rare but possible).
//   3. Mirror by conjugate symmetry to populate the upper half of the
//      DFT.
//   4. Inverse DFT with a linear-phase delay of (N-1)/2 samples to center
//      the impulse response.
//   5. Hamming window to suppress sidelobes.
//
// No max-gain clamp here (the forward profile attenuates rather than
// boosts). The output coefficients are double — the analog-front-end model
// is conceptual reference; precision narrowing on this stage isn't part of
// the sweep (the sweep narrows the *equalizer's* precision, which is the
// stage that exists in real digital scope hardware).
// ============================================================================

mtl::vec::dense_vector<double>
design_forward_fir(const CalibrationProfile& profile,
                   std::size_t num_taps,
                   double sample_rate_hz) {
	if (num_taps < 3)
		throw std::invalid_argument("design_forward_fir: num_taps must be >= 3");
	if (!(sample_rate_hz > 0.0))
		throw std::invalid_argument("design_forward_fir: sample_rate_hz must be > 0");

	const std::size_t N    = num_taps;
	const double      pi   = std::numbers::pi_v<double>;

	// 1+2: sample H(f) on the lower half-spectrum.
	std::vector<std::complex<double>> H_d(N);
	for (std::size_t k = 0; k <= N / 2; ++k) {
		const double f       = static_cast<double>(k) * sample_rate_hz / N;
		const double gain_dB = profile.gain_dB(f);
		const double phase   = profile.phase_rad(f);
		const double mag     = std::pow(10.0, gain_dB / 20.0);
		H_d[k] = std::complex<double>(mag * std::cos(phase),
		                               mag * std::sin(phase));
	}
	auto force_real = [](const std::complex<double>& z) {
		return std::complex<double>(
			std::copysign(std::abs(z), z.real()), 0.0);
	};
	H_d[0] = force_real(H_d[0]);
	if (N % 2 == 0)
		H_d[N / 2] = force_real(H_d[N / 2]);
	using std::conj;
	for (std::size_t k = 1; k < (N + 1) / 2; ++k)
		H_d[N - k] = conj(H_d[k]);

	// 3+4: inverse DFT, centered.
	std::vector<double> h_centered(N);
	const double         delay = static_cast<double>(N - 1) / 2.0;
	for (std::size_t n = 0; n < N; ++n) {
		std::complex<double> acc{0.0, 0.0};
		for (std::size_t k = 0; k < N; ++k) {
			const double angle = 2.0 * pi * static_cast<double>(k) *
			                     (static_cast<double>(n) - delay) / N;
			acc += H_d[k] * std::complex<double>(std::cos(angle),
			                                     std::sin(angle));
		}
		h_centered[n] = acc.real() / static_cast<double>(N);
	}

	// 5: Hamming window.
	mtl::vec::dense_vector<double> taps(N);
	for (std::size_t n = 0; n < N; ++n) {
		const double w = 0.54 - 0.46 * std::cos(
			2.0 * pi * static_cast<double>(n) / static_cast<double>(N - 1));
		taps[n] = h_centered[n] * w;
	}
	return taps;
}

// ============================================================================
// ADC simulation: clean source → analog-front-end distortion → quantization
//
// This function models the entire signal-acquisition path that precedes the
// digital pipeline:
//
//   clean source  --(forward calibration FIR)-->  distorted analog signal
//                                                        |
//                                                        v  (12-bit quantize)
//                                                  ADC samples
//                                                        |
//                                                        v
//                                                EqualizerFilter
//                                                (un-distorts; inverse of
//                                                 the same profile)
//
// The clean source is what you'd see at the SOURCE — an oscilloscope probe
// touching an ideal signal generator. By the time the signal reaches the
// ADC, the analog front end (probe + amplifier + sample-and-hold network)
// has imposed a non-trivial frequency response on it: high-frequency
// attenuation, group-delay variation, possibly a small DC offset. The
// equalizer's job in the digital domain is to UN-DISTORT this — to recover
// the source signal from the front-end-distorted samples.
//
// Without pre-distortion the demo would be applying the equalizer to a
// signal that doesn't need correction, and the equalizer's mixed-precision
// stress test would be artificially mild. With pre-distortion the
// equalizer is doing substantial arithmetic work — boosting the attenuated
// high-frequency content back to its source amplitude — and the
// precision-sensitivity comparison becomes meaningful.
//
// The forward FIR's group delay (≈(N-1)/2 = 15 samples at N=31) is mirrored
// by the equalizer's matching group delay, so the post-equalizer signal is
// time-aligned with the SOURCE delayed by the COMBINED filter delay (~30
// samples). The SNR-vs-source comparator below accounts for this.
//
// AWGN is added AFTER the forward FIR so the noise floor sits on top of
// the distorted signal — same as a real scope where the front-end's
// thermal noise is added at the ADC, not before the analog stages.
// ============================================================================

std::vector<double> simulate_clean_source(unsigned seed = 0xACDC) {
	// Builds the clean reference source — what an ideal signal generator
	// would put out before any analog front-end distortion. No AWGN, no
	// quantization, no profile coloring. The post-equalizer streaming
	// output should approximate this signal (delayed by the forward+inverse
	// FIR group delay).
	std::vector<double> source(params.num_samples);
	const double dt          = 1.0 / params.sample_rate_hz;
	const double glitch_t0   = params.glitch_offset_s;
	const double glitch_t1   = glitch_t0 + params.glitch_width_s;
	const std::size_t half_period_samples =
		static_cast<std::size_t>(std::round(
			0.5 * params.sample_rate_hz / params.signal_freq_hz));
	const std::size_t cycle_samples = 2 * half_period_samples;
	(void)seed;
	for (std::size_t n = 0; n < params.num_samples; ++n) {
		const double t = static_cast<double>(n) * dt;
		const std::size_t phase_n = n % cycle_samples;
		const double sq = (phase_n < half_period_samples)
		                   ? params.signal_amp : -params.signal_amp;
		source[n] = (t >= glitch_t0 && t < glitch_t1)
		             ? params.glitch_peak : sq;
	}
	return source;
}

std::vector<double> simulate_adc(const std::vector<double>& source,
                                  const CalibrationProfile& profile,
                                  unsigned seed = 0xACDC) {
	// Output length tracks the source argument, not params.num_samples.
	// Keeps the helper self-consistent with its input (and lets callers
	// pass a different-length vector for unit testing or sub-segment
	// experiments).
	std::vector<double> samples(source.size());
	std::mt19937 rng(seed);
	std::normal_distribution<double> noise(0.0, params.noise_rms);

	const double half_levels = std::ldexp(1.0, params.adc_bits - 1);
	const double q_step      = 1.0 / half_levels;
	const double code_max    = half_levels - 1.0;
	const double code_min    = -half_levels;

	// Design the forward analog-front-end FIR from the SAME profile the
	// equalizer inverts. This filter REPLACES the analog stages of a real
	// scope (probe, amplifier, sample-and-hold) with a digital model that
	// applies the profile's frequency response to the clean source signal.
	auto fwd_taps = design_forward_fir(profile, params.eq_taps,
	                                    params.sample_rate_hz);
	// Bare FIR convolution in double — this is the conceptual analog
	// front end, not part of the precision sweep. The FIR head is
	// zero-padded (k <= n bound) rather than indexing source[0] for
	// n < k, which would convolve the first source sample N times with
	// itself and create a synthetic prehistory artifact.
	const std::size_t N = fwd_taps.size();
	for (std::size_t n = 0; n < source.size(); ++n) {
		double y = 0.0;
		for (std::size_t k = 0; k < N && k <= n; ++k) {
			y += fwd_taps[k] * source[n - k];
		}
		// Add thermal noise AFTER the front-end distortion (matches real
		// scope topology: front-end thermal noise is added at the ADC
		// input, not before the analog amp).
		const double noisy = y + noise(rng);

		// Quantize at the ADC.
		double code = std::floor(noisy / q_step);
		code = std::clamp(code, code_min, code_max);
		samples[n] = code * q_step;
	}
	return samples;
}

// ============================================================================
// Synthetic analog-front-end calibration profile.
//
// Models a realistic high-bandwidth scope front end: very shallow
// rolloff in-band (the signal of interest at 50 MHz is barely
// touched, -0.5 dB), a -3 dB corner near 100 MHz, then progressively
// steeper attenuation toward Nyquist:
//
//     freq           gain
//      0 Hz          0 dB     (DC: flat)
//     50 MHz       -0.5 dB    (in-band: slight rolloff)
//    100 MHz        -3 dB     (corner of the bandwidth-limited path)
//    250 MHz        -6 dB     (well above corner; in-band edge attenuation)
//    500 MHz       -10 dB     (Nyquist: deep stop-band)
//
// Phase walks roughly linearly with frequency, modelling the front
// end's group delay. These numbers are representative of a real
// high-speed front end where the analog stages have a 2-3x bandwidth
// margin above the carrier of interest, but a noticeable rolloff
// above their design corner.
//
// Now that the input is *pre-distorted* (the source signal is run
// through this profile's forward FIR before the ADC), the equalizer's
// inverse boost is RESTORING attenuated content rather than
// over-amplifying clean content. So the rolloff here is realistic,
// not destructive: the equalizer's inverse FIR tries to invert these
// dB attenuations, recovering the source.
//
// Pre-distortion and the precision sweep:
//
//   With pre-distortion, the equalizer is doing SUBSTANTIAL arithmetic
//   work (boosting up to +10 dB at Nyquist to undo the -10 dB
//   attenuation), which makes its per-stage precision sensitivity
//   much more pronounced. The posit16 / float / posit32 plans
//   therefore show larger SNR spread than the v0.6 demo (which fed
//   the equalizer a clean signal needing only a tiny correction).
//
// Why the corner stops at -10 dB and not -18 dB:
//
//   A 31-tap Hamming-windowed FIR cascade can faithfully invert a
//   -10 dB attenuation at Nyquist; a -18 dB attenuation runs into
//   the cascade's own bandwidth limit (the inverse FIR can't boost
//   that much without windowing artifacts), leaving residual error
//   in the equalized signal. -10 dB is the deepest attenuation the
//   31-tap cascade can recover with sample-level SNR-vs-source
//   above the 30 dB acceptance criterion from #172.
// ============================================================================

CalibrationProfile make_test_profile() {
	// Profile aggressiveness was chosen to satisfy two competing
	// constraints:
	//   1. Be realistic — a -3 dB corner near 100 MHz with progressive
	//      attenuation above it matches a real high-bandwidth scope.
	//   2. Stay within what a 31-tap Hamming-windowed FIR cascade can
	//      faithfully invert. The forward and inverse FIRs each have
	//      finite frequency support; trying to recover a -18 dB
	//      attenuation by +18 dB boost runs into the FIR's own
	//      bandwidth limit, accumulating residual error in the
	//      equalized signal. -10 dB at Nyquist is the deepest
	//      attenuation the 31-tap cascade can recover with
	//      sample-level SNR-vs-source > 30 dB.
	std::vector<double> freqs    = {0.0,   50e6, 100e6, 250e6, 500e6};
	std::vector<double> gains_dB = {0.0,  -0.5,  -3.0,  -6.0, -10.0};
	std::vector<double> phases   = {0.0,  -0.10, -0.20, -0.40, -0.60};
	return CalibrationProfile(std::move(freqs),
	                           std::move(gains_dB),
	                           std::move(phases));
}

// ============================================================================
// Stage timings for a single pipeline run
// ============================================================================

struct StageTimingsNs {
	double equalizer      = 0.0;
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
	std::string plan_name;          // e.g. "eq_posit16_storage_double"
	std::string eq_coeff_type;      // calibration FIR coefficient type
	std::string eq_state_type;      // calibration FIR state type
	std::string eq_sample_type;     // calibration FIR sample type
	std::string storage_type;       // trigger/ring/peak-detect/envelope type
	std::size_t storage_bytes_per_sample = 0;

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
	// Envelope SNR vs the *original clean source* (post-equalizer-output
	// vs source delayed by the combined forward+inverse FIR group delay).
	// Distinct from output_snr_db, which is plan vs. reference plan.
	// source_snr_db answers "how well does this plan recover the source
	// signal that the analog front-end distorted away?" — the equalizer's
	// reason for existing.
	double      source_snr_db        =
		std::numeric_limits<double>::quiet_NaN();
	std::size_t captured_length      = 0;

	StageTimingsNs timings;

	// Per-pixel envelope (mins, maxs) — used to write the CSV. Cast to
	// double regardless of SampleScalar precision so we have one schema.
	std::vector<double> envelope_min;
	std::vector<double> envelope_max;

	// Full post-equalizer streaming output (in double) for the
	// SNR-vs-source comparison. Populated in run_pipeline; consumed once
	// by the SNR-vs-source comparator and then cleared.
	std::vector<double> equalized_signal;
};

// ============================================================================
// run_pipeline<EqCoeff, EqState, EqSample, StorageScalar>
//
// One scope pipeline run with a per-stage precision plan:
//   - EqualizerFilter<EqCoeff, EqState, EqSample> applies the calibration
//     correction. EqSample is the input/output type of the equalizer's
//     streaming arithmetic (FIR multiply-accumulate), so this is the
//     stage where streaming-arithmetic precision actually shows up.
//   - Trigger / ring buffer / peak-detect / envelope all run in
//     StorageScalar — the precision used for memory storage and all the
//     downstream comparison-only stages.
//
// The equalizer's output is cast to StorageScalar at the equalizer ->
// trigger boundary, modelling a pipeline that re-quantizes after the
// arithmetic stage to limit downstream storage bandwidth.
// ============================================================================

template <class EqCoeff, class EqState, class EqSample, class StorageScalar>
ConfigResult run_pipeline(const std::vector<double>& adc_in_double,
                          const std::string& plan_name,
                          const std::string& eq_coeff_tag,
                          const std::string& eq_state_tag,
                          const std::string& eq_sample_tag,
                          const std::string& storage_tag,
                          std::size_t storage_bytes_per_sample,
                          const CalibrationProfile& profile) {
	ConfigResult result;
	result.plan_name                 = plan_name;
	result.eq_coeff_type             = eq_coeff_tag;
	result.eq_state_type             = eq_state_tag;
	result.eq_sample_type            = eq_sample_tag;
	result.storage_type              = storage_tag;
	result.storage_bytes_per_sample  = storage_bytes_per_sample;

	// --- Stage 1: equalizer ---
	// Project ADC samples into EqSample, equalize sample-by-sample, then
	// cast to StorageScalar at the equalizer -> trigger boundary.
	auto t0 = chrono::high_resolution_clock::now();
	EqualizerFilter<EqCoeff, EqState, EqSample>
		eq(profile, params.eq_taps, params.sample_rate_hz);

	std::vector<StorageScalar> adc_in(adc_in_double.size());
	// Capture the equalizer's output in double alongside the storage-cast
	// stream. Used by snr_db_against_source() to score how well this plan
	// recovers the original (pre-distortion) source signal.
	result.equalized_signal.assign(adc_in_double.size(), 0.0);
	for (std::size_t n = 0; n < adc_in_double.size(); ++n) {
		const EqSample in_eq  = static_cast<EqSample>(adc_in_double[n]);
		const EqSample out_eq = eq.process(in_eq);
		result.equalized_signal[n] = static_cast<double>(out_eq);
		adc_in[n] = static_cast<StorageScalar>(out_eq);
	}
	auto t1 = chrono::high_resolution_clock::now();
	result.timings.equalizer =
		chrono::duration<double, std::nano>(t1 - t0).count();

	// --- Stage 2: trigger + ring buffer in lockstep ---
	//
	// Skip the first (eq_taps - 1) samples — these are the FIR cascade's
	// settling transient, where the forward FIR (analog model) and the
	// inverse FIR (equalizer) haven't yet seen a full input window. Without
	// this skip, the trigger would fire inside the transient on the first
	// rising 0-crossing it sees in the noisy startup samples, capturing a
	// pre-trigger window full of FIR ringing instead of the steady-state
	// carrier. Cost: trim eq_taps-1 samples (30) off the head of the
	// streaming input — negligible vs num_samples (8192).
	t0 = chrono::high_resolution_clock::now();
	EdgeTrigger<StorageScalar> trig(
		static_cast<StorageScalar>(params.trigger_level),
		Slope::Rising,
		static_cast<StorageScalar>(params.trigger_hyst));
	AutoTriggerWrapper<EdgeTrigger<StorageScalar>>
		auto_trig(trig, params.auto_trigger_to);
	TriggerRingBuffer<StorageScalar> ring(params.pre_trigger, params.post_trigger);

	const std::size_t fir_settle = params.eq_taps - 1;
	bool triggered = false;
	for (std::size_t n = fir_settle; n < adc_in.size(); ++n) {
		const StorageScalar x = adc_in[n];
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
	t1 = chrono::high_resolution_clock::now();
	result.timings.trigger_ring =
		chrono::duration<double, std::nano>(t1 - t0).count();

	if (!ring.capture_complete()) {
		// No trigger fired — leave envelope empty and bail.
		return result;
	}
	auto segment = ring.captured_segment();
	result.captured_length = segment.size();

	// --- Stage 3: peak-detect decimation ---
	t0 = chrono::high_resolution_clock::now();
	PeakDetectDecimator<StorageScalar> pd(params.peak_detect_R);
	auto pd_env = pd.process_block(segment);
	t1 = chrono::high_resolution_clock::now();
	result.timings.peak_detect =
		chrono::duration<double, std::nano>(t1 - t0).count();

	// --- Stage 4: render envelope to pixel_width columns ---
	t0 = chrono::high_resolution_clock::now();
	std::span<const StorageScalar> max_span(pd_env.maxs.data(), pd_env.maxs.size());
	std::span<const StorageScalar> min_span(pd_env.mins.data(), pd_env.mins.size());
	auto disp_max = render_envelope<StorageScalar>(max_span, params.pixel_width);
	auto disp_min = render_envelope<StorageScalar>(min_span, params.pixel_width);
	t1 = chrono::high_resolution_clock::now();
	result.timings.envelope =
		chrono::duration<double, std::nano>(t1 - t0).count();

	// --- Stage 5: measurements (always accumulate in double) ---
	//
	// rms / mean read the full segment — the glitch is part of what the
	// user wants to see in those numbers.
	//
	// rise_time / period / frequency read a glitch-free pre-window so
	// the glitch's leading edge doesn't create an extra zero-crossing
	// (period bias) or push the 90% threshold above the carrier
	// amplitude (rise time would measure carrier->glitch instead).
	t0 = chrono::high_resolution_clock::now();
	result.rms = rms<StorageScalar>(segment);
	result.mean = mean<StorageScalar>(segment);
	const std::size_t mw_len =
		std::min(segment.size(), params.pre_glitch_window);
	auto measurement_window = segment.subspan(0, mw_len);
	result.rise_time_samples =
		rise_time_samples<StorageScalar>(measurement_window, 0.1, 0.9);
	result.period_samples =
		period_samples<StorageScalar>(measurement_window,
		                               static_cast<StorageScalar>(0));
	result.frequency_hz =
		frequency_hz<StorageScalar>(measurement_window, params.sample_rate_hz,
		                             static_cast<StorageScalar>(0));
	t1 = chrono::high_resolution_clock::now();
	result.timings.measurements =
		chrono::duration<double, std::nano>(t1 - t0).count();

	result.timings.total = result.timings.equalizer
	                     + result.timings.trigger_ring
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
// Equalized-signal SNR vs the original clean source
//
// The forward FIR's group delay is (eq_taps - 1) / 2 samples; the
// equalizer's matching FIR has the same group delay. So the post-equalizer
// output at sample n corresponds to the source at sample
// n - (eq_taps - 1) — accumulating both delays.
//
// We compare equalized[delay..N] vs source[0..N-delay] sample-by-sample,
// excluding the first `delay` samples on each end where the FIRs haven't
// yet seen a full input window. The metric is meaningful only when the
// post-equalizer output is approximately a delayed copy of the source —
// which is exactly the test the equalizer's design is meant to pass.
//
// Returns NaN if the equalized signal isn't available (e.g., a plan
// failed to run) or if the input length is shorter than the FIR delay.
// ============================================================================

double snr_db_against_source(const ConfigResult& test,
                              const std::vector<double>& source) {
	if (test.equalized_signal.empty() || source.empty()
	    || test.equalized_signal.size() != source.size())
		return std::numeric_limits<double>::quiet_NaN();
	// Combined group delay: forward FIR + inverse FIR, each at (N-1)/2.
	const std::size_t combined_delay = params.eq_taps - 1;
	if (source.size() <= combined_delay)
		return std::numeric_limits<double>::quiet_NaN();

	double sig_pow = 0.0, err_pow = 0.0;
	for (std::size_t n = combined_delay; n < source.size(); ++n) {
		const double s = source[n - combined_delay];
		const double e = test.equalized_signal[n] - s;
		sig_pow += s * s;
		err_pow += e * e;
	}
	if (err_pow <= 0.0) return std::numeric_limits<double>::infinity();
	return 10.0 * std::log10(sig_pow / err_pow);
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
	// Schema: pipeline,plan_name,eq_coeff,eq_state,eq_sample,storage,
	//         storage_bytes_per_sample,
	//         pixel_index,envelope_min,envelope_max,
	//         glitch_survived,glitch_peak,rise_time,rms,mean,output_snr_db
	out << "pipeline,plan_name,eq_coeff,eq_state,eq_sample,storage,"
	       "storage_bytes_per_sample,"
	       "pixel_index,envelope_min,envelope_max,"
	       "glitch_survived,glitch_peak,rise_time_samples,rms,mean,"
	       "output_snr_db,source_snr_db\n";
	for (const auto& r : results) {
		for (std::size_t i = 0; i < r.envelope_max.size(); ++i) {
			out << "scope_demo,"
			    << r.plan_name        << ','
			    << r.eq_coeff_type    << ','
			    << r.eq_state_type    << ','
			    << r.eq_sample_type   << ','
			    << r.storage_type     << ','
			    << r.storage_bytes_per_sample << ','
			    << i << ','
			    << r.envelope_min[i] << ','
			    << r.envelope_max[i] << ','
			    << (r.glitch_survived ? 1 : 0) << ','
			    << r.glitch_peak_observed << ','
			    << r.rise_time_samples << ','
			    << r.rms << ','
			    << r.mean << ','
			    << r.output_snr_db << ','
			    << r.source_snr_db
			    << '\n';
		}
	}
}

// ============================================================================
// Console summary table
// ============================================================================

void print_summary_header() {
	std::cout << "\n=== Mixed-precision scope sweep ===\n";
	std::cout << std::left  << std::setw(34) << "plan (EQ x storage)"
	          << std::right << std::setw(8)  << "B/samp"
	          << std::right << std::setw(8)  << "glitch?"
	          << std::right << std::setw(10) << "peak"
	          << std::right << std::setw(10) << "rise"
	          << std::right << std::setw(11) << "freq(MHz)"
	          << std::right << std::setw(10) << "SNRref"
	          << std::right << std::setw(10) << "SNRsrc"
	          << "\n";
	std::cout << std::string(34 + 8 + 8 + 10 + 10 + 11 + 10 + 10, '-') << "\n";
}

void print_summary_row(const ConfigResult& r) {
	// Compose a "plan_name (eq_sample x storage)" label so the row
	// shows what was narrowed and what wasn't, not just an opaque tag.
	std::string label = r.plan_name + " ("
	                  + r.eq_sample_type + " x " + r.storage_type + ")";
	if (label.size() > 33) label = label.substr(0, 33);
	std::cout << std::left  << std::setw(34) << label
	          << std::right << std::setw(8)  << r.storage_bytes_per_sample
	          << std::right << std::setw(8)  << (r.glitch_survived ? "PASS" : "fail")
	          << std::right << std::setw(10) << std::fixed;
	auto print_or_nan = [](double v, int prec, double scale = 1.0) {
		if (std::isnan(v)) std::cout << "NaN";
		else std::cout << std::setprecision(prec) << (v * scale);
	};
	print_or_nan(r.glitch_peak_observed, 3);
	std::cout << std::right << std::setw(10);
	print_or_nan(r.rise_time_samples, 2);
	std::cout << std::right << std::setw(11);
	print_or_nan(r.frequency_hz, 3, 1.0 / 1e6);
	std::cout << std::right << std::setw(10);
	print_or_nan(r.output_snr_db, 2);
	std::cout << std::right << std::setw(10);
	print_or_nan(r.source_snr_db, 2);
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
	print("equalizer",      reference.timings.equalizer);
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
	          << "  signal:    50 MHz square wave +- " << params.signal_amp
	          << " (5 ns +" << params.glitch_peak << " glitch at "
	          << params.glitch_offset_s * 1e9 << " ns)\n"
	          << "  ADC:       " << params.adc_bits << "-bit, "
	          << params.sample_rate_hz / 1e9 << " GSPS, "
	          << params.num_samples << " samples\n"
	          << "  capture:   " << params.pre_trigger << " pre + 1 trigger + "
	          << params.post_trigger << " post\n"
	          << "  display:   peak-detect R=" << params.peak_detect_R
	          << ", " << params.pixel_width << " pixels\n"
	          << "  front-end: " << params.eq_taps
	          << "-tap FIR pre-distortion (forward calibration profile)\n"
	          << "  equalizer: " << params.eq_taps
	          << "-tap FIR (inverse profile), -0.5/-3/-6/-10 dB at "
	             "50/100/250/500 MHz\n";

	const double expected_rise_samples = 0.8;
	const auto profile = make_test_profile();
	const auto source  = simulate_clean_source();
	const auto adc     = simulate_adc(source, profile);

	// =========================================================================
	// Precision plans
	//
	// Each plan picks types per stage:
	//   - (EqCoeff, EqState, EqSample): the calibration FIR's three scalars.
	//     This stage does the only streaming arithmetic in the pipeline, so
	//     its precision dominates the SNR measurement.
	//   - StorageScalar: trigger / ring buffer / peak-detect / envelope.
	//     These are comparison-only and copy-only stages — narrowing this
	//     trades memory bandwidth for nothing in measurement quality.
	//
	// Plan 0 is the all-double reference. Subsequent plans tighten one
	// dimension or the other to expose the per-stage precision trade.
	// =========================================================================
	std::array<ConfigResult, 5> results{{
		// reference: all double everywhere — the SNR baseline.
		run_pipeline<double, double, double, double>(
			adc, "reference",
			"double", "double", "double", "double",
			sizeof(double), profile),

		// High-precision EQ + ADC-native fixpnt storage. The equalizer
		// runs in double so its arithmetic doesn't add error; the
		// downstream comparison-only chain runs in 16-bit fixpnt to
		// quantify the memory savings (4x reduction vs double).
		// Expected: SNR ~ reference, storage cost halved-or-better.
		run_pipeline<double, double, double, fx16_storage>(
			adc, "eq_double_storage_fx16",
			"double", "double", "double", "fixpnt<16,12>",
			sizeof(fx16_storage), profile),

		// Narrow EQ in posit32 + double storage. Isolates the cost of
		// narrowing only the streaming arithmetic — should show a small
		// SNR drop relative to reference.
		run_pipeline<p32, p32, p32, double>(
			adc, "eq_posit32_storage_double",
			"posit<32,2>", "posit<32,2>", "posit<32,2>", "double",
			sizeof(double), profile),

		// Narrow EQ in posit16 + double storage. The headline mixed-
		// precision case: 16-bit streaming arithmetic should produce
		// visible SNR degradation (~30-40 dB lower than reference).
		run_pipeline<p16, p16, p16, double>(
			adc, "eq_posit16_storage_double",
			"posit<16,2>", "posit<16,2>", "posit<16,2>", "double",
			sizeof(double), profile),

		// FPGA-pragmatic mix: float EQ (fast on most hardware, smaller
		// than double) + ADC-native fixpnt storage (memory-optimal).
		// Expected: small SNR loss + 4x storage saving.
		run_pipeline<float, float, float, fx16_storage>(
			adc, "eq_float_storage_fx16",
			"float", "float", "float", "fixpnt<16,12>",
			sizeof(fx16_storage), profile),
	}};

	// Two SNR metrics per plan:
	//   output_snr_db = vs the all-double reference plan (apples-to-apples
	//                   comparison across plans of the same pipeline).
	//   source_snr_db = vs the original clean source signal (the equalizer's
	//                   reason for existing — measures how well it un-distorts
	//                   the front-end-distorted ADC samples back to the
	//                   source). Computed before envelope rendering so it
	//                   reflects the equalizer's full-rate streaming output.
	for (auto& r : results) {
		r.rise_time_expected = expected_rise_samples;
		r.output_snr_db = snr_db_against_reference(r, results[0]);
		r.source_snr_db = snr_db_against_source(r, source);
		// Free the captured equalized signal — it's only needed by the
		// snr_db_against_source comparator above.
		r.equalized_signal.clear();
		r.equalized_signal.shrink_to_fit();
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
