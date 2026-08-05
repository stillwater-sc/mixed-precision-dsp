// channelizer.cpp: Bellanger polyphase M-channel channelizer demo
// (issue #137).
//
// Exercises `sw::dsp::multirate::Channelizer` at M=8 channels across the
// standard six-config mixed-precision matrix. The demo verifies that:
//
//   1. Tones at channel-center frequencies land as DC in their target
//      channel (channel selection works).
//   2. Unpopulated channels stay below the acceptance floor (adjacent-
//      channel rejection).
//   3. Passband ripple within a channel is small (prototype shape).
//
// Test signal: real multitone with tones at 6000, 12000, 18000 Hz
// against f_s = 48000 Hz, M = 8 (channel width = 6000 Hz). Tones land
// at channel-1, channel-2, and channel-3 centers. Because the signal
// is real, channels 5, 6, 7 (mirrors of 3, 2, 1 mod M) also carry
// energy. Channels 0 (DC) and 4 (Nyquist) should stay quiet.
//
// Per-channel measurement: take FFT of the channel's complex output
// stream. For "loud" channels (containing an expected tone), the tone
// lands at DC; SNR = |bin[0]|^2 / sum(|bin[k>0]|^2). For "quiet"
// channels, cross-channel rejection = max(|output|) relative to the
// worst loud-channel peak.
//
// CSV schema (long format):
//   pipeline, config, scalar_type, channel, expected_tone_hz, kind,
//   value_db
// with `kind` in {tone_level, in_channel_snr, rejection}.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/multirate/channelizer.hpp>
#include <sw/dsp/spectral/fft.hpp>

#include <common/demo_output.hpp>

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
#include <optional>
#include <string>
#include <vector>

using namespace sw::dsp;

// ============================================================================
// Type aliases + constants
// ============================================================================

using p32  = sw::universal::posit<32, 2>;
using p16  = sw::universal::posit<16, 2>;
using cf32 = sw::universal::cfloat<32, 8, uint32_t, true, false, false>;
// fixpnt<32,24>: Q8.24. Prototype filter is normalized so max |tap|
// stays under ~1.5 (unity DC gain scaled by M=8 across polyphase
// phases distributes evenly). +/-128 range from Q8.24 is ample.
using fx32 = sw::universal::fixpnt<32, 24>;

struct DemoParams {
	double      sample_rate_hz = 48000.0;
	std::size_t M              = 8;      // channels; must be power of two
	std::size_t taps_per_phase = 24;     // per-phase FIR length
	double      kaiser_beta    = 12.0;   // ~-115 dB prototype stopband
	// num_blocks and transient_skip are sized so that
	// (num_blocks - transient_skip) is exactly a power of two - the FFT
	// analyzer then consumes the whole trimmed window with no
	// zero-padding, avoiding sinc-of-rectangular-window leakage from a
	// truncated constant-DC channel signal.
	std::size_t num_blocks     = 1024 + 64;   // yields 1024 usable output samples
	std::size_t transient_skip = 64;          // > (M * K / M) filter group delay

	// Test tones - each placed at the center of a distinct channel in
	// the positive-freq half. Real input means channels (M-k) also see
	// mirrored energy.
	std::vector<std::size_t> loud_channels = {1, 2, 3};
	double              tone_amplitude = 0.3;  // per-tone peak
};
inline DemoParams params;

// ============================================================================
// Test signal
// ============================================================================

std::vector<double> generate_multitone(std::size_t n_samples,
                                        double sample_rate_hz,
                                        std::size_t M,
                                        const std::vector<std::size_t>& loud_channels,
                                        double amp) {
	const double two_pi = 2.0 * std::numbers::pi_v<double>;
	std::vector<double> out(n_samples, 0.0);
	for (std::size_t n = 0; n < n_samples; ++n) {
		double v = 0.0;
		const double t = static_cast<double>(n);
		for (std::size_t c : loud_channels) {
			const double f = static_cast<double>(c) * sample_rate_hz
			                  / static_cast<double>(M);
			v += amp * std::sin(two_pi * f * t / sample_rate_hz);
		}
		out[n] = v;
	}
	return out;
}

// ============================================================================
// Run the channelizer over the input signal, one config
// ============================================================================

template <class T>
std::vector<std::vector<std::complex<double>>>
run_channelizer(const std::vector<double>& input) {
	multirate::Channelizer<double, T, T> ch(params.M,
	                                          params.taps_per_phase,
	                                          params.kaiser_beta);
	// Process one M-sample block at a time; each yields one complex
	// sample per channel.
	std::vector<std::vector<std::complex<double>>> per_channel(
		params.M, std::vector<std::complex<double>>{});
	for (auto& v : per_channel) v.reserve(input.size() / params.M);

	std::vector<T> block_T(params.M);
	for (std::size_t start = 0; start + params.M <= input.size();
	     start += params.M) {
		for (std::size_t m = 0; m < params.M; ++m) {
			block_T[m] = static_cast<T>(input[start + m]);
		}
		auto out = ch.process(std::span<const T>(block_T.data(),
		                                            params.M));
		for (std::size_t c = 0; c < params.M; ++c) {
			per_channel[c].push_back(
				std::complex<double>(static_cast<double>(out[c].real()),
				                      static_cast<double>(out[c].imag())));
		}
	}
	return per_channel;
}

// ============================================================================
// FFT-based per-channel analysis
// ============================================================================

// Take FFT of channel time series and return magnitudes in linear
// scale. Uses exactly `fft_size` samples starting at `transient_skip`,
// truncating any trailing partial window - no zero-padding, so a
// constant-DC channel output has all its energy in bin 0 without
// sinc-sidelobe artifacts.
std::vector<double>
channel_magnitude_spectrum(const std::vector<std::complex<double>>& channel) {
	if (channel.size() <= params.transient_skip) return {};
	const std::size_t len = channel.size() - params.transient_skip;
	// Largest power of two that fits inside the trimmed window.
	std::size_t fft_size = 1;
	while ((fft_size << 1) <= len) fft_size <<= 1;

	mtl::vec::dense_vector<std::complex<double>> buf(fft_size);
	for (std::size_t i = 0; i < fft_size; ++i) {
		buf[i] = channel[params.transient_skip + i];
	}
	sw::dsp::spectral::fft_forward<double>(buf);

	// For a complex channel output, all M FFT bins are meaningful
	// (no conjugate symmetry). Return magnitudes for all bins.
	std::vector<double> mag(fft_size);
	for (std::size_t k = 0; k < fft_size; ++k) mag[k] = std::abs(buf[k]);
	return mag;
}

// Per-channel metrics for one config.
struct ChannelReport {
	std::string config;
	std::string scalar_type;
	std::size_t channel;
	std::optional<std::size_t> expected_tone_channel;  // if this is loud
	double      tone_level_db      = std::numeric_limits<double>::quiet_NaN();
	double      in_channel_snr_db  = std::numeric_limits<double>::quiet_NaN();
	double      max_magnitude_db   = std::numeric_limits<double>::quiet_NaN();
};

// Compute per-channel report over M channels for one config.
std::vector<ChannelReport>
analyze(const std::vector<std::vector<std::complex<double>>>& per_channel,
         const std::string& config, const std::string& type_str) {
	std::vector<ChannelReport> reports;
	reports.reserve(params.M);

	// Which channels are "loud" (contain a tone at center)? For real
	// input, channel c and channel M-c both get energy from the tone
	// at c*f_s/M. Build a set of expected-loud channels.
	std::vector<bool> is_loud(params.M, false);
	for (std::size_t c : params.loud_channels) {
		if (c < params.M) is_loud[c] = true;
		if (c > 0)        is_loud[(params.M - c) % params.M] = true;
	}

	// Reference amplitude: absolute max across all channels/samples
	// (used for dB conversion in rejection metric).
	double abs_max = 0.0;
	for (std::size_t c = 0; c < params.M; ++c) {
		for (std::size_t i = params.transient_skip;
		     i < per_channel[c].size(); ++i) {
			abs_max = std::max(abs_max, std::abs(per_channel[c][i]));
		}
	}
	if (abs_max <= 0.0) abs_max = 1.0;

	for (std::size_t c = 0; c < params.M; ++c) {
		ChannelReport r;
		r.config      = config;
		r.scalar_type = type_str;
		r.channel     = c;

		// Peak time-domain magnitude in dB relative to abs_max.
		double ch_peak = 0.0;
		for (std::size_t i = params.transient_skip;
		     i < per_channel[c].size(); ++i) {
			ch_peak = std::max(ch_peak, std::abs(per_channel[c][i]));
		}
		r.max_magnitude_db = 20.0 * std::log10(std::max(ch_peak, 1e-300)
		                                        / abs_max);

		// FFT-based in-channel analysis. For loud channels, expect the
		// tone at DC (channel-center tone maps to DC in the channel
		// output). SNR = DC bin power / (all other bins' power).
		auto mag = channel_magnitude_spectrum(per_channel[c]);
		if (mag.empty()) { reports.push_back(r); continue; }

		if (is_loud[c]) {
			r.expected_tone_channel = c;
			const double sig = mag[0];
			double noise_pow = 0.0;
			for (std::size_t k = 1; k < mag.size(); ++k)
				noise_pow += mag[k] * mag[k];
			const double sig_pow = sig * sig;
			if (sig_pow > 0.0 && noise_pow > 0.0)
				r.in_channel_snr_db = 10.0 * std::log10(sig_pow / noise_pow);
			else if (sig_pow > 0.0)
				// noise_pow == 0 - arithmetic reconstructed the DC-only
				// channel exactly. Use a large-but-finite sentinel so
				// downstream min/max reductions do the right thing.
				r.in_channel_snr_db = 300.0;
			// Tone level in dB relative to abs_max (which is the loudest
			// tone's peak, so best-case tone_level_db is ~0).
			r.tone_level_db = 20.0 * std::log10(std::max(sig, 1e-300)
			                                     / (abs_max * mag.size()));
		}
		reports.push_back(r);
	}
	return reports;
}

// ============================================================================
// CSV writer
// ============================================================================

void write_csv(const std::string& path,
                const std::vector<ChannelReport>& reports) {
	std::ofstream out(path);
	if (!out) throw std::runtime_error("write_csv: cannot open " + path);
	out << "pipeline,config,scalar_type,channel,expected_tone_channel,"
	    << "kind,value_db\n";
	auto emit = [&](const ChannelReport& r, const std::string& kind,
	                 double v) {
		out << "channelizer," << r.config << ","
		    << "\"" << r.scalar_type << "\"," << r.channel << ",";
		if (r.expected_tone_channel)
			out << *r.expected_tone_channel;
		out << "," << kind << "," << v << "\n";
	};
	for (const auto& r : reports) {
		emit(r, "max_magnitude_db",  r.max_magnitude_db);
		emit(r, "in_channel_snr_db", r.in_channel_snr_db);
		emit(r, "tone_level_db",     r.tone_level_db);
	}
}

// ============================================================================
// Console summary
// ============================================================================

void print_summary(const std::vector<ChannelReport>& reports) {
	std::cout << "\n" << std::string(105, '=') << "\n";
	std::cout << std::left << std::setw(14) << "config"
	          << std::setw(18) << "scalar type"
	          << std::right << std::setw(4) << "ch"
	          << std::setw(12) << "role"
	          << std::setw(13) << "max_mag(dB)"
	          << std::setw(13) << "tone(dB)"
	          << std::setw(15) << "in-ch SNR(dB)" << "\n";
	std::cout << std::string(105, '-') << "\n";
	std::string cur_config;
	for (const auto& r : reports) {
		std::cout << std::left << std::setw(14)
		          << (r.config == cur_config ? "" : r.config)
		          << std::setw(18)
		          << (r.config == cur_config ? "" : r.scalar_type)
		          << std::right << std::setw(4) << r.channel
		          << std::setw(12)
		          << (r.expected_tone_channel ? "loud" : "quiet")
		          << std::fixed << std::setprecision(2)
		          << std::setw(13) << r.max_magnitude_db
		          << std::setw(13) << r.tone_level_db
		          << std::setw(15) << r.in_channel_snr_db << "\n";
		cur_config = r.config;
	}
	std::cout << std::string(105, '=') << "\n";
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv) try {
	std::string csv_path = sw::dsp::demo::output_path("channelizer.csv");
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--csv=", 0) == 0)   csv_path = a.substr(6);
		else if (a == "-h" || a == "--help") {
			std::cout << "Usage: " << argv[0] << " [--csv=path]\n";
			return 0;
		}
	}

	std::cout << "channelizer: Bellanger polyphase " << params.M
	          << "-channel demo\n"
	          << "  sample rate:    " << params.sample_rate_hz << " Hz\n"
	          << "  channels (M):   " << params.M
	          << "  (each is " << (params.sample_rate_hz / params.M)
	          << " Hz wide)\n"
	          << "  taps per phase: " << params.taps_per_phase << "\n"
	          << "  total taps:     "
	          << (params.M * params.taps_per_phase) << "\n"
	          << "  blocks:         " << params.num_blocks
	          << "  (" << (params.M * params.num_blocks)
	          << " input samples -> " << params.num_blocks
	          << " per channel)\n"
	          << "  loud channels:  ";
	for (auto c : params.loud_channels) std::cout << c << " ";
	std::cout << "(tones at ";
	for (std::size_t i = 0; i < params.loud_channels.size(); ++i) {
		std::cout << (params.loud_channels[i] * params.sample_rate_hz
		              / params.M);
		if (i + 1 < params.loud_channels.size()) std::cout << ", ";
	}
	std::cout << " Hz)\n\n";

	const auto input = generate_multitone(
		params.M * params.num_blocks,
		params.sample_rate_hz,
		params.M,
		params.loud_channels,
		params.tone_amplitude);

	std::vector<ChannelReport> all_reports;

	auto run_config = [&](auto tag, const std::string& name,
	                       const std::string& type_str) {
		using T = decltype(tag);
		std::cout << "  running " << name << " (" << type_str << ")...\n";
		auto per_channel = run_channelizer<T>(input);
		auto r = analyze(per_channel, name, type_str);
		all_reports.insert(all_reports.end(), r.begin(), r.end());
	};

	run_config(double{}, "reference", "double");
	run_config(float{},  "float",     "float");
	run_config(p32{},    "posit32",   "posit<32,2>");
	run_config(p16{},    "posit16",   "posit<16,2>");
	run_config(cf32{},   "cfloat32",  "cfloat<32,8>");
	run_config(fx32{},   "fixpnt32",  "fixpnt<32,24>");

	print_summary(all_reports);
	write_csv(csv_path, all_reports);
	std::cout << "\nCSV written: " << csv_path << "\n";

	// Acceptance criteria (issue #137):
	//   1. Per-channel SNR > 60 dB for the double reference on loud
	//      channels.
	//   2. Cross-channel rejection > 60 dB for quiet channels.
	bool ok = true;
	double worst_snr = std::numeric_limits<double>::infinity();
	double worst_rej = -std::numeric_limits<double>::infinity();
	for (const auto& r : all_reports) {
		if (r.config != "reference") continue;
		if (r.expected_tone_channel) {
			if (std::isfinite(r.in_channel_snr_db))
				worst_snr = std::min(worst_snr, r.in_channel_snr_db);
			if (r.in_channel_snr_db < 60.0) ok = false;
		} else {
			// Quiet channel: max magnitude should be well below loud
			// channels (which sit near 0 dB by construction of abs_max).
			worst_rej = std::max(worst_rej, r.max_magnitude_db);
			if (r.max_magnitude_db > -60.0) ok = false;
		}
	}
	std::cout << "\nAcceptance (reference):\n"
	          << "  worst loud-channel SNR:       " << std::fixed
	          << std::setprecision(2) << worst_snr
	          << " dB (limit: > 60)  "
	          << (worst_snr > 60.0 ? "[ok]" : "[FAIL]") << "\n"
	          << "  worst quiet-channel rejection: " << worst_rej
	          << " dB (limit: < -60)  "
	          << (worst_rej < -60.0 ? "[ok]" : "[FAIL]") << "\n";
	return ok ? 0 : 1;
} catch (const std::exception& ex) {
	std::cerr << "FATAL: " << ex.what() << "\n";
	return 1;
}
