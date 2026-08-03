#pragma once
// bode.hpp: numerical Bode-plot analyzer for LTI pipeline blocks.
//
// sweep_bode() probes any LTI block that exposes a
// `process(sample_scalar)` method by feeding a cosine test signal at
// each frequency in a log-spaced grid, running the block until
// steady state, then correlating the output against cos and sin
// bases to extract the block's magnitude and phase response at that
// frequency. Works on FIR filters, IIR biquad cascades, any user-
// assembled linear pipeline stage - no closed-form transfer function
// needed. The analytical companion (#158) handles closed-form
// pole/zero extraction from filters whose prototype we own.
//
// Sign convention: the input is cos(2*pi*f*n/fs), and phase is
// measured as arg(H(f)) = atan2(a_sin, a_cos) where a_sin and a_cos
// come from the standard basis projections
//   a_cos =  (2/N) sum y[n] cos(2 pi f n / fs)
//   a_sin = -(2/N) sum y[n] sin(2 pi f n / fs)
// so that a cos-input, cos-output at 0 phase gives phase = 0.
//
// LTIBlock requirements:
//   * expose `using sample_scalar = ...;`  (defines the input/output type)
//   * expose `sample_scalar process(sample_scalar)`
//   * expose `void reset()`   (called between frequencies)
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::dsp::transfer_function {

// ============================================================================
// BodeResult - one point per swept frequency
// ============================================================================

struct BodeResult {
	std::vector<double> freqs_hz;
	std::vector<double> magnitudes_dB;
	std::vector<double> phases_rad;

	void dump_csv(const std::string& path) const {
		std::ofstream out(path);
		if (!out) throw std::runtime_error(
			"BodeResult::dump_csv: cannot open " + path);
		out << "freq_hz,magnitude_dB,phase_rad\n";
		out << std::setprecision(17);
		for (std::size_t i = 0; i < freqs_hz.size(); ++i) {
			out << freqs_hz[i] << "," << magnitudes_dB[i]
			    << "," << phases_rad[i] << "\n";
		}
	}
};

// ============================================================================
// sweep_bode
// ============================================================================

// Sweep the block's magnitude and phase response over a log-spaced
// frequency grid from freq_min_hz to freq_max_hz.
//
// settle_samples: how many samples to feed before starting the
//   measurement. Must exceed the block's group delay + any transient
//   duration. Default 512 handles typical FIR (< 200 taps) and IIR
//   filters comfortably.
// target_cycles: minimum number of full periods per test frequency
//   the correlation window should span. The projection formula has
//   O(1/cycles) bias for non-integer cycles, so > 20 cycles is
//   necessary for < 0.1 dB magnitude precision.
// max_measure_samples: upper bound on the auto-picked measurement
//   window size. At the very-low-frequency end of the sweep the
//   window would otherwise become impractically large; capping trades
//   measurement bias for runtime.
template <class LTIBlock>
BodeResult sweep_bode(LTIBlock& block,
                       double sample_rate_hz,
                       double freq_min_hz,
                       double freq_max_hz,
                       std::size_t num_points = 200,
                       std::size_t settle_samples = 512,
                       double      target_cycles = 32.0,
                       std::size_t max_measure_samples = 32768) {
	if (num_points < 2)
		throw std::invalid_argument("sweep_bode: num_points must be >= 2");
	if (!(sample_rate_hz > 0.0))
		throw std::invalid_argument("sweep_bode: sample_rate_hz must be > 0");
	if (!(freq_min_hz > 0.0) || !(freq_max_hz > freq_min_hz))
		throw std::invalid_argument(
			"sweep_bode: require 0 < freq_min_hz < freq_max_hz");
	if (freq_max_hz >= sample_rate_hz / 2.0)
		throw std::invalid_argument(
			"sweep_bode: freq_max_hz must be < fs/2 (Nyquist)");

	using T = typename LTIBlock::sample_scalar;
	const double two_pi = 2.0 * std::numbers::pi_v<double>;
	const double log_min = std::log10(freq_min_hz);
	const double log_max = std::log10(freq_max_hz);

	BodeResult result;
	result.freqs_hz.reserve(num_points);
	result.magnitudes_dB.reserve(num_points);
	result.phases_rad.reserve(num_points);

	for (std::size_t k = 0; k < num_points; ++k) {
		const double frac = static_cast<double>(k)
		                     / static_cast<double>(num_points - 1);
		const double f = std::pow(10.0, log_min + frac * (log_max - log_min));
		const double omega = two_pi * f / sample_rate_hz;

		// Adaptive measurement window: at least target_cycles periods
		// of the test frequency, capped at max_measure_samples.
		std::size_t measure_samples = static_cast<std::size_t>(
			std::ceil(target_cycles * sample_rate_hz / f));
		measure_samples = std::max(measure_samples, std::size_t{512});
		measure_samples = std::min(measure_samples, max_measure_samples);

		block.reset();

		// Settle: prime the block with `settle_samples` cosine values
		// so transients decay before measurement begins.
		for (std::size_t n = 0; n < settle_samples; ++n) {
			const double phi = omega * static_cast<double>(n);
			block.process(static_cast<T>(std::cos(phi)));
		}

		// Measure: correlate output against cos and sin bases, with a
		// Hann window over the measurement region to suppress edge
		// effects from non-integer-cycle windows. Without windowing,
		// low frequencies (< 10 cycles per measure_samples window) get
		// meaningful bias from the residual DC + 2f components of
		// cos*cos and cos*sin.
		double a_cos = 0.0, a_sin = 0.0, w_sum = 0.0;
		const double denom = static_cast<double>(measure_samples - 1);
		for (std::size_t m = 0; m < measure_samples; ++m) {
			const std::size_t n = settle_samples + m;
			const double phi = omega * static_cast<double>(n);
			const double c = std::cos(phi);
			const double s = std::sin(phi);
			const double w = 0.5 * (1.0 - std::cos(
				two_pi * static_cast<double>(m) / denom));
			const double y = static_cast<double>(
				block.process(static_cast<T>(c)));
			a_cos += y * c * w;
			a_sin -= y * s * w;
			w_sum += w;
		}
		// Normalization: for cos input of unit amplitude,
		// sum(cos^2 * w) ~ w_sum / 2 in the limit of many cycles.
		// So the coherent projection reads A_out when we divide by
		// w_sum / 2.
		if (w_sum <= 0.0) w_sum = 1.0;
		a_cos *= 2.0 / w_sum;
		a_sin *= 2.0 / w_sum;

		const double mag = std::sqrt(a_cos * a_cos + a_sin * a_sin);
		const double phase = std::atan2(a_sin, a_cos);

		result.freqs_hz.push_back(f);
		result.magnitudes_dB.push_back(
			(mag > 1e-300) ? 20.0 * std::log10(mag) : -300.0);
		result.phases_rad.push_back(phase);
	}
	return result;
}

} // namespace sw::dsp::transfer_function
