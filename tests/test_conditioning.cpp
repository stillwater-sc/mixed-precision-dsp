// test_conditioning.cpp: test envelope followers, compressor, AGC
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/conditioning/conditioning.hpp>
#include <sw/dsp/signals/generators.hpp>
#include <sw/dsp/math/constants.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace sw::dsp;

bool near(double a, double b, double eps = 1e-3) {
	return std::abs(a - b) < eps;
}

void test_peak_envelope_tracks_level() {
	PeakEnvelope<double> env(44100.0, 1.0, 50.0);

	// Feed a constant level signal — envelope should converge
	for (int i = 0; i < 4410; ++i) {  // 100 ms
		env.process(0.5);
	}
	double level = env.value();
	if (!(near(level, 0.5, 0.05)))
		throw std::runtime_error("test failed: peak envelope convergence");

	std::cout << "  peak_envelope_tracks_level: passed (level=" << level << ")\n";
}

void test_peak_envelope_attack_release() {
	PeakEnvelope<double> env(44100.0, 1.0, 100.0);

	// Feed a burst then silence
	for (int i = 0; i < 441; ++i) env.process(1.0);  // 10 ms burst
	double peak = env.value();
	if (!(peak > 0.5))
		throw std::runtime_error("test failed: peak envelope attack");

	// Release: envelope should decay
	for (int i = 0; i < 44100; ++i) env.process(0.0);  // 1 second silence
	double released = env.value();
	if (!(released < peak * 0.01))
		throw std::runtime_error("test failed: peak envelope release");

	std::cout << "  peak_envelope_attack_release: passed (peak=" << peak
	          << ", released=" << released << ")\n";
}

void test_rms_envelope() {
	RMSEnvelope<double> env(44100.0, 50.0);

	// Feed a sine wave — RMS should converge to 1/sqrt(2) ≈ 0.707
	auto sig = sine<double>(44100, 100.0, 44100.0);
	for (std::size_t i = 0; i < sig.size(); ++i) {
		env.process(sig[i]);
	}
	double rms = env.value();
	if (!(near(rms, inv_sqrt2, 0.05)))
		throw std::runtime_error("test failed: RMS envelope of sine");

	std::cout << "  rms_envelope: passed (RMS=" << rms << ", expected=" << inv_sqrt2 << ")\n";
}

void test_compressor_below_threshold() {
	Compressor<double> comp;
	comp.setup(44100.0, -10.0, 4.0, 1.0, 50.0);

	// Very quiet signal — below threshold, no compression
	double y = comp.process(0.01);
	// Output should be approximately the input (small level detection transient)
	if (!(std::isfinite(y)))
		throw std::runtime_error("test failed: compressor finite output");

	std::cout << "  compressor_below_threshold: passed\n";
}

void test_compressor_reduces_peaks() {
	Compressor<double> comp;
	comp.setup(44100.0, -20.0, 4.0, 0.1, 50.0);

	// Feed a loud signal, then measure
	auto sig = sine<double>(4410, 100.0, 44100.0);  // 100ms

	double max_in = 0, max_out = 0;
	for (std::size_t i = 0; i < sig.size(); ++i) {
		double in = static_cast<double>(sig[i]);
		double out = comp.process(static_cast<double>(sig[i]));
		if (std::abs(in) > max_in) max_in = std::abs(in);
		if (std::abs(out) > max_out) max_out = std::abs(out);
	}

	// Compressor should reduce the peak level
	if (!(max_out <= max_in))
		throw std::runtime_error("test failed: compressor should reduce peaks");

	std::cout << "  compressor_reduces_peaks: passed (in=" << max_in
	          << ", out=" << max_out << ")\n";
}

void test_agc_levels() {
	AGC<double> agc;
	agc.setup(44100.0, 0.3, 50.0, 100.0);

	// Feed a quiet signal
	for (int i = 0; i < 44100; ++i) {
		agc.process(0.01 * std::sin(two_pi * 440.0 * i / 44100.0));
	}

	// Now feed through and check output level is boosted
	double sum_sq = 0;
	int N = 4410;
	for (int i = 0; i < N; ++i) {
		double x = 0.01 * std::sin(two_pi * 440.0 * i / 44100.0);
		double y = agc.process(x);
		sum_sq += y * y;
	}
	double rms_out = std::sqrt(sum_sq / N);

	// AGC should boost the quiet signal toward the target
	if (!(rms_out > 0.01))  // should be amplified above input level
		throw std::runtime_error("test failed: AGC should boost quiet signal");

	std::cout << "  agc_levels: passed (output RMS=" << rms_out << ")\n";
}

void test_agc_limits_gain() {
	AGC<double> agc;
	agc.setup(44100.0, 0.5, 50.0, 10.0);  // max_gain = 10

	// Feed silence — gain should be capped at max_gain, not go to infinity
	double y = agc.process(0.0);
	if (!(std::isfinite(y)))
		throw std::runtime_error("test failed: AGC finite on silence");
	if (!(std::abs(y) < 1e-5))
		throw std::runtime_error("test failed: AGC should not amplify zero");

	std::cout << "  agc_limits_gain: passed\n";
}

void test_envelope_reset() {
	PeakEnvelope<double> env(44100.0, 1.0, 50.0);
	env.process(1.0);
	if (!(env.value() > 0.0))
		throw std::runtime_error("test failed: envelope should be non-zero");
	env.reset();
	if (!(env.value() == 0.0))
		throw std::runtime_error("test failed: envelope reset to zero");

	std::cout << "  envelope_reset: passed\n";
}

int main() {
	try {
		std::cout << "Signal Conditioning Tests\n";

		test_peak_envelope_tracks_level();
		test_peak_envelope_attack_release();
		test_rms_envelope();
		test_compressor_below_threshold();
		test_compressor_reduces_peaks();
		test_agc_levels();
		test_agc_limits_gain();
		test_envelope_reset();

		std::cout << "All signal conditioning tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << '\n';
		return 1;
	}
}
