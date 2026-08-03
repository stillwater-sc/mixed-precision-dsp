// test_probe_views.cpp: tests for probe domain-view helpers.
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)`.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>

#include <sw/dsp/probe/signal_probe.hpp>
#include <sw/dsp/probe/views.hpp>

using sw::dsp::probe::SignalProbe;
using sw::dsp::probe::WindowType;
using sw::dsp::probe::time_view;
using sw::dsp::probe::magnitude_spectrum;
using sw::dsp::probe::phase_spectrum;
using sw::dsp::probe::iq_constellation;

// Return the FFT-bin index with the largest magnitude in the vector.
static std::size_t argmax(const std::vector<double>& v) {
	std::size_t k = 0;
	double best = -std::numeric_limits<double>::infinity();
	for (std::size_t i = 0; i < v.size(); ++i) {
		if (v[i] > best) { best = v[i]; k = i; }
	}
	return k;
}

// ---------------------------------------------------------------------------
// time_view: raw samples cast to double, sample_rate carried through.
// ---------------------------------------------------------------------------
static void test_time_view() {
	SignalProbe<double> p("tv", 4, 100.0);
	for (int i = 0; i < 4; ++i) p.push(static_cast<double>(i));
	auto v = time_view(p);
	if (v.sample_rate_hz != 100.0)
		throw std::runtime_error("time_view: sample_rate not carried");
	if (v.label != std::string("tv"))
		throw std::runtime_error("time_view: label not carried");
	if (v.samples.size() != 4)
		throw std::runtime_error("time_view: sample count wrong");
	for (std::size_t i = 0; i < 4; ++i) {
		if (v.samples[i] != static_cast<double>(i))
			throw std::runtime_error("time_view: sample value wrong");
	}
	const std::string path = "/tmp/_test_probe_time_view.csv";
	v.dump_csv(path);
	std::ifstream in(path);
	if (!in) throw std::runtime_error("time_view: CSV not created");
	std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// magnitude_spectrum on a pure tone: peak at expected bin.
// ---------------------------------------------------------------------------
static void test_magnitude_peak_at_tone() {
	const std::size_t N  = 512;
	const double      fs = 1024.0;
	const double      f0 = 128.0;              // exact bin: 128/1024 * 512 = 64
	const double two_pi  = 2.0 * std::numbers::pi_v<double>;
	SignalProbe<double> p("tone", N, fs);
	for (std::size_t n = 0; n < N; ++n) {
		p.push(std::sin(two_pi * f0 * static_cast<double>(n) / fs));
	}
	auto spec = magnitude_spectrum(p, WindowType::Rectangular);
	std::size_t k = argmax(spec.magnitudes_dB);
	// Bin freqs = k * fs / fft_size. fft_size = next_pow2(512) = 512.
	double f_peak = spec.freqs_hz[k];
	if (std::abs(f_peak - f0) > (fs / N))
		throw std::runtime_error("magnitude peak far from tone freq");
	// Bins away from the tone should be < -20 dB below the peak.
	// (Rectangular has poor leakage, but off by 10+ bins should still
	// be well below the peak.)
	double peak_dB = spec.magnitudes_dB[k];
	if (k + 20 < spec.magnitudes_dB.size()) {
		if (spec.magnitudes_dB[k + 20] > peak_dB - 20.0)
			throw std::runtime_error("off-peak bin too loud");
	}
}

// ---------------------------------------------------------------------------
// magnitude_spectrum on DC: peak at bin 0.
// ---------------------------------------------------------------------------
static void test_magnitude_dc() {
	SignalProbe<double> p("dc", 256, 1000.0);
	for (std::size_t n = 0; n < 256; ++n) p.push(0.5);
	auto spec = magnitude_spectrum(p, WindowType::Rectangular);
	std::size_t k = argmax(spec.magnitudes_dB);
	if (k != 0) throw std::runtime_error("DC signal peak must be at bin 0");
}

// ---------------------------------------------------------------------------
// phase_spectrum: sin vs cos of the same freq should differ by 90 deg
// at the tone bin.
// ---------------------------------------------------------------------------
static void test_phase_sine_cosine() {
	const std::size_t N  = 512;
	const double      fs = 1024.0;
	const double      f0 = 128.0;
	const double two_pi  = 2.0 * std::numbers::pi_v<double>;
	SignalProbe<double> pc("cos", N, fs), ps("sin", N, fs);
	for (std::size_t n = 0; n < N; ++n) {
		double t = static_cast<double>(n) / fs;
		pc.push(std::cos(two_pi * f0 * t));
		ps.push(std::sin(two_pi * f0 * t));
	}
	auto phc = phase_spectrum(pc, /*unwrap=*/false, WindowType::Rectangular);
	auto phs = phase_spectrum(ps, /*unwrap=*/false, WindowType::Rectangular);
	auto ms  = magnitude_spectrum(pc,               WindowType::Rectangular);
	std::size_t k = argmax(ms.magnitudes_dB);
	double diff = phs.phases_rad[k] - phc.phases_rad[k];
	// Wrap to (-pi, pi].
	const double two_pi_d = 2.0 * std::numbers::pi_v<double>;
	while (diff >  std::numbers::pi_v<double>) diff -= two_pi_d;
	while (diff < -std::numbers::pi_v<double>) diff += two_pi_d;
	// sin lags cos by pi/2 (i.e., phase(sin) - phase(cos) = -pi/2).
	if (std::abs(diff + std::numbers::pi_v<double> / 2.0) > 0.05)
		throw std::runtime_error("sin-cos phase difference not ~-pi/2");
}

// ---------------------------------------------------------------------------
// iq_constellation: a complex tone traces a circle of the right radius.
// ---------------------------------------------------------------------------
static void test_iq_constellation() {
	const std::size_t N  = 256;
	const double      fs = 1000.0;
	const double      f0 = 50.0;
	const double      A  = 0.7;
	const double two_pi  = 2.0 * std::numbers::pi_v<double>;
	SignalProbe<std::complex<double>> p("iq", N, fs);
	for (std::size_t n = 0; n < N; ++n) {
		double phi = two_pi * f0 * static_cast<double>(n) / fs;
		p.push(std::complex<double>(A * std::cos(phi), A * std::sin(phi)));
	}
	auto c = iq_constellation(p);
	if (c.i_values.size() != N) throw std::runtime_error("iq: wrong count");
	for (std::size_t n = 0; n < N; ++n) {
		double r = std::sqrt(c.i_values[n] * c.i_values[n]
		                      + c.q_values[n] * c.q_values[n]);
		if (std::abs(r - A) > 1e-9)
			throw std::runtime_error("iq: point off the expected circle");
	}
	const std::string path = "/tmp/_test_probe_iq.csv";
	c.dump_csv(path);
	std::ifstream in(path);
	if (!in) throw std::runtime_error("iq: CSV not created");
	std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Two-sided complex magnitude spectrum: positive freq for e^{+j2pi f n/fs}
// lands at +f, not at -f.
// ---------------------------------------------------------------------------
static void test_complex_magnitude_sign() {
	const std::size_t N  = 256;
	const double      fs = 1000.0;
	const double      f0 = 100.0;              // exact bin
	const double two_pi  = 2.0 * std::numbers::pi_v<double>;
	SignalProbe<std::complex<double>> p("pos", N, fs);
	for (std::size_t n = 0; n < N; ++n) {
		double phi = two_pi * f0 * static_cast<double>(n) / fs;
		p.push(std::complex<double>(std::cos(phi), std::sin(phi)));
	}
	auto spec = magnitude_spectrum(p, WindowType::Rectangular);
	std::size_t k = argmax(spec.magnitudes_dB);
	double f_peak = spec.freqs_hz[k];
	if (std::abs(f_peak - f0) > (fs / N))
		throw std::runtime_error("complex tone: peak not at +f0");
}

int main() {
	try {
		std::cout << "test_probe_views\n";
		test_time_view();               std::cout << "  time_view              PASS\n";
		test_magnitude_peak_at_tone();  std::cout << "  magnitude_peak         PASS\n";
		test_magnitude_dc();            std::cout << "  magnitude_dc           PASS\n";
		test_phase_sine_cosine();       std::cout << "  phase_sine_cosine      PASS\n";
		test_iq_constellation();        std::cout << "  iq_constellation       PASS\n";
		test_complex_magnitude_sign();  std::cout << "  complex_magnitude_sign PASS\n";
		std::cout << "OK\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "FAIL: " << ex.what() << "\n";
		return 1;
	}
}
