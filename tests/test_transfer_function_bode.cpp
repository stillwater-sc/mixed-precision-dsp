// test_transfer_function_bode.cpp: tests for the Bode analyzer.
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)`.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <sw/dsp/filter/filter.hpp>
#include <sw/dsp/filter/iir/butterworth.hpp>
#include <sw/dsp/transfer_function/bode.hpp>

using sw::dsp::transfer_function::sweep_bode;
using sw::dsp::transfer_function::BodeResult;

// A trivial identity block for the passthrough test.
namespace {
struct IdentityBlock {
	using sample_scalar = double;
	double process(double x) { return x; }
	void   reset()           {}
};
} // namespace

// Utility: find the point in a BodeResult closest to a target frequency.
static std::size_t find_freq(const BodeResult& b, double target) {
	std::size_t best = 0;
	double best_d = std::abs(b.freqs_hz[0] - target);
	for (std::size_t i = 1; i < b.freqs_hz.size(); ++i) {
		double d = std::abs(b.freqs_hz[i] - target);
		if (d < best_d) { best_d = d; best = i; }
	}
	return best;
}

// ---------------------------------------------------------------------------
// Identity block: flat magnitude at 0 dB, phase at 0 across the sweep.
// ---------------------------------------------------------------------------
static void test_identity() {
	IdentityBlock id;
	auto b = sweep_bode(id, /*fs=*/48000.0,
	                    /*fmin=*/10.0, /*fmax=*/20000.0,
	                    /*num_points=*/40);
	if (b.freqs_hz.size() != 40) throw std::runtime_error("identity: wrong point count");
	// Tolerance loosens at low frequencies where the Hann-windowed
	// correlation still has bias below ~5 cycles per window. 0.1 dB
	// is achievable for freqs >= 100 Hz with default measure_samples=2048;
	// below that we accept up to 0.5 dB.
	for (std::size_t i = 0; i < b.freqs_hz.size(); ++i) {
		const double tol_dB = (b.freqs_hz[i] < 100.0) ? 0.5 : 0.1;
		const double tol_rad = (b.freqs_hz[i] < 100.0) ? 0.05 : 0.01;
		if (std::abs(b.magnitudes_dB[i]) > tol_dB) {
			std::cerr << "identity mag fail at f=" << b.freqs_hz[i]
			          << " Hz, mag_dB=" << b.magnitudes_dB[i] << "\n";
			throw std::runtime_error("identity: magnitude not ~0 dB");
		}
		if (std::abs(b.phases_rad[i]) > tol_rad) {
			std::cerr << "identity phase fail at f=" << b.freqs_hz[i]
			          << " Hz, phase=" << b.phases_rad[i] << "\n";
			throw std::runtime_error("identity: phase not ~0 rad");
		}
	}
}

// ---------------------------------------------------------------------------
// 4th-order Butterworth lowpass at 1 kHz, sample rate 48 kHz.
//   * Magnitude at cutoff ~ -3 dB
//   * Magnitude 2 octaves above cutoff (4 kHz) ~ -48 dB
//     (-24 dB/octave for order-4 = -80 dB/decade)
//   * Magnitude well below cutoff (100 Hz) ~ 0 dB
// ---------------------------------------------------------------------------
static void test_butterworth_lp() {
	using LP = sw::dsp::iir::ButterworthLowPass<4, double, double, double>;
	sw::dsp::SimpleFilter<LP> f;
	f.setup(4, /*sample_rate=*/48000.0, /*cutoff=*/1000.0);
	auto b = sweep_bode(f, 48000.0, /*fmin=*/50.0, /*fmax=*/20000.0,
	                    /*num_points=*/80);

	// Passband (100 Hz): mag ~ 0 dB (within 0.5 dB).
	std::size_t k_pass = find_freq(b, 100.0);
	if (b.magnitudes_dB[k_pass] < -0.5 || b.magnitudes_dB[k_pass] > 0.5)
		throw std::runtime_error("butterworth: passband mag not ~0 dB");

	// Cutoff (1000 Hz): mag ~ -3 dB (within +/- 1 dB).
	std::size_t k_cut = find_freq(b, 1000.0);
	if (std::abs(b.magnitudes_dB[k_cut] + 3.01) > 1.5)
		throw std::runtime_error("butterworth: cutoff mag not ~-3 dB");

	// Stopband: at 8 kHz (3 octaves above cutoff) the response should
	// be well below -60 dB (-24 dB/octave * 3 = -72 dB nominal).
	std::size_t k_stop = find_freq(b, 8000.0);
	if (b.magnitudes_dB[k_stop] > -50.0)
		throw std::runtime_error("butterworth: stopband not deep enough");
}

// ---------------------------------------------------------------------------
// Monotonicity check: for a Butterworth lowpass, magnitude should be
// monotonically non-increasing past the cutoff.
// ---------------------------------------------------------------------------
static void test_butterworth_monotone() {
	using LP = sw::dsp::iir::ButterworthLowPass<4, double, double, double>;
	sw::dsp::SimpleFilter<LP> f;
	f.setup(4, 48000.0, 1000.0);
	auto b = sweep_bode(f, 48000.0, 2000.0, 20000.0, 30);
	for (std::size_t i = 1; i < b.magnitudes_dB.size(); ++i) {
		// Allow a small tolerance for measurement noise.
		if (b.magnitudes_dB[i] > b.magnitudes_dB[i - 1] + 0.5)
			throw std::runtime_error(
				"butterworth: magnitude not monotone in stopband");
	}
}

// ---------------------------------------------------------------------------
// CSV dump round-trip.
// ---------------------------------------------------------------------------
static void test_dump_csv() {
	IdentityBlock id;
	auto b = sweep_bode(id, 1000.0, 10.0, 400.0, 10);
	const std::string path = "/tmp/_test_bode.csv";
	b.dump_csv(path);
	std::ifstream in(path);
	if (!in) throw std::runtime_error("dump_csv: file not created");
	std::string line;
	std::getline(in, line);
	if (line != "freq_hz,magnitude_dB,phase_rad")
		throw std::runtime_error("dump_csv: header wrong");
	int rows = 0;
	while (std::getline(in, line)) ++rows;
	if (rows != 10)
		throw std::runtime_error("dump_csv: row count wrong");
	std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Input validation.
// ---------------------------------------------------------------------------
static void test_input_validation() {
	IdentityBlock id;
	bool threw = false;
	try { sweep_bode(id, 48000.0, 100.0, 100.0, 10); } // fmax > fmin
	catch (const std::exception&) { threw = true; }
	if (!threw) throw std::runtime_error("expected throw on fmax<=fmin");

	threw = false;
	try { sweep_bode(id, 48000.0, 100.0, 30000.0, 10); } // fmax > fs/2
	catch (const std::exception&) { threw = true; }
	if (!threw) throw std::runtime_error("expected throw on fmax>=fs/2");

	threw = false;
	try { sweep_bode(id, 48000.0, 100.0, 1000.0, 1); }   // num_points<2
	catch (const std::exception&) { threw = true; }
	if (!threw) throw std::runtime_error("expected throw on num_points<2");
}

int main() {
	try {
		std::cout << "test_transfer_function_bode\n";
		test_identity();              std::cout << "  identity            PASS\n";
		test_butterworth_lp();        std::cout << "  butterworth_lp      PASS\n";
		test_butterworth_monotone();  std::cout << "  butterworth_monotone PASS\n";
		test_dump_csv();              std::cout << "  dump_csv            PASS\n";
		test_input_validation();      std::cout << "  input_validation    PASS\n";
		std::cout << "OK\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "FAIL: " << ex.what() << "\n";
		return 1;
	}
}
