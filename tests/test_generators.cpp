// test_generators.cpp: test signal generator functions
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/signals/generators.hpp>

#include <cmath>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <vector>

using namespace sw::dsp;

constexpr double tol = 1e-6;

bool near(double a, double b, double eps = tol) {
	return std::abs(a - b) < eps;
}

void test_sine() {
	// 1 Hz sine at 100 Hz sample rate, 100 samples = 1 complete cycle
	auto s = sine<double>(100, 1.0, 100.0);
	if (!(s.size() == 100)) throw std::runtime_error("test failed: s.size() == 100");

	// s[0] should be 0 (sin(0))
	if (!(near(s[0], 0.0))) throw std::runtime_error("test failed: near(s[0], 0.0)");

	// s[25] should be 1.0 (sin(pi/2))
	if (!(near(s[25], 1.0, 1e-4))) throw std::runtime_error("test failed: near(s[25], 1.0, 1e-4)");

	// s[50] should be ~0 (sin(pi))
	if (!(near(s[50], 0.0, 1e-4))) throw std::runtime_error("test failed: near(s[50], 0.0, 1e-4)");

	// s[75] should be ~-1 (sin(3*pi/2))
	if (!(near(s[75], -1.0, 1e-4))) throw std::runtime_error("test failed: near(s[75], -1.0, 1e-4)");

	// All values should be in [-1, 1]
	for (auto v : s) {
		if (!(v >= -1.0 - tol && v <= 1.0 + tol)) throw std::runtime_error("test failed: v >= -1.0 - tol && v <= 1.0 + tol");
	}

	// Test amplitude scaling
	auto s2 = sine<double>(100, 1.0, 100.0, 0.5);
	if (!(near(s2[25], 0.5, 1e-4))) throw std::runtime_error("test failed: near(s2[25], 0.5, 1e-4)");

	std::cout << "  sine: passed\n";
}

void test_cosine() {
	auto c = cosine<double>(100, 1.0, 100.0);
	// c[0] should be 1.0 (cos(0))
	if (!(near(c[0], 1.0, 1e-4))) throw std::runtime_error("test failed: near(c[0], 1.0, 1e-4)");
	// c[25] should be ~0 (cos(pi/2))
	if (!(near(c[25], 0.0, 1e-4))) throw std::runtime_error("test failed: near(c[25], 0.0, 1e-4)");

	std::cout << "  cosine: passed\n";
}

void test_triangle() {
	auto t = triangle<double>(400, 1.0, 400.0);
	if (!(t.size() == 400)) throw std::runtime_error("test failed: t.size() == 400");

	// Triangle should start at 0
	if (!(near(t[0], 0.0, 1e-3))) throw std::runtime_error("test failed: near(t[0], 0.0, 1e-3)");

	// Peak at 1/4 period (sample 100)
	if (!(near(t[100], 1.0, 1e-2))) throw std::runtime_error("test failed: near(t[100], 1.0, 1e-2)");

	// Zero crossing at 1/2 period (sample 200)
	if (!(near(t[200], 0.0, 1e-2))) throw std::runtime_error("test failed: near(t[200], 0.0, 1e-2)");

	// Trough at 3/4 period (sample 300)
	if (!(near(t[300], -1.0, 1e-2))) throw std::runtime_error("test failed: near(t[300], -1.0, 1e-2)");

	// All values in [-1, 1]
	for (auto v : t) {
		if (!(v >= -1.0 - tol && v <= 1.0 + tol)) throw std::runtime_error("test failed: v >= -1.0 - tol && v <= 1.0 + tol");
	}

	std::cout << "  triangle: passed\n";
}

void test_square() {
	auto s = square<double>(100, 1.0, 100.0);
	if (!(s.size() == 100)) throw std::runtime_error("test failed: s.size() == 100");

	// First half should be +1, second half -1 (50% duty cycle)
	for (int n = 0; n < 50; ++n) {
		if (!(near(s[n], 1.0))) throw std::runtime_error("test failed: near(s[n], 1.0)");
	}
	for (int n = 50; n < 100; ++n) {
		if (!(near(s[n], -1.0))) throw std::runtime_error("test failed: near(s[n], -1.0)");
	}

	// Test 25% duty cycle
	auto s25 = square<double>(100, 1.0, 100.0, 1.0, 0.25);
	for (int n = 0; n < 25; ++n) {
		if (!(near(s25[n], 1.0))) throw std::runtime_error("test failed: near(s25[n], 1.0)");
	}
	for (int n = 25; n < 100; ++n) {
		if (!(near(s25[n], -1.0))) throw std::runtime_error("test failed: near(s25[n], -1.0)");
	}

	std::cout << "  square: passed\n";
}

void test_sawtooth() {
	auto s = sawtooth<double>(100, 1.0, 100.0);
	if (!(s.size() == 100)) throw std::runtime_error("test failed: s.size() == 100");

	// Starts at -1 (phase=0: 2*0-1=-1)
	if (!(near(s[0], -1.0))) throw std::runtime_error("test failed: near(s[0], -1.0)");

	// Midpoint should be ~0
	if (!(near(s[50], 0.0, 0.05))) throw std::runtime_error("test failed: near(s[50], 0.0, 0.05)");

	// End should approach +1
	if (!(s[99] > 0.9)) throw std::runtime_error("test failed: s[99] > 0.9");

	std::cout << "  sawtooth: passed\n";
}

void test_impulse() {
	auto imp = impulse<double>(10);
	if (!(imp[0] == 1.0)) throw std::runtime_error("test failed: imp[0] == 1.0");
	for (std::size_t n = 1; n < 10; ++n) {
		if (!(imp[n] == 0.0)) throw std::runtime_error("test failed: imp[n] == 0.0");
	}

	// Delayed impulse
	auto imp_d = impulse<double>(10, 5, 2.0);
	if (!(imp_d[5] == 2.0)) throw std::runtime_error("test failed: imp_d[5] == 2.0");
	if (!(imp_d[0] == 0.0)) throw std::runtime_error("test failed: imp_d[0] == 0.0");
	if (!(imp_d[4] == 0.0)) throw std::runtime_error("test failed: imp_d[4] == 0.0");
	if (!(imp_d[6] == 0.0)) throw std::runtime_error("test failed: imp_d[6] == 0.0");

	std::cout << "  impulse: passed\n";
}

void test_step() {
	auto s = step<double>(10, 3, 0.5);
	for (std::size_t n = 0; n < 3; ++n) if (!(s[n] == 0.0)) throw std::runtime_error("test failed: s[n] == 0.0");
	for (std::size_t n = 3; n < 10; ++n) if (!(s[n] == 0.5)) throw std::runtime_error("test failed: s[n] == 0.5");

	std::cout << "  step: passed\n";
}

void test_ramp() {
	auto r = ramp<double>(5, 2.0);
	if (!(near(r[0], 0.0))) throw std::runtime_error("test failed: near(r[0], 0.0)");
	if (!(near(r[1], 2.0))) throw std::runtime_error("test failed: near(r[1], 2.0)");
	if (!(near(r[2], 4.0))) throw std::runtime_error("test failed: near(r[2], 4.0)");
	if (!(near(r[3], 6.0))) throw std::runtime_error("test failed: near(r[3], 6.0)");
	if (!(near(r[4], 8.0))) throw std::runtime_error("test failed: near(r[4], 8.0)");

	std::cout << "  ramp: passed\n";
}

void test_white_noise() {
	// Deterministic with seed
	auto n1 = white_noise<double>(1000, 1.0, 42);
	auto n2 = white_noise<double>(1000, 1.0, 42);
	for (std::size_t i = 0; i < 1000; ++i) {
		if (!(n1[i] == n2[i])) throw std::runtime_error("test failed: n1[i] == n2[i]");  // same seed = same output
	}

	// All values in [-1, 1]
	for (auto v : n1) {
		if (!(v >= -1.0 && v <= 1.0)) throw std::runtime_error("test failed: v >= -1.0 && v <= 1.0");
	}

	// Mean should be near 0
	double mean = std::accumulate(n1.begin(), n1.end(), 0.0) / 1000.0;
	if (!(std::abs(mean) < 0.1)) throw std::runtime_error("test failed: std::abs(mean) < 0.1");

	std::cout << "  white_noise: passed\n";
}

void test_gaussian_noise() {
	auto n = gaussian_noise<double>(10000, 1.0, 42);

	// Mean should be near 0
	double mean = std::accumulate(n.begin(), n.end(), 0.0) / 10000.0;
	if (!(std::abs(mean) < 0.1)) throw std::runtime_error("test failed: std::abs(mean) < 0.1");

	// Stddev should be near 1.0
	double var = 0;
	for (auto v : n) var += (v - mean) * (v - mean);
	double stddev = std::sqrt(var / 10000.0);
	if (!(near(stddev, 1.0, 0.1))) throw std::runtime_error("test failed: near(stddev, 1.0, 0.1)");

	std::cout << "  gaussian_noise: passed\n";
}

void test_pink_noise() {
	auto p = pink_noise<double>(1000, 1.0, 42);
	if (!(p.size() == 1000)) throw std::runtime_error("test failed: p.size() == 1000");

	// All values should be finite
	for (auto v : p) if (!(std::isfinite(v))) throw std::runtime_error("test failed: std::isfinite(v)");

	std::cout << "  pink_noise: passed\n";
}

void test_chirp() {
	// Sweep from 100 Hz to 1000 Hz over 44100 samples at 44100 Hz (1 second)
	auto c = chirp<double>(44100, 100.0, 1000.0, 44100.0);
	if (!(c.size() == 44100)) throw std::runtime_error("test failed: c.size() == 44100");

	// All values in [-1, 1]
	for (auto v : c) {
		if (!(v >= -1.0 - tol && v <= 1.0 + tol)) throw std::runtime_error("test failed: v >= -1.0 - tol && v <= 1.0 + tol");
	}

	std::cout << "  chirp: passed\n";
}

void test_multitone() {
	// Two tones: 100 Hz and 1000 Hz
	std::vector<double> freqs = {100.0, 1000.0};
	auto m = multitone<double>(44100, std::span<const double>(freqs), 44100.0);
	if (!(m.size() == 44100)) throw std::runtime_error("test failed: m.size() == 44100");

	// All values should be finite and bounded by amplitude
	for (auto v : m) {
		if (!(std::isfinite(v))) throw std::runtime_error("test failed: std::isfinite(v)");
		if (!(v >= -1.0 - tol && v <= 1.0 + tol)) throw std::runtime_error("test failed: v >= -1.0 - tol && v <= 1.0 + tol");
	}

	std::cout << "  multitone: passed\n";
}

void test_float_compilation() {
	// Verify generators compile and work with float
	auto s = sine<float>(100, 1.0f, 100.0f);
	if (!(s.size() == 100)) throw std::runtime_error("test failed: s.size() == 100");
	if (!(near(s[25], 1.0f, 1e-3))) throw std::runtime_error("test failed: near(s[25], 1.0f, 1e-3)");

	auto t = triangle<float>(100, 1.0f, 100.0f);
	if (!(t.size() == 100)) throw std::runtime_error("test failed: t.size() == 100");

	auto sq = square<float>(100, 1.0f, 100.0f);
	if (!(sq.size() == 100)) throw std::runtime_error("test failed: sq.size() == 100");
	if (!(near(sq[0], 1.0f))) throw std::runtime_error("test failed: near(sq[0], 1.0f)");

	auto imp = impulse<float>(10);
	if (!(imp[0] == 1.0f)) throw std::runtime_error("test failed: imp[0] == 1.0f");

	auto n = white_noise<float>(100, 1.0f, 42);
	if (!(n.size() == 100)) throw std::runtime_error("test failed: n.size() == 100");

	auto c = chirp<float>(100, 100.0f, 1000.0f, 44100.0f);
	if (!(c.size() == 100)) throw std::runtime_error("test failed: c.size() == 100");

	std::cout << "  float_compilation: passed\n";
}

int main() {
	std::cout << "Signal Generator Tests\n";

	test_sine();
	test_cosine();
	test_triangle();
	test_square();
	test_sawtooth();
	test_impulse();
	test_step();
	test_ramp();
	test_white_noise();
	test_gaussian_noise();
	test_pink_noise();
	test_chirp();
	test_multitone();
	test_float_compilation();

	std::cout << "All signal generator tests passed.\n";
	return 0;
}
