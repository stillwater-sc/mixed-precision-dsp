// test_windows.cpp: test window functions and signal utilities
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/windows/windows.hpp>
#include <sw/dsp/signals/signal.hpp>
#include <sw/dsp/signals/sampling.hpp>
#include <sw/dsp/signals/generators.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>

#include <universal/number/posit/posit.hpp>

using namespace sw::dsp;

bool near(double a, double b, double eps = 1e-6) {
	return std::abs(a - b) < eps;
}

// ========== Window Tests ==========

void test_rectangular() {
	auto w = rectangular_window<double>(64);
	if (!(w.size() == 64)) throw std::runtime_error("test failed: rectangular size");
	for (std::size_t i = 0; i < 64; ++i) {
		if (!(w[i] == 1.0)) throw std::runtime_error("test failed: rectangular value");
	}
	std::cout << "  rectangular: passed\n";
}

void test_hamming() {
	auto w = hamming_window<double>(64);
	if (!(w.size() == 64)) throw std::runtime_error("test failed: hamming size");
	// Endpoints should be 0.08 (not zero)
	if (!(near(w[0], 0.08, 0.01))) throw std::runtime_error("test failed: hamming endpoint");
	// Symmetric
	if (!(near(w[0], w[63], 1e-10))) throw std::runtime_error("test failed: hamming symmetry");
	// Peak at center
	if (!(near(w[31], 1.0, 0.01) || near(w[32], 1.0, 0.01)))
		throw std::runtime_error("test failed: hamming peak");
	std::cout << "  hamming: passed\n";
}

void test_hanning() {
	auto w = hanning_window<double>(64);
	if (!(w.size() == 64)) throw std::runtime_error("test failed: hanning size");
	// Endpoints should be 0 (Hann window)
	if (!(near(w[0], 0.0, 1e-10))) throw std::runtime_error("test failed: hanning endpoint");
	// Symmetric
	if (!(near(w[0], w[63], 1e-10))) throw std::runtime_error("test failed: hanning symmetry");
	// Peak at center ~1.0
	if (!(w[31] > 0.95 || w[32] > 0.95))
		throw std::runtime_error("test failed: hanning peak");
	std::cout << "  hanning: passed\n";
}

void test_blackman() {
	auto w = blackman_window<double>(64);
	if (!(w.size() == 64)) throw std::runtime_error("test failed: blackman size");
	// Endpoints near zero
	if (!(std::abs(w[0]) < 0.01)) throw std::runtime_error("test failed: blackman endpoint");
	// Symmetric
	if (!(near(w[0], w[63], 1e-10))) throw std::runtime_error("test failed: blackman symmetry");
	// Peak
	if (!(w[31] > 0.9 || w[32] > 0.9))
		throw std::runtime_error("test failed: blackman peak");
	std::cout << "  blackman: passed\n";
}

void test_kaiser() {
	auto w = kaiser_window<double>(64, 8.6);
	if (!(w.size() == 64)) throw std::runtime_error("test failed: kaiser size");
	// Symmetric
	if (!(near(w[0], w[63], 1e-10))) throw std::runtime_error("test failed: kaiser symmetry");
	// Peak at center = 1.0
	if (!(near(w[31], 1.0, 0.01) || near(w[32], 1.0, 0.01)))
		throw std::runtime_error("test failed: kaiser peak");
	// Endpoints should be small for large beta
	if (!(w[0] < 0.1)) throw std::runtime_error("test failed: kaiser endpoint");
	// beta=0 should give rectangular
	auto w0 = kaiser_window<double>(64, 0.0);
	if (!(near(w0[0], 1.0, 1e-10))) throw std::runtime_error("test failed: kaiser beta=0");
	std::cout << "  kaiser: passed\n";
}

void test_flat_top() {
	auto w = flat_top_window<double>(64);
	if (!(w.size() == 64)) throw std::runtime_error("test failed: flat_top size");
	// Flat-top has negative values near edges
	bool has_negative = false;
	for (std::size_t i = 0; i < 64; ++i) {
		if (w[i] < 0) has_negative = true;
	}
	if (!has_negative) throw std::runtime_error("test failed: flat_top should have negative lobes");
	// Symmetric
	if (!(near(w[0], w[63], 1e-10))) throw std::runtime_error("test failed: flat_top symmetry");
	std::cout << "  flat_top: passed\n";
}

void test_tukey() {
	auto w = tukey_window<double>(64, 0.5);
	if (!(w.size() == 64)) throw std::runtime_error("test failed: tukey size");
	// Symmetric
	if (!(near(w[0], w[63], 1e-10))) throw std::runtime_error("test failed: tukey symmetry");
	// Center should be 1.0 (flat region)
	if (!(near(w[31], 1.0, 1e-10) || near(w[32], 1.0, 1e-10)))
		throw std::runtime_error("test failed: tukey center should be 1.0");
	// alpha=0 should give rectangular
	auto w0 = tukey_window<double>(64, 0.0);
	if (!(near(w0[0], 1.0, 1e-10))) throw std::runtime_error("test failed: tukey alpha=0");
	// alpha=1 should match Hanning (endpoints near zero)
	auto w1 = tukey_window<double>(64, 1.0);
	if (!(near(w1[0], 0.0, 1e-10))) throw std::runtime_error("test failed: tukey alpha=1 endpoint");
	std::cout << "  tukey: passed\n";
}

void test_gaussian() {
	auto w = gaussian_window<double>(64, 0.4);
	if (!(w.size() == 64)) throw std::runtime_error("test failed: gaussian size");
	// Symmetric
	if (!(near(w[0], w[63], 1e-10))) throw std::runtime_error("test failed: gaussian symmetry");
	// Peak at center = 1.0
	double center = static_cast<double>(w[31]);
	double center2 = static_cast<double>(w[32]);
	if (!(near(center, 1.0, 0.01) || near(center2, 1.0, 0.01)))
		throw std::runtime_error("test failed: gaussian peak");
	// Endpoints should be small for sigma=0.4
	if (!(w[0] < 0.1)) throw std::runtime_error("test failed: gaussian endpoint");
	// Larger sigma -> wider window (less attenuation at edges)
	auto w_wide = gaussian_window<double>(64, 1.0);
	if (!(w_wide[0] > w[0]))
		throw std::runtime_error("test failed: gaussian sigma=1.0 should be wider");
	std::cout << "  gaussian: passed\n";
}

void test_dolph_chebyshev() {
	auto w = dolph_chebyshev_window<double>(64, 100.0);
	if (!(w.size() == 64)) throw std::runtime_error("test failed: dolph_chebyshev size");
	// Symmetric
	if (!(near(w[0], w[63], 1e-6))) throw std::runtime_error("test failed: dolph_chebyshev symmetry");
	// Peak = 1.0 (normalized)
	double max_val = 0;
	for (std::size_t i = 0; i < 64; ++i) {
		if (static_cast<double>(w[i]) > max_val)
			max_val = static_cast<double>(w[i]);
	}
	if (!(near(max_val, 1.0, 1e-6)))
		throw std::runtime_error("test failed: dolph_chebyshev peak should be 1.0");
	// All values should be positive for reasonable attenuation
	for (std::size_t i = 0; i < 64; ++i) {
		if (!(static_cast<double>(w[i]) > -0.01))
			throw std::runtime_error("test failed: dolph_chebyshev negative value at " +
			                         std::to_string(i));
	}
	std::cout << "  dolph_chebyshev: passed\n";
}

void test_bartlett_hann() {
	auto w = bartlett_hann_window<double>(64);
	if (!(w.size() == 64)) throw std::runtime_error("test failed: bartlett_hann size");
	// Symmetric
	if (!(near(w[0], w[63], 1e-10)))
		throw std::runtime_error("test failed: bartlett_hann symmetry");
	// Endpoints should be small (zero at n=0 for this formula)
	if (!(w[0] >= 0.0 && w[0] < 0.2))
		throw std::runtime_error("test failed: bartlett_hann endpoint");
	// Peak near center
	if (!(w[31] > 0.9 || w[32] > 0.9))
		throw std::runtime_error("test failed: bartlett_hann peak");
	std::cout << "  bartlett_hann: passed\n";
}

void test_apply_window() {
	auto sig = sine<double>(64, 1.0, 64.0);
	auto win = hamming_window<double>(64);
	auto result = windowed(sig, win);
	if (!(result.size() == 64)) throw std::runtime_error("test failed: windowed size");
	// Windowed signal should have smaller magnitude at endpoints
	if (!(std::abs(result[0]) < std::abs(sig[1]) + 0.01))
		throw std::runtime_error("test failed: windowed endpoint attenuation");
	std::cout << "  apply_window: passed\n";
}

void test_window_float() {
	// Verify windows compile with float
	auto w = hamming_window<float>(32);
	if (!(w.size() == 32)) throw std::runtime_error("test failed: float hamming size");
	auto k = kaiser_window<float>(32, 5.0f);
	if (!(k.size() == 32)) throw std::runtime_error("test failed: float kaiser size");
	std::cout << "  window_float: passed\n";
}

// ========== Signal Wrapper Tests ==========

void test_signal_wrapper() {
	Signal<double> sig(100, 44100.0);
	if (!(sig.size() == 100)) throw std::runtime_error("test failed: signal size");
	if (!(near(sig.sample_rate(), 44100.0))) throw std::runtime_error("test failed: signal sample_rate");
	if (!(near(sig.duration(), 100.0 / 44100.0, 1e-10))) throw std::runtime_error("test failed: signal duration");

	sig[0] = 1.0;
	if (!(sig[0] == 1.0)) throw std::runtime_error("test failed: signal access");

	// Construct from generator output
	auto gen = sine<double>(256, 440.0, 44100.0);
	Signal<double> sig2(std::move(gen), 44100.0);
	if (!(sig2.size() == 256)) throw std::runtime_error("test failed: signal from generator");

	std::cout << "  signal_wrapper: passed\n";
}

// ========== Sampling Tests ==========

void test_upsample() {
	mtl::vec::dense_vector<double> input({1.0, 2.0, 3.0});
	auto up = upsample(input, 3);
	if (!(up.size() == 9)) throw std::runtime_error("test failed: upsample size");
	if (!(up[0] == 1.0)) throw std::runtime_error("test failed: upsample[0]");
	if (!(up[1] == 0.0)) throw std::runtime_error("test failed: upsample[1]");
	if (!(up[2] == 0.0)) throw std::runtime_error("test failed: upsample[2]");
	if (!(up[3] == 2.0)) throw std::runtime_error("test failed: upsample[3]");
	if (!(up[6] == 3.0)) throw std::runtime_error("test failed: upsample[6]");
	std::cout << "  upsample: passed\n";
}

void test_downsample() {
	mtl::vec::dense_vector<double> input({1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0});
	auto down = downsample(input, 3);
	if (!(down.size() == 3)) throw std::runtime_error("test failed: downsample size");
	if (!(down[0] == 1.0)) throw std::runtime_error("test failed: downsample[0]");
	if (!(down[1] == 2.0)) throw std::runtime_error("test failed: downsample[1]");
	if (!(down[2] == 3.0)) throw std::runtime_error("test failed: downsample[2]");
	std::cout << "  downsample: passed\n";
}

void test_upsample_downsample_roundtrip() {
	mtl::vec::dense_vector<double> input({1.0, 2.0, 3.0, 4.0});
	auto up = upsample(input, 4);
	auto down = downsample(up, 4);
	if (!(down.size() == input.size()))
		throw std::runtime_error("test failed: roundtrip size");
	for (std::size_t i = 0; i < input.size(); ++i) {
		if (!(down[i] == input[i]))
			throw std::runtime_error("test failed: roundtrip value");
	}
	std::cout << "  upsample_downsample_roundtrip: passed\n";
}

// ============================================================================
// Posit<32,2> regression: verify window templates run intermediate math in T.
// Compares posit-designed windows against double references; agreement must
// be within posit<32,2> precision (~2^-28 ULP near unit magnitude).
// ============================================================================

void test_windows_in_posit_precision() {
	using posit_t = sw::universal::posit<32, 2>;
	constexpr std::size_t N = 64;

	auto compare_window = [&](const char* name, const auto& win_d, const auto& win_p,
	                          double eps) {
		if (win_d.size() != N || win_p.size() != N)
			throw std::runtime_error(std::string("compare_window: ") + name +
				" size mismatch: win_d=" + std::to_string(win_d.size()) +
				" win_p=" + std::to_string(win_p.size()) +
				" expected=" + std::to_string(N));
		double max_diff = 0.0;
		for (std::size_t i = 0; i < N; ++i) {
			double diff = std::abs(static_cast<double>(win_p[i]) - win_d[i]);
			if (diff > max_diff) max_diff = diff;
		}
		if (max_diff > eps)
			throw std::runtime_error(std::string("test failed: ") + name +
				" posit-vs-double max diff = " + std::to_string(max_diff) +
				" (eps=" + std::to_string(eps) + ")");
		return max_diff;
	};

	// Hamming — simple cosine
	auto ham_d = hamming_window<double>(N);
	auto ham_p = hamming_window<posit_t>(N);
	double ham_diff = compare_window("hamming", ham_d, ham_p, 1e-7);

	// Kaiser — exercises bessel_I0 template and sqrt
	auto kai_d = kaiser_window<double>(N, 8.6);
	auto kai_p = kaiser_window<posit_t>(N, 8.6);
	// Kaiser's Bessel series has more accumulated rounding; allow slightly wider.
	double kai_diff = compare_window("kaiser", kai_d, kai_p, 1e-6);

	// Gaussian — exercises exp
	auto gau_d = gaussian_window<double>(N, 0.4);
	auto gau_p = gaussian_window<posit_t>(N, 0.4);
	double gau_diff = compare_window("gaussian", gau_d, gau_p, 1e-7);

	std::cout << "  windows_in_posit_precision: hamming=" << ham_diff
	          << " kaiser=" << kai_diff
	          << " gaussian=" << gau_diff
	          << ", passed\n";
}

int main() {
	try {
		std::cout << "Window & Signal Tests\n";

		test_rectangular();
		test_hamming();
		test_hanning();
		test_blackman();
		test_kaiser();
		test_flat_top();
		test_tukey();
		test_gaussian();
		test_dolph_chebyshev();
		test_bartlett_hann();
		test_apply_window();
		test_window_float();
		test_windows_in_posit_precision();

		test_signal_wrapper();

		test_upsample();
		test_downsample();
		test_upsample_downsample_roundtrip();

		std::cout << "All window & signal tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << '\n';
		return 1;
	}
}
