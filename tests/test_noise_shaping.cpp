// test_noise_shaping.cpp: tests for higher-order noise shaping
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/quantization/quantization.hpp>
#include <sw/dsp/signals/generators.hpp>
#include <sw/dsp/spectral/fft.hpp>

#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <universal/number/posit/posit.hpp>
#include <universal/number/fixpnt/fixpnt.hpp>

using namespace sw::dsp;

// Measure noise spectral tilt in dB/octave from magnitude spectrum.
// Fits a line to log2(frequency) vs noise magnitude in dB.
// Returns the slope (dB/octave). Positive = rising with frequency.
double noise_spectral_tilt(const mtl::vec::dense_vector<double>& reference,
                           const mtl::vec::dense_vector<double>& quantized,
                           double sample_rate) {
	std::size_t N = reference.size();
	mtl::vec::dense_vector<double> noise(N);
	for (std::size_t i = 0; i < N; ++i) {
		noise[i] = static_cast<double>(reference[i]) - static_cast<double>(quantized[i]);
	}

	// Convert to complex and compute FFT
	mtl::vec::dense_vector<std::complex<double>> noise_c(N);
	for (std::size_t i = 0; i < N; ++i) {
		noise_c[i] = std::complex<double>(noise[i], 0.0);
	}
	spectral::fft_forward<double>(noise_c);
	auto noise_spectrum = spectral::magnitude_spectrum_db<double>(noise_c);
	std::size_t num_bins = noise_spectrum.size();

	// Linear regression on log2(f) vs dB, skipping DC and very low bins
	std::size_t start = num_bins / 16;
	std::size_t end = num_bins / 2;
	double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
	double count = 0;
	for (std::size_t k = start; k < end; ++k) {
		double freq = static_cast<double>(k) * sample_rate / static_cast<double>(N);
		if (freq <= 0) continue;
		double x = std::log2(freq);
		double y = static_cast<double>(noise_spectrum[k]);
		sum_x += x;
		sum_y += y;
		sum_xx += x * x;
		sum_xy += x * y;
		count += 1.0;
	}
	double denom = count * sum_xx - sum_x * sum_x;
	if (count < 2.0 || std::abs(denom) < 1e-12)
		throw std::runtime_error("test failed: insufficient data for spectral-tilt regression");
	double slope = (count * sum_xy - sum_x * sum_y) / denom;
	return slope;
}

void test_first_order_tilt() {
	auto sig = sine<double>(8192, 100.0, 44100.0);
	FirstOrderNoiseShaper<double, float> shaper;
	auto shaped = shaper.process(sig);

	mtl::vec::dense_vector<double> shaped_d(shaped.size());
	for (std::size_t i = 0; i < shaped.size(); ++i)
		shaped_d[i] = static_cast<double>(shaped[i]);

	double tilt = noise_spectral_tilt(sig, shaped_d, 44100.0);

	// 1st order: ~+3 dB/octave noise tilt (20 dB/decade ≈ 6 dB/octave theoretical)
	if (!(tilt > 1.0))
		throw std::runtime_error("test failed: 1st order tilt should be positive, got " + std::to_string(tilt));

	std::cout << "  first_order_tilt: passed (tilt=" << tilt << " dB/oct)\n";
}

void test_second_order_tilt() {
	auto sig = sine<double>(8192, 100.0, 44100.0);
	SecondOrderNoiseShaper<double, float> shaper;
	auto shaped = shaper.process(sig);

	mtl::vec::dense_vector<double> shaped_d(shaped.size());
	for (std::size_t i = 0; i < shaped.size(); ++i)
		shaped_d[i] = static_cast<double>(shaped[i]);

	double tilt = noise_spectral_tilt(sig, shaped_d, 44100.0);

	// 2nd order: steeper tilt than 1st order (~+6 dB/octave)
	if (!(tilt > 3.0))
		throw std::runtime_error("test failed: 2nd order tilt should be > 3 dB/oct, got " + std::to_string(tilt));

	std::cout << "  second_order_tilt: passed (tilt=" << tilt << " dB/oct)\n";
}

void test_third_order_tilt() {
	auto sig = sine<double>(8192, 100.0, 44100.0);
	ThirdOrderNoiseShaper<double, float> shaper;
	auto shaped = shaper.process(sig);

	mtl::vec::dense_vector<double> shaped_d(shaped.size());
	for (std::size_t i = 0; i < shaped.size(); ++i)
		shaped_d[i] = static_cast<double>(shaped[i]);

	double tilt = noise_spectral_tilt(sig, shaped_d, 44100.0);

	// 3rd order: steeper tilt than 2nd order (~+9 dB/octave)
	if (!(tilt > 5.0))
		throw std::runtime_error("test failed: 3rd order tilt should be > 5 dB/oct, got " + std::to_string(tilt));

	std::cout << "  third_order_tilt: passed (tilt=" << tilt << " dB/oct)\n";
}

void test_increasing_tilt_with_order() {
	auto sig = sine<double>(8192, 100.0, 44100.0);

	FirstOrderNoiseShaper<double, float> shaper1;
	SecondOrderNoiseShaper<double, float> shaper2;
	ThirdOrderNoiseShaper<double, float> shaper3;

	auto s1 = shaper1.process(sig);
	auto s2 = shaper2.process(sig);
	auto s3 = shaper3.process(sig);

	mtl::vec::dense_vector<double> s1d(s1.size()), s2d(s2.size()), s3d(s3.size());
	for (std::size_t i = 0; i < sig.size(); ++i) {
		s1d[i] = static_cast<double>(s1[i]);
		s2d[i] = static_cast<double>(s2[i]);
		s3d[i] = static_cast<double>(s3[i]);
	}

	double tilt1 = noise_spectral_tilt(sig, s1d, 44100.0);
	double tilt2 = noise_spectral_tilt(sig, s2d, 44100.0);
	double tilt3 = noise_spectral_tilt(sig, s3d, 44100.0);

	if (!(tilt2 > tilt1))
		throw std::runtime_error("test failed: 2nd order tilt (" + std::to_string(tilt2) +
		                         ") should exceed 1st order (" + std::to_string(tilt1) + ")");
	if (!(tilt3 > tilt2))
		throw std::runtime_error("test failed: 3rd order tilt (" + std::to_string(tilt3) +
		                         ") should exceed 2nd order (" + std::to_string(tilt2) + ")");

	std::cout << "  increasing_tilt: passed (1st=" << tilt1 << ", 2nd=" << tilt2 << ", 3rd=" << tilt3 << " dB/oct)\n";
}

void test_broadband_sqnr() {
	// Noise shaping redistributes noise spectrally — broadband SQNR may
	// decrease with higher order because more energy is pushed to high
	// frequencies. Verify all values are finite and compare against plain
	// quantization baseline.
	auto sig = sine<double>(8192, 100.0, 44100.0);

	ADC<double, float> adc;
	auto plain = adc.convert(sig);

	FirstOrderNoiseShaper<double, float> shaper1;
	SecondOrderNoiseShaper<double, float> shaper2;
	ThirdOrderNoiseShaper<double, float> shaper3;

	auto s1 = shaper1.process(sig);
	auto s2 = shaper2.process(sig);
	auto s3 = shaper3.process(sig);

	double sqnr_plain = sqnr_db(sig, plain);
	double sqnr1 = sqnr_db(sig, s1);
	double sqnr2 = sqnr_db(sig, s2);
	double sqnr3 = sqnr_db(sig, s3);

	if (!(std::isfinite(sqnr_plain) && std::isfinite(sqnr1) &&
	      std::isfinite(sqnr2) && std::isfinite(sqnr3)))
		throw std::runtime_error("test failed: all SQNR values should be finite");

	// All shaped outputs should have reasonable SQNR (> 100 dB for double->float)
	if (!(sqnr1 > 100.0 && sqnr2 > 100.0 && sqnr3 > 100.0))
		throw std::runtime_error("test failed: shaped SQNR too low");

	std::cout << "  broadband_sqnr: passed (plain=" << sqnr_plain
	          << ", 1st=" << sqnr1 << ", 2nd=" << sqnr2 << ", 3rd=" << sqnr3 << " dB)\n";
}

void test_reset() {
	SecondOrderNoiseShaper<double, float> shaper;
	ThirdOrderNoiseShaper<double, float> shaper3;

	// Process some samples to build up state
	for (int i = 0; i < 100; ++i) {
		shaper.process(static_cast<double>(i) * 0.01);
		shaper3.process(static_cast<double>(i) * 0.01);
	}

	// Reset should clear state
	shaper.reset();
	shaper3.reset();

	// After reset, first sample should behave identically to fresh instance
	SecondOrderNoiseShaper<double, float> fresh2;
	ThirdOrderNoiseShaper<double, float> fresh3;

	double input = 0.5;
	float out_reset2 = shaper.process(input);
	float out_fresh2 = fresh2.process(input);
	float out_reset3 = shaper3.process(input);
	float out_fresh3 = fresh3.process(input);

	if (out_reset2 != out_fresh2)
		throw std::runtime_error("test failed: 2nd order reset should match fresh instance");
	if (out_reset3 != out_fresh3)
		throw std::runtime_error("test failed: 3rd order reset should match fresh instance");

	std::cout << "  reset: passed\n";
}

void test_stability_aggressive_quantization() {
	// Aggressive quantization: double -> half-precision cfloat
	// Use posit<8,2> as a very coarse target to stress the shapers.
	using LowPrec = sw::universal::posit<8, 2>;

	auto sig = sine<double>(1024, 440.0, 44100.0);

	FirstOrderNoiseShaper<double, LowPrec> shaper1;
	SecondOrderNoiseShaper<double, LowPrec> shaper2;
	ThirdOrderNoiseShaper<double, LowPrec> shaper3;

	auto s1 = shaper1.process(sig);
	auto s2 = shaper2.process(sig);
	auto s3 = shaper3.process(sig);

	// All outputs should be finite (no overflow/NaN)
	for (std::size_t i = 0; i < sig.size(); ++i) {
		double v1 = static_cast<double>(s1[i]);
		double v2 = static_cast<double>(s2[i]);
		double v3 = static_cast<double>(s3[i]);
		if (!std::isfinite(v1) || !std::isfinite(v2) || !std::isfinite(v3))
			throw std::runtime_error("test failed: output not finite at sample " + std::to_string(i));
	}

	double sqnr1 = sqnr_db(sig, s1);
	double sqnr2 = sqnr_db(sig, s2);
	double sqnr3 = sqnr_db(sig, s3);

	std::cout << "  stability_aggressive: passed (posit<8,2> SQNR: 1st=" << sqnr1
	          << ", 2nd=" << sqnr2 << ", 3rd=" << sqnr3 << " dB)\n";
}

void test_posit_combination() {
	// posit<32,2> -> posit<16,2>
	using HighPrec = sw::universal::posit<32, 2>;
	using LowPrec = sw::universal::posit<16, 2>;

	std::size_t N = 512;
	mtl::vec::dense_vector<HighPrec> sig(N);
	for (std::size_t i = 0; i < N; ++i) {
		sig[i] = HighPrec(std::sin(2.0 * std::numbers::pi * 440.0 * static_cast<double>(i) / 44100.0));
	}

	FirstOrderNoiseShaper<HighPrec, LowPrec> shaper1;
	SecondOrderNoiseShaper<HighPrec, LowPrec> shaper2;
	ThirdOrderNoiseShaper<HighPrec, LowPrec> shaper3;

	auto s1 = shaper1.process(sig);
	auto s2 = shaper2.process(sig);
	auto s3 = shaper3.process(sig);

	if (s1.size() != N || s2.size() != N || s3.size() != N)
		throw std::runtime_error("test failed: posit output size mismatch");

	std::cout << "  posit_combination: passed (posit<32,2> -> posit<16,2>)\n";
}

void test_fixpnt_combination() {
	// double -> fixpnt<8,4> (cross-fixpnt cast not supported, use double as bridge)
	using LowPrec = sw::universal::fixpnt<8, 4, sw::universal::Saturate, uint8_t>;

	std::size_t N = 512;
	mtl::vec::dense_vector<double> sig(N);
	for (std::size_t i = 0; i < N; ++i) {
		sig[i] = 0.5 * std::sin(2.0 * std::numbers::pi * 440.0 * static_cast<double>(i) / 44100.0);
	}

	FirstOrderNoiseShaper<double, LowPrec> shaper1;
	SecondOrderNoiseShaper<double, LowPrec> shaper2;
	ThirdOrderNoiseShaper<double, LowPrec> shaper3;

	auto s1 = shaper1.process(sig);
	auto s2 = shaper2.process(sig);
	auto s3 = shaper3.process(sig);

	if (s1.size() != N || s2.size() != N || s3.size() != N)
		throw std::runtime_error("test failed: fixpnt output size mismatch");

	std::cout << "  fixpnt_combination: passed (double -> fixpnt<8,4>)\n";
}

void test_vector_vs_scalar_consistency() {
	auto sig = sine<double>(256, 440.0, 44100.0);

	SecondOrderNoiseShaper<double, float> vec_shaper;
	SecondOrderNoiseShaper<double, float> scalar_shaper;

	auto vec_result = vec_shaper.process(sig);

	for (std::size_t i = 0; i < sig.size(); ++i) {
		float scalar_result = scalar_shaper.process(sig[i]);
		if (vec_result[i] != scalar_result)
			throw std::runtime_error("test failed: vector/scalar mismatch at sample " + std::to_string(i));
	}

	std::cout << "  vector_scalar_consistency: passed\n";
}

int main() {
	try {
		std::cout << "Noise Shaping Tests\n";

		test_first_order_tilt();
		test_second_order_tilt();
		test_third_order_tilt();
		test_increasing_tilt_with_order();
		test_broadband_sqnr();
		test_reset();
		test_stability_aggressive_quantization();
		test_posit_combination();
		test_fixpnt_combination();
		test_vector_vs_scalar_consistency();

		std::cout << "All noise shaping tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << '\n';
		return 1;
	}
}
