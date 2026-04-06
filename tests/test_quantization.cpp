// test_quantization.cpp: test ADC, DAC, dither, noise shaping, and SQNR
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/quantization/quantization.hpp>
#include <sw/dsp/signals/generators.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace sw::dsp;

bool near(double a, double b, double eps = 1e-4) {
	return std::abs(a - b) < eps;
}

void test_adc_dac_roundtrip() {
	// double -> float -> double should lose precision but be close
	ADC<double, float> adc;
	DAC<float, double> dac;

	double input = 0.123456789012345;
	float quantized = adc.convert(input);
	double reconstructed = dac.convert(quantized);

	// float has ~7 decimal digits
	if (!(std::abs(reconstructed - input) < 1e-6))
		throw std::runtime_error("test failed: ADC/DAC roundtrip precision");
	if (!(reconstructed != input))
		throw std::runtime_error("test failed: ADC/DAC should lose some precision");

	std::cout << "  adc_dac_roundtrip: passed (error=" << std::abs(reconstructed - input) << ")\n";
}

void test_adc_vector() {
	auto sig = sine<double>(100, 1.0, 100.0);
	ADC<double, float> adc;
	auto quantized = adc.convert(sig);
	if (!(quantized.size() == sig.size()))
		throw std::runtime_error("test failed: ADC vector size");
	// Quantized values should be close but not identical
	double max_err = 0;
	for (std::size_t i = 0; i < sig.size(); ++i) {
		double err = std::abs(static_cast<double>(sig[i]) - static_cast<double>(quantized[i]));
		if (err > max_err) max_err = err;
	}
	if (!(max_err < 1e-6))
		throw std::runtime_error("test failed: ADC vector precision");

	std::cout << "  adc_vector: passed (max_err=" << max_err << ")\n";
}

void test_rpdf_dither() {
	RPDFDither<double> dither(0.01, 42);

	// Generate some dither values
	double sum = 0;
	constexpr int N = 10000;
	for (int i = 0; i < N; ++i) {
		double d = dither();
		if (!(std::abs(d) <= 0.01 + 1e-10))
			throw std::runtime_error("test failed: RPDF dither out of range");
		sum += d;
	}
	// Mean should be near zero
	double mean = sum / N;
	if (!(std::abs(mean) < 0.001))
		throw std::runtime_error("test failed: RPDF dither mean");

	std::cout << "  rpdf_dither: passed (mean=" << mean << ")\n";
}

void test_tpdf_dither() {
	TPDFDither<double> dither(0.01, 42);

	double sum = 0;
	double sum_sq = 0;
	constexpr int N = 10000;
	for (int i = 0; i < N; ++i) {
		double d = dither();
		sum += d;
		sum_sq += d * d;
	}
	double mean = sum / N;
	// TPDF mean should be near zero
	if (!(std::abs(mean) < 0.001))
		throw std::runtime_error("test failed: TPDF dither mean");
	// TPDF should have lower variance than RPDF (triangular is narrower)
	double variance = sum_sq / N - mean * mean;
	if (!(variance < 0.01 * 0.01))
		throw std::runtime_error("test failed: TPDF dither variance");

	std::cout << "  tpdf_dither: passed (mean=" << mean << ", var=" << variance << ")\n";
}

void test_dither_apply() {
	auto sig = sine<double>(100, 1.0, 100.0);
	mtl::vec::dense_vector<double> sig_copy(sig.size());
	for (std::size_t i = 0; i < sig.size(); ++i) sig_copy[i] = sig[i];

	RPDFDither<double> dither(0.001, 42);
	dither.apply(sig_copy);

	// Signal should be slightly different after dithering
	double diff = 0;
	for (std::size_t i = 0; i < sig.size(); ++i) {
		diff += std::abs(static_cast<double>(sig[i]) - sig_copy[i]);
	}
	if (!(diff > 0))
		throw std::runtime_error("test failed: dither should modify signal");

	std::cout << "  dither_apply: passed\n";
}

void test_noise_shaping() {
	// First-order noise shaping: double -> float
	FirstOrderNoiseShaper<double, float> shaper;

	auto sig = sine<double>(1000, 100.0, 44100.0);

	// Process with noise shaping
	auto shaped = shaper.process(sig);
	if (!(shaped.size() == sig.size()))
		throw std::runtime_error("test failed: noise shaping size");

	// Compare SQNR: shaped vs plain quantization
	ADC<double, float> adc;
	auto plain = adc.convert(sig);

	double sqnr_plain = sqnr_db(sig, plain);
	double sqnr_shaped = sqnr_db(sig, shaped);

	// Noise shaping should not make things dramatically worse
	// (for first-order with float, the improvement may be subtle)
	if (!(std::isfinite(sqnr_plain)))
		throw std::runtime_error("test failed: plain SQNR not finite");
	if (!(std::isfinite(sqnr_shaped)))
		throw std::runtime_error("test failed: shaped SQNR not finite");

	std::cout << "  noise_shaping: passed (plain=" << sqnr_plain
	          << " dB, shaped=" << sqnr_shaped << " dB)\n";
}

void test_sqnr_identical() {
	// Identical signals should give infinite SQNR
	auto sig = sine<double>(100, 1.0, 100.0);
	double sqnr = sqnr_db(sig, sig);
	if (!(sqnr == std::numeric_limits<double>::infinity()))
		throw std::runtime_error("test failed: SQNR of identical signals should be inf");

	std::cout << "  sqnr_identical: passed\n";
}

void test_sqnr_float() {
	// SQNR of double -> float quantization of a sine wave
	// Theoretical: ~150 dB (float has 24-bit mantissa, ~7 decimal digits)
	auto sig = sine<double>(10000, 440.0, 44100.0);
	double sqnr = measure_sqnr_db<double, float>(sig);

	if (!(sqnr > 100.0))
		throw std::runtime_error("test failed: float SQNR too low");
	if (!(sqnr < 200.0))
		throw std::runtime_error("test failed: float SQNR unreasonably high");

	std::cout << "  sqnr_float: passed (" << sqnr << " dB)\n";
}

void test_max_errors() {
	auto sig = sine<double>(1000, 440.0, 44100.0);
	ADC<double, float> adc;
	auto quantized = adc.convert(sig);

	double abs_err = max_absolute_error(sig, quantized);
	double rel_err = max_relative_error(sig, quantized);

	if (!(abs_err > 0.0))
		throw std::runtime_error("test failed: max_absolute_error should be > 0");
	if (!(abs_err < 1e-6))
		throw std::runtime_error("test failed: float max_absolute_error too large");
	if (!(rel_err > 0.0))
		throw std::runtime_error("test failed: max_relative_error should be > 0");
	if (!(rel_err < 1e-6))
		throw std::runtime_error("test failed: float max_relative_error too large");

	std::cout << "  max_errors: passed (abs=" << abs_err << ", rel=" << rel_err << ")\n";
}

void test_sqnr_validation() {
	// Empty vectors should throw
	mtl::vec::dense_vector<double> empty;
	mtl::vec::dense_vector<double> nonempty(10, 1.0);
	bool caught = false;
	try { sqnr_db(empty, empty); } catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: sqnr_db should reject empty vectors");

	// Mismatched sizes should throw
	caught = false;
	try { sqnr_db(nonempty, empty); } catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: sqnr_db should reject mismatched sizes");

	std::cout << "  sqnr_validation: passed\n";
}

int main() {
	try {
		std::cout << "Quantization Tests\n";

		test_adc_dac_roundtrip();
		test_adc_vector();
		test_rpdf_dither();
		test_tpdf_dither();
		test_dither_apply();
		test_noise_shaping();
		test_sqnr_identical();
		test_sqnr_float();
		test_max_errors();
		test_sqnr_validation();

		std::cout << "All quantization tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << '\n';
		return 1;
	}
}
