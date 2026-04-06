// test_fir.cpp: test FIR filter and design
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/filter/fir/fir.hpp>
#include <sw/dsp/windows/windows.hpp>
#include <sw/dsp/signals/generators.hpp>
#include <sw/dsp/math/constants.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace sw::dsp;

bool near(double a, double b, double eps = 1e-6) {
	return std::abs(a - b) < eps;
}

void test_fir_impulse_response() {
	// Impulse response of FIR = tap coefficients
	mtl::vec::dense_vector<double> taps({0.25, 0.5, 0.25});
	FIRFilter<double> f(taps);

	double y0 = f.process(1.0);  // h[0] = 0.25
	double y1 = f.process(0.0);  // h[1] = 0.5
	double y2 = f.process(0.0);  // h[2] = 0.25
	double y3 = f.process(0.0);  // h[3] = 0 (FIR is finite)

	if (!(near(y0, 0.25))) throw std::runtime_error("test failed: FIR h[0]");
	if (!(near(y1, 0.5)))  throw std::runtime_error("test failed: FIR h[1]");
	if (!(near(y2, 0.25))) throw std::runtime_error("test failed: FIR h[2]");
	if (!(near(y3, 0.0)))  throw std::runtime_error("test failed: FIR h[3]");

	std::cout << "  fir_impulse_response: passed\n";
}

void test_fir_passthrough() {
	// Single-tap FIR with coefficient 1.0 = passthrough
	mtl::vec::dense_vector<double> taps({1.0});
	FIRFilter<double> f(taps);

	for (double x : {0.5, -0.3, 0.99, 0.0}) {
		double y = f.process(x);
		if (!(near(y, x))) throw std::runtime_error("test failed: passthrough");
	}

	std::cout << "  fir_passthrough: passed\n";
}

void test_fir_delay() {
	// Two-tap FIR [0, 1] = unit delay
	mtl::vec::dense_vector<double> taps({0.0, 1.0});
	FIRFilter<double> f(taps);

	double y0 = f.process(1.0);  // output = 0 (delayed)
	double y1 = f.process(0.0);  // output = 1 (previous input)
	double y2 = f.process(0.5);  // output = 0
	double y3 = f.process(0.0);  // output = 0.5

	if (!(near(y0, 0.0))) throw std::runtime_error("test failed: delay y0");
	if (!(near(y1, 1.0))) throw std::runtime_error("test failed: delay y1");
	if (!(near(y2, 0.0))) throw std::runtime_error("test failed: delay y2");
	if (!(near(y3, 0.5))) throw std::runtime_error("test failed: delay y3");

	std::cout << "  fir_delay: passed\n";
}

void test_fir_reset() {
	mtl::vec::dense_vector<double> taps({0.5, 0.5});
	FIRFilter<double> f(taps);

	f.process(1.0);
	f.process(0.5);
	f.reset();

	// After reset, delay line is zero
	double y = f.process(1.0);
	if (!(near(y, 0.5)))  // 0.5*1.0 + 0.5*0.0
		throw std::runtime_error("test failed: reset");

	std::cout << "  fir_reset: passed\n";
}

void test_fir_block_processing() {
	mtl::vec::dense_vector<double> taps({0.25, 0.5, 0.25});
	FIRFilter<double> f1(taps);
	FIRFilter<double> f2(taps);

	// Process sample-by-sample
	std::vector<double> ref;
	for (double x : {1.0, 0.0, 0.0, 0.0, 0.5, 0.0}) {
		ref.push_back(f1.process(x));
	}

	// Process as block
	std::vector<double> input = {1.0, 0.0, 0.0, 0.0, 0.5, 0.0};
	std::vector<double> output(6);
	f2.process_block(
		std::span<const double>(input),
		std::span<double>(output));

	for (std::size_t i = 0; i < 6; ++i) {
		if (!(near(ref[i], output[i], 1e-12)))
			throw std::runtime_error("test failed: block processing mismatch at " + std::to_string(i));
	}

	std::cout << "  fir_block_processing: passed\n";
}

void test_fir_mixed_precision() {
	// float coefficients, double state, float samples
	mtl::vec::dense_vector<float> taps({0.25f, 0.5f, 0.25f});
	FIRFilter<float, double, float> f(taps);

	float y = f.process(1.0f);
	if (!(std::isfinite(y)))
		throw std::runtime_error("test failed: mixed precision not finite");
	if (!(near(y, 0.25f, 1e-5)))
		throw std::runtime_error("test failed: mixed precision h[0]");

	std::cout << "  fir_mixed_precision: passed\n";
}

void test_design_lowpass() {
	// Design a 31-tap lowpass at 0.2 normalized frequency
	auto win = hamming_window<double>(31);
	auto taps = design_fir_lowpass<double>(31, 0.2, win);

	if (!(taps.size() == 31))
		throw std::runtime_error("test failed: lowpass taps size");

	// Center tap should be the largest
	double center = static_cast<double>(taps[15]);
	for (std::size_t i = 0; i < 31; ++i) {
		if (!(std::abs(static_cast<double>(taps[i])) <= std::abs(center) + 1e-10))
			throw std::runtime_error("test failed: center tap not largest");
	}

	// Symmetry (linear phase)
	for (std::size_t i = 0; i < 15; ++i) {
		if (!(near(static_cast<double>(taps[i]), static_cast<double>(taps[30 - i]), 1e-12)))
			throw std::runtime_error("test failed: lowpass symmetry");
	}

	// Frequency response: should pass DC, attenuate at 0.4
	FIRFilter<double> f(taps);
	// DC response: sum of taps
	double dc_sum = 0;
	for (std::size_t i = 0; i < taps.size(); ++i) dc_sum += static_cast<double>(taps[i]);
	double dc_db = 20.0 * std::log10(std::abs(dc_sum));
	if (!(near(dc_db, 0.0, 1.0)))
		throw std::runtime_error("test failed: lowpass DC gain not ~0 dB");

	std::cout << "  design_lowpass: passed (DC=" << dc_db << " dB)\n";
}

void test_design_highpass() {
	auto win = hamming_window<double>(31);
	auto taps = design_fir_highpass<double>(31, 0.2, win);

	if (!(taps.size() == 31))
		throw std::runtime_error("test failed: highpass taps size");

	// DC response should be near zero (highpass blocks DC)
	double dc_sum = 0;
	for (std::size_t i = 0; i < taps.size(); ++i) dc_sum += static_cast<double>(taps[i]);
	if (!(std::abs(dc_sum) < 0.01))
		throw std::runtime_error("test failed: highpass DC not near zero");

	// Nyquist response (alternating sum) should be near 1
	double nyq_sum = 0;
	for (std::size_t i = 0; i < taps.size(); ++i) {
		nyq_sum += static_cast<double>(taps[i]) * ((i % 2 == 0) ? 1.0 : -1.0);
	}
	if (!(std::abs(nyq_sum) > 0.5))
		throw std::runtime_error("test failed: highpass Nyquist response too low");

	std::cout << "  design_highpass: passed (DC=" << dc_sum << ", Nyquist=" << nyq_sum << ")\n";
}

void test_design_bandpass() {
	auto win = hamming_window<double>(63);
	auto taps = design_fir_bandpass<double>(63, 0.15, 0.35, win);

	if (!(taps.size() == 63))
		throw std::runtime_error("test failed: bandpass taps size");

	// DC should be near zero
	double dc_sum = 0;
	for (std::size_t i = 0; i < taps.size(); ++i) dc_sum += static_cast<double>(taps[i]);
	if (!(std::abs(dc_sum) < 0.05))
		throw std::runtime_error("test failed: bandpass DC not near zero");

	std::cout << "  design_bandpass: passed (DC=" << dc_sum << ")\n";
}

void test_fir_filtering_signal() {
	// Design lowpass at 2000 Hz / 44100 Hz = 0.0454
	double fc = 2000.0 / 44100.0;
	auto win = hamming_window<double>(65);
	auto taps = design_fir_lowpass<double>(65, fc, win);
	FIRFilter<double> f(taps);

	// Generate 500 Hz + 10000 Hz signal
	constexpr int N = 512;
	auto low = sine<double>(N, 500.0, 44100.0);
	auto high = sine<double>(N, 10000.0, 44100.0);

	std::vector<double> input(N), output(N);
	for (int i = 0; i < N; ++i) {
		input[i] = 0.5 * static_cast<double>(low[i]) + 0.5 * static_cast<double>(high[i]);
	}

	// Filter
	for (int i = 0; i < N; ++i) {
		output[i] = f.process(input[i]);
	}

	// After settling (skip first 65 samples for delay), output should
	// have much less high-frequency content
	double high_energy_in = 0, high_energy_out = 0;
	for (int i = 100; i < N; ++i) {
		// Crude high-frequency energy: difference between adjacent samples
		if (i > 100) {
			double d_in = input[i] - input[i-1];
			double d_out = output[i] - output[i-1];
			high_energy_in += d_in * d_in;
			high_energy_out += d_out * d_out;
		}
	}
	// FIR lowpass should reduce high-frequency energy significantly
	if (!(high_energy_out < high_energy_in * 0.1))
		throw std::runtime_error("test failed: FIR lowpass did not attenuate high frequencies");

	std::cout << "  fir_filtering_signal: passed (HF energy ratio="
	          << high_energy_out / high_energy_in << ")\n";
}

void test_fir_validation() {
	// Empty taps should throw
	bool caught = false;
	try {
		mtl::vec::dense_vector<double> empty;
		FIRFilter<double> f(empty);
	} catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: empty taps should throw");

	std::cout << "  fir_validation: passed\n";
}

int main() {
	try {
		std::cout << "FIR Filter Tests\n";

		test_fir_impulse_response();
		test_fir_passthrough();
		test_fir_delay();
		test_fir_reset();
		test_fir_block_processing();
		test_fir_mixed_precision();
		test_design_lowpass();
		test_design_highpass();
		test_design_bandpass();
		test_fir_filtering_signal();
		test_fir_validation();

		std::cout << "All FIR filter tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << '\n';
		return 1;
	}
}
