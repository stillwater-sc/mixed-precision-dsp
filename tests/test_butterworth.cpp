// test_butterworth.cpp: end-to-end Butterworth filter design and processing
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/filter/iir/butterworth.hpp>
#include <sw/dsp/filter/filter.hpp>
#include <sw/dsp/math/constants.hpp>

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace sw::dsp;

constexpr double tolerance = 1e-4;

bool near(double a, double b, double eps = tolerance) {
	return std::abs(a - b) < eps;
}

void test_butterworth_lowpass_response() {
	// 4th-order Butterworth lowpass at 1000 Hz, 44100 Hz sample rate
	iir::ButterworthLowPass<4> filter;
	filter.setup(4, 44100.0, 1000.0);

	const auto& cascade = filter.cascade();
	assert(cascade.num_stages() == 2);  // 4th order = 2 biquad stages

	// At DC (f=0), response should be ~1.0 (0 dB)
	auto r_dc = cascade.response(0.0);
	double mag_dc = std::abs(r_dc);
	assert(near(mag_dc, 1.0, 0.01));

	// At cutoff (1000/44100), response should be ~-3 dB = 0.7071
	double fc = 1000.0 / 44100.0;
	auto r_cutoff = cascade.response(fc);
	double mag_cutoff = std::abs(r_cutoff);
	double db_cutoff = 20.0 * std::log10(mag_cutoff);
	assert(near(db_cutoff, -3.0, 0.5));  // within 0.5 dB of -3 dB

	// Well above cutoff (10000 Hz), should be strongly attenuated
	double f_high = 10000.0 / 44100.0;
	auto r_high = cascade.response(f_high);
	double mag_high = std::abs(r_high);
	double db_high = 20.0 * std::log10(mag_high);
	assert(db_high < -40.0);  // 4th order = -80 dB/decade, at ~1 decade above

	std::cout << "  lowpass_response: passed (DC=" << 20*std::log10(mag_dc)
	          << "dB, cutoff=" << db_cutoff << "dB, 10kHz=" << db_high << "dB)\n";
}

void test_butterworth_highpass_response() {
	// 4th-order Butterworth highpass at 1000 Hz, 44100 Hz sample rate
	iir::ButterworthHighPass<4> filter;
	filter.setup(4, 44100.0, 1000.0);

	const auto& cascade = filter.cascade();
	assert(cascade.num_stages() == 2);

	// At Nyquist/2 (~11 kHz), response should be ~1.0
	auto r_high = cascade.response(0.25);
	double mag_high = std::abs(r_high);
	assert(near(mag_high, 1.0, 0.1));

	// At cutoff, response should be ~-3 dB
	double fc = 1000.0 / 44100.0;
	auto r_cutoff = cascade.response(fc);
	double db_cutoff = 20.0 * std::log10(std::abs(r_cutoff));
	assert(near(db_cutoff, -3.0, 0.5));

	// Well below cutoff (100 Hz), should be strongly attenuated
	double f_low = 100.0 / 44100.0;
	auto r_low = cascade.response(f_low);
	double db_low = 20.0 * std::log10(std::abs(r_low));
	assert(db_low < -40.0);

	std::cout << "  highpass_response: passed (cutoff=" << db_cutoff
	          << "dB, 100Hz=" << db_low << "dB)\n";
}

void test_butterworth_bandpass_response() {
	// 2nd-order Butterworth bandpass centered at 4000 Hz, BW 2000 Hz
	iir::ButterworthBandPass<2> filter;
	filter.setup(2, 44100.0, 4000.0, 2000.0);

	const auto& cascade = filter.cascade();
	// 2nd-order analog -> 4th-order digital for bandpass -> 2 biquad stages
	assert(cascade.num_stages() == 2);

	// At center frequency, response should peak near 0 dB
	double fc = 4000.0 / 44100.0;
	auto r_center = cascade.response(fc);
	double mag_center = std::abs(r_center);
	double db_center = 20.0 * std::log10(mag_center);
	assert(db_center > -6.0);  // should be near 0 dB

	// At DC, should be attenuated
	auto r_dc = cascade.response(0.001);  // near DC but not exactly 0
	double db_dc = 20.0 * std::log10(std::abs(r_dc));
	assert(db_dc < -10.0);

	std::cout << "  bandpass_response: passed (center=" << db_center
	          << "dB, DC=" << db_dc << "dB)\n";
}

void test_butterworth_bandstop_response() {
	// 2nd-order Butterworth bandstop centered at 4000 Hz, BW 2000 Hz
	iir::ButterworthBandStop<2> filter;
	filter.setup(2, 44100.0, 4000.0, 2000.0);

	const auto& cascade = filter.cascade();
	assert(cascade.num_stages() == 2);

	// At center frequency, response should be deeply attenuated
	double fc = 4000.0 / 44100.0;
	auto r_center = cascade.response(fc);
	double db_center = 20.0 * std::log10(std::abs(r_center));
	assert(db_center < -10.0);

	// At DC, should pass through near 0 dB
	auto r_dc = cascade.response(0.0);
	double mag_dc = std::abs(r_dc);
	assert(near(mag_dc, 1.0, 0.2));

	std::cout << "  bandstop_response: passed (center=" << db_center
	          << "dB, DC=" << 20*std::log10(mag_dc) << "dB)\n";
}

void test_butterworth_impulse_response() {
	// Process an impulse through a lowpass filter and verify:
	// 1) Output is finite
	// 2) Impulse response decays
	// 3) All three state forms produce the same result
	iir::ButterworthLowPass<4> filter;
	filter.setup(4, 44100.0, 1000.0);

	const auto& cascade = filter.cascade();
	constexpr int N = 100;

	// DirectFormII
	std::array<DirectFormII<double>, 2> state_df2{};
	std::vector<double> h_df2(N);
	for (int n = 0; n < N; ++n) {
		double x = (n == 0) ? 1.0 : 0.0;
		h_df2[n] = cascade.process(x, state_df2);
		assert(std::isfinite(h_df2[n]));
	}

	// DirectFormI
	std::array<DirectFormI<double>, 2> state_df1{};
	std::vector<double> h_df1(N);
	for (int n = 0; n < N; ++n) {
		double x = (n == 0) ? 1.0 : 0.0;
		h_df1[n] = cascade.process(x, state_df1);
	}

	// TransposedDirectFormII
	std::array<TransposedDirectFormII<double>, 2> state_tdf2{};
	std::vector<double> h_tdf2(N);
	for (int n = 0; n < N; ++n) {
		double x = (n == 0) ? 1.0 : 0.0;
		h_tdf2[n] = cascade.process(x, state_tdf2);
	}

	// All forms should agree
	for (int n = 0; n < N; ++n) {
		assert(near(h_df2[n], h_df1[n], 1e-10));
		assert(near(h_df2[n], h_tdf2[n], 1e-10));
	}

	// Impulse response energy should decay: last 10 samples smaller than first 10
	double energy_head = 0, energy_tail = 0;
	for (int n = 0; n < 10; ++n) energy_head += h_df2[n] * h_df2[n];
	for (int n = N-10; n < N; ++n) energy_tail += h_df2[n] * h_df2[n];
	assert(energy_tail < energy_head);

	std::cout << "  impulse_response: passed (h[0]=" << h_df2[0]
	          << ", h[99]=" << h_df2[N-1] << ")\n";
}

void test_butterworth_simple_filter() {
	// Test the SimpleFilter convenience wrapper
	SimpleFilter<iir::ButterworthLowPass<4>> f;
	f.setup(4, 44100.0, 1000.0);

	// Process impulse
	double y0 = f.process(1.0);
	double y1 = f.process(0.0);
	double y2 = f.process(0.0);

	assert(std::isfinite(y0));
	assert(std::isfinite(y1));
	assert(std::isfinite(y2));
	assert(y0 != 0.0);

	// Reset and verify same output
	f.reset();
	double y0b = f.process(1.0);
	assert(near(y0, y0b, 1e-15));

	std::cout << "  simple_filter: passed\n";
}

void test_butterworth_odd_order() {
	// Odd-order filter (3rd order) to test single-pole handling
	iir::ButterworthLowPass<3> filter;
	filter.setup(3, 44100.0, 2000.0);

	const auto& cascade = filter.cascade();
	assert(cascade.num_stages() == 2);  // 3rd order: 1 pair + 1 single = 2 stages

	auto r_dc = cascade.response(0.0);
	assert(near(std::abs(r_dc), 1.0, 0.01));

	// Process some samples
	std::array<DirectFormII<double>, 2> state{};
	double y = cascade.process(1.0, state);
	assert(std::isfinite(y));

	std::cout << "  odd_order: passed\n";
}

void test_butterworth_orders() {
	// Test multiple orders compile and produce valid responses
	for (int order = 1; order <= 8; ++order) {
		iir::ButterworthLowPass<8> filter;
		filter.setup(order, 48000.0, 5000.0);

		auto r_dc = filter.cascade().response(0.0);
		assert(near(std::abs(r_dc), 1.0, 0.02));

		double fc = 5000.0 / 48000.0;
		auto r_cutoff = filter.cascade().response(fc);
		double db = 20.0 * std::log10(std::abs(r_cutoff));
		assert(near(db, -3.0, 1.0));  // -3dB at cutoff within 1dB
	}

	std::cout << "  orders_1_to_8: passed\n";
}

void test_butterworth_shelf() {
	// Low shelf with +6 dB gain
	iir::ButterworthLowShelf<4> filter;
	filter.setup(4, 44100.0, 1000.0, 6.0);

	const auto& cascade = filter.cascade();
	assert(cascade.num_stages() == 2);

	// Process should be finite
	std::array<DirectFormII<double>, 2> state{};
	double y = cascade.process(1.0, state);
	assert(std::isfinite(y));

	std::cout << "  low_shelf: passed\n";
}

int main() {
	std::cout << "Phase 3: Butterworth Filter Tests\n";

	test_butterworth_lowpass_response();
	test_butterworth_highpass_response();
	test_butterworth_bandpass_response();
	test_butterworth_bandstop_response();
	test_butterworth_impulse_response();
	test_butterworth_simple_filter();
	test_butterworth_odd_order();
	test_butterworth_orders();
	test_butterworth_shelf();

	std::cout << "All Phase 3 tests passed.\n";
	return 0;
}
