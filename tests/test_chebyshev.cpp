// test_chebyshev.cpp: test Chebyshev Type I and Type II filter designs
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/filter/iir/chebyshev1.hpp>
#include <sw/dsp/filter/iir/chebyshev2.hpp>
#include <sw/dsp/filter/iir/rbj.hpp>
#include <sw/dsp/filter/filter.hpp>
#include <sw/dsp/math/constants.hpp>

#include <cmath>
#include <stdexcept>
#include <iostream>

using namespace sw::dsp;

bool near(double a, double b, double eps = 1e-4) {
	return std::abs(a - b) < eps;
}

double mag_db(auto response) {
	return 20.0 * std::log10(std::max(std::abs(response), 1e-15));
}

// ========== Chebyshev Type I Tests ==========

void test_cheby1_lowpass_dc() {
	// Even-order Chebyshev I: gain at DC = -ripple_db
	iir::ChebyshevILowPass<4> f;
	f.setup(4, 44100.0, 1000.0, 1.0);  // 1 dB ripple

	double db_dc = mag_db(f.cascade().response(0.0));
	// Even order: DC gain = -ripple_db
	if (!(near(db_dc, -1.0, 0.5))) throw std::runtime_error("test failed: near(db_dc, -1.0, 0.5)");

	std::cout << "  cheby1_lowpass_dc: passed (DC=" << db_dc << " dB)\n";
}

void test_cheby1_lowpass_cutoff() {
	// At cutoff, gain = -ripple_db (Chebyshev I definition)
	iir::ChebyshevILowPass<4> f;
	double ripple = 1.0;
	f.setup(4, 44100.0, 1000.0, ripple);

	double fc = 1000.0 / 44100.0;
	double db_cutoff = mag_db(f.cascade().response(fc));
	// Should be within the ripple band
	if (!(db_cutoff < 0.0)) throw std::runtime_error("test failed: db_cutoff < 0.0");
	if (!(db_cutoff > -ripple - 1.0)) throw std::runtime_error("test failed: db_cutoff > -ripple - 1.0");

	std::cout << "  cheby1_lowpass_cutoff: passed (cutoff=" << db_cutoff << " dB)\n";
}

void test_cheby1_lowpass_stopband() {
	// Chebyshev I should have steep rolloff
	iir::ChebyshevILowPass<4> f;
	f.setup(4, 44100.0, 1000.0, 1.0);

	double db_high = mag_db(f.cascade().response(5000.0 / 44100.0));
	if (!(db_high < -40.0)) throw std::runtime_error("test failed: db_high < -40.0");

	std::cout << "  cheby1_lowpass_stopband: passed (5kHz=" << db_high << " dB)\n";
}

void test_cheby1_highpass() {
	iir::ChebyshevIHighPass<4> f;
	f.setup(4, 44100.0, 1000.0, 1.0);

	// Near Nyquist/2 should be close to passband
	double db_high = mag_db(f.cascade().response(0.25));
	if (!(db_high > -3.0)) throw std::runtime_error("test failed: db_high > -3.0");

	// Below cutoff should be attenuated
	double db_low = mag_db(f.cascade().response(100.0 / 44100.0));
	if (!(db_low < -30.0)) throw std::runtime_error("test failed: db_low < -30.0");

	std::cout << "  cheby1_highpass: passed\n";
}

void test_cheby1_bandpass() {
	iir::ChebyshevIBandPass<2> f;
	f.setup(2, 44100.0, 4000.0, 2000.0, 1.0);

	// Near center should peak
	double db_center = mag_db(f.cascade().response(4000.0 / 44100.0));
	if (!(db_center > -6.0)) throw std::runtime_error("test failed: db_center > -6.0");

	std::cout << "  cheby1_bandpass: passed (center=" << db_center << " dB)\n";
}

void test_cheby1_odd_order() {
	// Odd-order: DC gain = 0 dB
	iir::ChebyshevILowPass<3> f;
	f.setup(3, 44100.0, 2000.0, 1.0);

	double db_dc = mag_db(f.cascade().response(0.0));
	if (!(near(db_dc, 0.0, 0.5))) throw std::runtime_error("test failed: near(db_dc, 0.0, 0.5)");

	std::cout << "  cheby1_odd_order: passed (DC=" << db_dc << " dB)\n";
}

// ========== Chebyshev Type II Tests ==========

void test_cheby2_lowpass_dc() {
	// Chebyshev II has unity gain at DC (monotonic passband)
	iir::ChebyshevIILowPass<4> f;
	f.setup(4, 44100.0, 1000.0, 40.0);  // 40 dB stopband

	double db_dc = mag_db(f.cascade().response(0.0));
	if (!(near(db_dc, 0.0, 0.5))) throw std::runtime_error("test failed: near(db_dc, 0.0, 0.5)");

	std::cout << "  cheby2_lowpass_dc: passed (DC=" << db_dc << " dB)\n";
}

void test_cheby2_lowpass_stopband() {
	// Stopband should reach specified attenuation
	iir::ChebyshevIILowPass<4> f;
	double stop_db = 40.0;
	f.setup(4, 44100.0, 1000.0, stop_db);

	// Well into stopband
	double db_stop = mag_db(f.cascade().response(10000.0 / 44100.0));
	if (!(db_stop < -30.0)) throw std::runtime_error("test failed: db_stop < -30.0");  // should approach -40 dB

	std::cout << "  cheby2_lowpass_stopband: passed (10kHz=" << db_stop << " dB)\n";
}

void test_cheby2_highpass() {
	iir::ChebyshevIIHighPass<4> f;
	f.setup(4, 44100.0, 1000.0, 40.0);

	double db_high = mag_db(f.cascade().response(0.25));
	if (!(db_high > -3.0)) throw std::runtime_error("test failed: db_high > -3.0");

	double db_low = mag_db(f.cascade().response(100.0 / 44100.0));
	if (!(db_low < -20.0)) throw std::runtime_error("test failed: db_low < -20.0");

	std::cout << "  cheby2_highpass: passed\n";
}

void test_cheby2_bandpass() {
	iir::ChebyshevIIBandPass<2> f;
	f.setup(2, 44100.0, 4000.0, 2000.0, 40.0);

	double db_center = mag_db(f.cascade().response(4000.0 / 44100.0));
	if (!(db_center > -6.0)) throw std::runtime_error("test failed: db_center > -6.0");

	std::cout << "  cheby2_bandpass: passed (center=" << db_center << " dB)\n";
}

// ========== RBJ Tests ==========

void test_rbj_lowpass() {
	iir::rbj::LowPass<> f;
	f.setup(44100.0, 1000.0, 0.7071);

	double db_dc = mag_db(f.cascade().response(0.0));
	if (!(near(db_dc, 0.0, 0.5))) throw std::runtime_error("test failed: near(db_dc, 0.0, 0.5)");

	double db_cutoff = mag_db(f.cascade().response(1000.0 / 44100.0));
	if (!(near(db_cutoff, -3.0, 1.0))) throw std::runtime_error("test failed: near(db_cutoff, -3.0, 1.0)");

	std::cout << "  rbj_lowpass: passed (DC=" << db_dc << ", cutoff=" << db_cutoff << " dB)\n";
}

void test_rbj_highpass() {
	iir::rbj::HighPass<> f;
	f.setup(44100.0, 1000.0);

	double db_high = mag_db(f.cascade().response(0.4));
	if (!(db_high > -1.0)) throw std::runtime_error("test failed: db_high > -1.0");

	double db_low = mag_db(f.cascade().response(100.0 / 44100.0));
	if (!(db_low < -20.0)) throw std::runtime_error("test failed: db_low < -20.0");

	std::cout << "  rbj_highpass: passed\n";
}

void test_rbj_bandpass() {
	iir::rbj::BandPass<> f;
	f.setup(44100.0, 4000.0, 1.0);

	double db_center = mag_db(f.cascade().response(4000.0 / 44100.0));
	if (!(db_center > -3.0)) throw std::runtime_error("test failed: db_center > -3.0");

	std::cout << "  rbj_bandpass: passed (center=" << db_center << " dB)\n";
}

void test_rbj_bandstop() {
	iir::rbj::BandStop<> f;
	f.setup(44100.0, 4000.0, 1.0);

	double db_center = mag_db(f.cascade().response(4000.0 / 44100.0));
	if (!(db_center < -20.0)) throw std::runtime_error("test failed: db_center < -20.0");

	double db_pass = mag_db(f.cascade().response(0.0));
	if (!(near(db_pass, 0.0, 0.5))) throw std::runtime_error("test failed: near(db_pass, 0.0, 0.5)");

	std::cout << "  rbj_bandstop: passed (center=" << db_center << ", DC=" << db_pass << " dB)\n";
}

void test_rbj_allpass() {
	iir::rbj::AllPass<> f;
	f.setup(44100.0, 4000.0, 0.7071);

	// Allpass: magnitude = 1 at all frequencies
	for (double freq = 0.01; freq < 0.49; freq += 0.05) {
		double mag = std::abs(f.cascade().response(freq));
		if (!(near(mag, 1.0, 0.01))) throw std::runtime_error("test failed: near(mag, 1.0, 0.01)");
	}

	std::cout << "  rbj_allpass: passed (|H|=1 at all frequencies)\n";
}

void test_rbj_shelves() {
	// Low shelf with +6 dB gain
	iir::rbj::LowShelf<> ls;
	ls.setup(44100.0, 1000.0, 6.0);

	double db_dc = mag_db(ls.cascade().response(0.0));
	if (!(near(db_dc, 6.0, 1.0))) throw std::runtime_error("test failed: near(db_dc, 6.0, 1.0)");

	// High shelf with +6 dB gain
	iir::rbj::HighShelf<> hs;
	hs.setup(44100.0, 1000.0, 6.0);

	double db_high = mag_db(hs.cascade().response(0.4));
	if (!(near(db_high, 6.0, 1.0))) throw std::runtime_error("test failed: near(db_high, 6.0, 1.0)");

	std::cout << "  rbj_shelves: passed (low DC=" << db_dc << ", high top=" << db_high << " dB)\n";
}

void test_rbj_simple_filter() {
	SimpleFilter<iir::rbj::LowPass<>> f;
	f.setup(44100.0, 1000.0);

	double y0 = f.process(1.0);
	double y1 = f.process(0.0);
	if (!(std::isfinite(y0))) throw std::runtime_error("test failed: std::isfinite(y0)");
	if (!(std::isfinite(y1))) throw std::runtime_error("test failed: std::isfinite(y1)");

	f.reset();
	double y0b = f.process(1.0);
	if (!(near(y0, y0b, 1e-15))) throw std::runtime_error("test failed: near(y0, y0b, 1e-15)");

	std::cout << "  rbj_simple_filter: passed\n";
}

int main() {
	try {
		std::cout << "Chebyshev & RBJ Filter Tests\n";

		test_cheby1_lowpass_dc();
		test_cheby1_lowpass_cutoff();
		test_cheby1_lowpass_stopband();
		test_cheby1_highpass();
		test_cheby1_bandpass();
		test_cheby1_odd_order();

		test_cheby2_lowpass_dc();
		test_cheby2_lowpass_stopband();
		test_cheby2_highpass();
		test_cheby2_bandpass();

		test_rbj_lowpass();
		test_rbj_highpass();
		test_rbj_bandpass();
		test_rbj_bandstop();
		test_rbj_allpass();
		test_rbj_shelves();
		test_rbj_simple_filter();

		std::cout << "All Chebyshev & RBJ tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << '\n';
		return 1;
	}
}
