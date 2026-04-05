// test_elliptic.cpp: test Elliptic (Cauer) filter designs
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/filter/iir/elliptic.hpp>
#include <sw/dsp/filter/iir/butterworth.hpp>
#include <sw/dsp/filter/filter.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/math/elliptic_integrals.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace sw::dsp;

bool near(double a, double b, double eps = 1e-4) {
	return std::abs(a - b) < eps;
}

double mag_db(auto response) {
	return 20.0 * std::log10(std::max(std::abs(response), 1e-15));
}

void test_elliptic_K() {
	// Verify elliptic K against known values
	// K(0) = pi/2
	double K0 = elliptic_K(0.0);
	if (!(near(K0, pi / 2.0, 1e-10)))
		throw std::runtime_error("test failed: K(0) != pi/2");

	// K should increase as k -> 1
	double K_half = elliptic_K(0.5);
	if (!(K_half > pi / 2.0))
		throw std::runtime_error("test failed: K(0.5) should be > pi/2");

	std::cout << "  elliptic_K: passed (K(0)=" << K0 << ", K(0.5)=" << K_half << ")\n";
}

void test_elliptic_lowpass_dc() {
	// Even-order: DC gain = -ripple_db
	iir::EllipticLowPass<4> f;
	f.setup(4, 44100.0, 1000.0, 1.0, 1.0);

	double db_dc = mag_db(f.cascade().response(0.0));
	if (!(near(db_dc, -1.0, 1.0)))
		throw std::runtime_error("test failed: elliptic LP DC gain");

	std::cout << "  elliptic_lowpass_dc: passed (DC=" << db_dc << " dB)\n";
}

void test_elliptic_lowpass_stopband() {
	// Elliptic should have very steep rolloff
	iir::EllipticLowPass<4> f;
	f.setup(4, 44100.0, 1000.0, 1.0, 1.0);

	double db_5k = mag_db(f.cascade().response(5000.0 / 44100.0));
	if (!(db_5k < -30.0))
		throw std::runtime_error("test failed: elliptic LP stopband attenuation");

	std::cout << "  elliptic_lowpass_stopband: passed (5kHz=" << db_5k << " dB)\n";
}

void test_elliptic_highpass() {
	iir::EllipticHighPass<4> f;
	f.setup(4, 44100.0, 1000.0, 1.0, 1.0);

	double db_high = mag_db(f.cascade().response(0.25));
	if (!(db_high > -3.0))
		throw std::runtime_error("test failed: elliptic HP passband");

	double db_low = mag_db(f.cascade().response(100.0 / 44100.0));
	if (!(db_low < -20.0))
		throw std::runtime_error("test failed: elliptic HP stopband");

	std::cout << "  elliptic_highpass: passed\n";
}

void test_elliptic_bandpass() {
	iir::EllipticBandPass<2> f;
	f.setup(2, 44100.0, 4000.0, 2000.0, 1.0, 1.0);

	double db_center = mag_db(f.cascade().response(4000.0 / 44100.0));
	if (!(db_center > -6.0))
		throw std::runtime_error("test failed: elliptic BP center");

	std::cout << "  elliptic_bandpass: passed (center=" << db_center << " dB)\n";
}

void test_elliptic_odd_order() {
	// Odd-order: DC gain = 0 dB
	iir::EllipticLowPass<3> f;
	f.setup(3, 44100.0, 2000.0, 1.0, 1.0);

	double db_dc = mag_db(f.cascade().response(0.0));
	if (!(near(db_dc, 0.0, 0.5)))
		throw std::runtime_error("test failed: elliptic odd-order DC gain");

	std::cout << "  elliptic_odd_order: passed (DC=" << db_dc << " dB)\n";
}

void test_elliptic_orders() {
	for (int order = 2; order <= 8; ++order) {
		iir::EllipticLowPass<8> f;
		f.setup(order, 48000.0, 5000.0, 1.0, 1.0);

		auto r_dc = f.cascade().response(0.0);
		double mag = std::abs(r_dc);
		if (!(mag > 0.5 && mag < 1.5))
			throw std::runtime_error("test failed: elliptic order " + std::to_string(order) + " DC magnitude");
	}

	std::cout << "  elliptic_orders_2_to_8: passed\n";
}

void test_elliptic_steeper_than_butterworth() {
	// Elliptic should have steeper rolloff than Butterworth at same order
	iir::EllipticLowPass<4> ell;
	ell.setup(4, 44100.0, 1000.0, 1.0, 1.0);

	iir::ButterworthLowPass<4> bw;
	bw.setup(4, 44100.0, 1000.0);

	double db_ell = mag_db(ell.cascade().response(3000.0 / 44100.0));
	double db_bw = mag_db(bw.cascade().response(3000.0 / 44100.0));

	// Elliptic should be more attenuated in the transition/stopband
	if (!(db_ell < db_bw))
		throw std::runtime_error("test failed: elliptic not steeper than butterworth");

	std::cout << "  elliptic_steeper_than_butterworth: passed (ell=" << db_ell
	          << " dB, bw=" << db_bw << " dB at 3kHz)\n";
}

void test_elliptic_simple_filter() {
	SimpleFilter<iir::EllipticLowPass<4>> f;
	f.setup(4, 44100.0, 1000.0, 1.0, 1.0);

	double y0 = f.process(1.0);
	if (!(std::isfinite(y0)))
		throw std::runtime_error("test failed: elliptic simple filter finite output");

	f.reset();
	double y0b = f.process(1.0);
	if (!(near(y0, y0b, 1e-15)))
		throw std::runtime_error("test failed: elliptic simple filter reset");

	std::cout << "  elliptic_simple_filter: passed\n";
}

int main() {
	try {
		std::cout << "Elliptic Filter Tests\n";

		test_elliptic_K();
		test_elliptic_lowpass_dc();
		test_elliptic_lowpass_stopband();
		test_elliptic_highpass();
		test_elliptic_bandpass();
		test_elliptic_odd_order();
		test_elliptic_orders();
		test_elliptic_steeper_than_butterworth();
		test_elliptic_simple_filter();

		std::cout << "All Elliptic filter tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << '\n';
		return 1;
	}
}
