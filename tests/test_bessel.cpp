// test_bessel.cpp: test Bessel and Legendre filter designs
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/filter/iir/bessel.hpp>
#include <sw/dsp/filter/iir/legendre.hpp>
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

// ========== Bessel Tests ==========

void test_bessel_lowpass_dc() {
	iir::BesselLowPass<4> f;
	f.setup(4, 44100.0, 1000.0);

	double db_dc = mag_db(f.cascade().response(0.0));
	if (!(near(db_dc, 0.0, 0.5))) throw std::runtime_error("test failed: near(db_dc, 0.0, 0.5)");

	std::cout << "  bessel_lowpass_dc: passed (DC=" << db_dc << " dB)\n";
}

void test_bessel_lowpass_rolloff() {
	iir::BesselLowPass<4> f;
	f.setup(4, 44100.0, 1000.0);

	// Bessel has slower rolloff than Butterworth but should still attenuate
	double db_5k = mag_db(f.cascade().response(5000.0 / 44100.0));
	if (!(db_5k < -10.0)) throw std::runtime_error("test failed: db_5k < -10.0");

	std::cout << "  bessel_lowpass_rolloff: passed (5kHz=" << db_5k << " dB)\n";
}

void test_bessel_highpass() {
	iir::BesselHighPass<4> f;
	f.setup(4, 44100.0, 1000.0);

	double db_high = mag_db(f.cascade().response(0.25));
	if (!(db_high > -3.0)) throw std::runtime_error("test failed: db_high > -3.0");

	double db_low = mag_db(f.cascade().response(100.0 / 44100.0));
	if (!(db_low < -20.0)) throw std::runtime_error("test failed: db_low < -20.0");

	std::cout << "  bessel_highpass: passed\n";
}

void test_bessel_bandpass() {
	iir::BesselBandPass<2> f;
	f.setup(2, 44100.0, 4000.0, 2000.0);

	double db_center = mag_db(f.cascade().response(4000.0 / 44100.0));
	if (!(db_center > -6.0)) throw std::runtime_error("test failed: db_center > -6.0");

	std::cout << "  bessel_bandpass: passed (center=" << db_center << " dB)\n";
}

void test_bessel_odd_order() {
	iir::BesselLowPass<3> f;
	f.setup(3, 44100.0, 2000.0);

	double db_dc = mag_db(f.cascade().response(0.0));
	if (!(near(db_dc, 0.0, 0.5))) throw std::runtime_error("test failed: near(db_dc, 0.0, 0.5)");

	std::cout << "  bessel_odd_order: passed (DC=" << db_dc << " dB)\n";
}

void test_bessel_orders() {
	for (int order = 1; order <= 8; ++order) {
		iir::BesselLowPass<8> f;
		f.setup(order, 48000.0, 5000.0);

		auto r_dc = f.cascade().response(0.0);
		if (!(near(std::abs(r_dc), 1.0, 0.1))) throw std::runtime_error("test failed: near(std::abs(r_dc), 1.0, 0.1)");
	}

	std::cout << "  bessel_orders_1_to_8: passed\n";
}

void test_bessel_group_delay_flatness() {
	// Bessel's key property: maximally flat group delay.
	// Check that group delay variation in the passband is small.
	iir::BesselLowPass<4> f;
	f.setup(4, 44100.0, 2000.0);

	// Estimate group delay at several frequencies by finite difference of phase
	auto phase_at = [&](double freq) -> double {
		double fn = freq / 44100.0;
		auto r = f.cascade().response(fn);
		return std::arg(r);
	};

	const double df = 10.0;  // Hz step
	double gd_100  = -(phase_at(100.0 + df) - phase_at(100.0)) / (two_pi * df / 44100.0);
	double gd_500  = -(phase_at(500.0 + df) - phase_at(500.0)) / (two_pi * df / 44100.0);
	double gd_1000 = -(phase_at(1000.0 + df) - phase_at(1000.0)) / (two_pi * df / 44100.0);

	// Group delay should be similar at these passband frequencies
	double max_gd = std::max({gd_100, gd_500, gd_1000});
	double min_gd = std::min({gd_100, gd_500, gd_1000});
	double variation = (max_gd - min_gd) / max_gd;
	if (!(variation < 0.15)) throw std::runtime_error("test failed: variation < 0.15");  // less than 15% variation in passband

	std::cout << "  bessel_group_delay_flatness: passed (variation="
	          << variation * 100 << "%)\n";
}

void test_bessel_simple_filter() {
	SimpleFilter<iir::BesselLowPass<4>> f;
	f.setup(4, 44100.0, 1000.0);

	double y0 = f.process(1.0);
	if (!(std::isfinite(y0))) throw std::runtime_error("test failed: std::isfinite(y0)");

	f.reset();
	double y0b = f.process(1.0);
	if (!(near(y0, y0b, 1e-15))) throw std::runtime_error("test failed: near(y0, y0b, 1e-15)");

	std::cout << "  bessel_simple_filter: passed\n";
}

// ========== Legendre Tests ==========

void test_legendre_lowpass_dc() {
	iir::LegendreLowPass<4> f;
	f.setup(4, 44100.0, 1000.0);

	double db_dc = mag_db(f.cascade().response(0.0));
	if (!(near(db_dc, 0.0, 0.5))) throw std::runtime_error("test failed: near(db_dc, 0.0, 0.5)");

	std::cout << "  legendre_lowpass_dc: passed (DC=" << db_dc << " dB)\n";
}

void test_legendre_lowpass_rolloff() {
	// Legendre should have steeper rolloff than Butterworth (monotonic passband)
	iir::LegendreLowPass<4> f;
	f.setup(4, 44100.0, 1000.0);

	double db_5k = mag_db(f.cascade().response(5000.0 / 44100.0));
	if (!(db_5k < -20.0)) throw std::runtime_error("test failed: db_5k < -20.0");

	std::cout << "  legendre_lowpass_rolloff: passed (5kHz=" << db_5k << " dB)\n";
}

void test_legendre_highpass() {
	iir::LegendreHighPass<4> f;
	f.setup(4, 44100.0, 1000.0);

	double db_high = mag_db(f.cascade().response(0.25));
	if (!(db_high > -3.0)) throw std::runtime_error("test failed: db_high > -3.0");

	std::cout << "  legendre_highpass: passed\n";
}

void test_legendre_odd_order() {
	iir::LegendreLowPass<3> f;
	f.setup(3, 44100.0, 2000.0);

	double db_dc = mag_db(f.cascade().response(0.0));
	if (!(near(db_dc, 0.0, 0.5))) throw std::runtime_error("test failed: near(db_dc, 0.0, 0.5)");

	std::cout << "  legendre_odd_order: passed (DC=" << db_dc << " dB)\n";
}

void test_legendre_orders() {
	for (int order = 1; order <= 6; ++order) {
		iir::LegendreLowPass<6> f;
		f.setup(order, 48000.0, 5000.0);

		auto r_dc = f.cascade().response(0.0);
		if (!(near(std::abs(r_dc), 1.0, 0.1))) throw std::runtime_error("test failed: near(std::abs(r_dc), 1.0, 0.1)");
	}

	std::cout << "  legendre_orders_1_to_6: passed\n";
}

int main() {
	try {
		std::cout << "Bessel & Legendre Filter Tests\n";

		test_bessel_lowpass_dc();
		test_bessel_lowpass_rolloff();
		test_bessel_highpass();
		test_bessel_bandpass();
		test_bessel_odd_order();
		test_bessel_orders();
		test_bessel_group_delay_flatness();
		test_bessel_simple_filter();

		test_legendre_lowpass_dc();
		test_legendre_lowpass_rolloff();
		test_legendre_highpass();
		test_legendre_odd_order();
		test_legendre_orders();

		std::cout << "All Bessel & Legendre tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << '\n';
		return 1;
	}
}
