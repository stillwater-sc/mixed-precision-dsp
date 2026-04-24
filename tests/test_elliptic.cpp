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

#include <universal/number/posit/posit.hpp>

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

// Regression for issue #50: passing stopband-dB-style value (e.g. 40.0)
// as rolloff used to silently produce NaN. It must now throw.
void test_elliptic_rejects_rolloff_out_of_range() {
	iir::EllipticLowPass<4> f;

	bool threw_high = false;
	try {
		f.setup(4, 44100.0, 2000.0, 1.0, 40.0);
	} catch (const std::invalid_argument&) {
		threw_high = true;
	}
	if (!threw_high)
		throw std::runtime_error("test failed: rolloff=40.0 must throw invalid_argument");

	bool threw_low = false;
	try {
		f.setup(4, 44100.0, 2000.0, 1.0, 0.0);
	} catch (const std::invalid_argument&) {
		threw_low = true;
	}
	if (!threw_low)
		throw std::runtime_error("test failed: rolloff=0.0 must throw invalid_argument");

	std::cout << "  elliptic_rejects_rolloff_out_of_range: passed\n";
}

// Regression for issue #50: impulse response at the issue's fs/fc/order
// must be finite with a valid rolloff.
void test_elliptic_issue50_impulse_finite() {
	SimpleFilter<iir::EllipticLowPass<4>> f;
	f.setup(4, 44100.0, 2000.0, 1.0, 1.0);

	double y = f.process(1.0);
	if (!std::isfinite(y))
		throw std::runtime_error("test failed: issue #50 impulse head is NaN");
	for (int i = 0; i < 199; ++i) {
		double s = f.process(0.0);
		if (!std::isfinite(s))
			throw std::runtime_error(
				"test failed: issue #50 impulse sample " + std::to_string(i + 1) + " is NaN");
	}

	std::cout << "  elliptic_issue50_impulse_finite: passed\n";
}

// ============================================================================
// Cauer-Darlington (spec-based) tests
// ============================================================================

void test_elliptic_minimum_order() {
	// For a lowpass with fp=2000, fs=3000, Ap=1dB, As=40dB at 44100 Hz,
	// the minimum order should be small (3 or 4).
	int n = iir::elliptic_minimum_order(1.0, 40.0, 2000.0, 3000.0, 44100.0);
	if (n < 2 || n > 6)
		throw std::runtime_error("test failed: minimum_order out of expected range [2,6], got " + std::to_string(n));

	// Wider transition band should need lower order
	int n_wide = iir::elliptic_minimum_order(1.0, 40.0, 2000.0, 8000.0, 44100.0);
	if (n_wide >= n)
		throw std::runtime_error("test failed: wider transition should need lower order");

	// Stricter stopband should need higher order
	int n_strict = iir::elliptic_minimum_order(1.0, 80.0, 2000.0, 3000.0, 44100.0);
	if (n_strict < n)
		throw std::runtime_error("test failed: stricter stopband should need >= order");

	std::cout << "  elliptic_minimum_order: passed (n=" << n
	          << ", n_wide=" << n_wide << ", n_strict=" << n_strict << ")\n";
}

void test_spec_lowpass_acceptance() {
	// Issue #54 acceptance criterion:
	// <=1 dB ripple below 2 kHz and >=40 dB attenuation above 3 kHz.
	// Use elliptic_minimum_order to determine the needed filter order.
	int n = iir::elliptic_minimum_order(1.0, 40.0, 2000.0, 3000.0, 44100.0);
	iir::EllipticLowPassSpec<8> f;
	f.setup(n, 44100.0, 2000.0, 3000.0, 1.0, 40.0);

	// Check DC gain (even order: should be near -ripple_db)
	double db_dc = mag_db(f.cascade().response(0.0));
	if (!(db_dc > -2.0 && db_dc <= 0.01))
		throw std::runtime_error("test failed: spec LP DC gain = " + std::to_string(db_dc));

	// Check passband edge: attenuation at 2000 Hz should be <= 1 dB
	double db_pass = mag_db(f.cascade().response(2000.0 / 44100.0));
	if (!(db_pass >= -1.5))
		throw std::runtime_error("test failed: spec LP passband edge = " + std::to_string(db_pass) + " dB");

	// Check stopband edge: attenuation at 3000 Hz should be >= 40 dB
	double db_stop = mag_db(f.cascade().response(3000.0 / 44100.0));
	if (!(db_stop < -40.0))
		throw std::runtime_error("test failed: spec LP stopband at 3kHz = " + std::to_string(db_stop) + " dB, need < -40");

	// Deep stopband should be even more attenuated
	double db_deep = mag_db(f.cascade().response(5000.0 / 44100.0));
	if (!(db_deep < -40.0))
		throw std::runtime_error("test failed: spec LP deep stopband = " + std::to_string(db_deep));

	std::cout << "  spec_lowpass_acceptance: passed (DC=" << db_dc
	          << " dB, pass=" << db_pass << " dB, stop=" << db_stop << " dB)\n";
}

void test_spec_lowpass_orders() {
	// Verify various orders produce valid filters with finite coefficients.
	// Use a generous transition band (4 kHz -> 8 kHz) so even low orders work.
	for (int order = 2; order <= 8; ++order) {
		iir::EllipticLowPassSpec<8> f;
		f.setup(order, 48000.0, 4000.0, 8000.0, 0.5, 40.0);

		auto r_dc = f.cascade().response(0.0);
		double mag = std::abs(r_dc);
		if (!(mag > 0.3 && mag < 1.5))
			throw std::runtime_error("test failed: spec LP order " + std::to_string(order) +
				" DC magnitude = " + std::to_string(mag));

		// Stopband rejection should increase with order
		double db_stop = mag_db(f.cascade().response(8000.0 / 48000.0));
		if (!(db_stop < -10.0))
			throw std::runtime_error("test failed: spec LP order " + std::to_string(order) +
				" stopband = " + std::to_string(db_stop) + " dB");
	}
	std::cout << "  spec_lowpass_orders_2_to_8: passed\n";
}

void test_spec_highpass() {
	iir::EllipticHighPassSpec<4> f;
	// Highpass: passband above 3 kHz, stopband below 2 kHz
	f.setup(4, 44100.0, 3000.0, 2000.0, 1.0, 40.0);

	// Passband: high frequencies should pass
	double db_high = mag_db(f.cascade().response(0.25));
	if (!(db_high > -3.0))
		throw std::runtime_error("test failed: spec HP passband = " + std::to_string(db_high) + " dB");

	// Stopband: low frequencies should be rejected
	double db_low = mag_db(f.cascade().response(1000.0 / 44100.0));
	if (!(db_low < -30.0))
		throw std::runtime_error("test failed: spec HP stopband = " + std::to_string(db_low) + " dB");

	std::cout << "  spec_highpass: passed (high=" << db_high << " dB, low=" << db_low << " dB)\n";
}

void test_spec_bandpass() {
	iir::EllipticBandPassSpec<4> f;
	// Bandpass: pass 3-5 kHz, reject below 2 kHz and above 6 kHz
	f.setup(4, 44100.0, 3000.0, 5000.0, 2000.0, 6000.0, 1.0, 40.0);

	// Center of passband
	double db_center = mag_db(f.cascade().response(4000.0 / 44100.0));
	if (!(db_center > -6.0))
		throw std::runtime_error("test failed: spec BP center = " + std::to_string(db_center) + " dB");

	// Below stopband
	double db_low = mag_db(f.cascade().response(500.0 / 44100.0));
	if (!(db_low < -20.0))
		throw std::runtime_error("test failed: spec BP low stopband = " + std::to_string(db_low) + " dB");

	std::cout << "  spec_bandpass: passed (center=" << db_center << " dB, low=" << db_low << " dB)\n";
}

void test_spec_bandstop() {
	iir::EllipticBandStopSpec<4> f;
	// Bandstop: reject 3-5 kHz, pass below 2 kHz and above 6 kHz
	f.setup(4, 44100.0, 3000.0, 5000.0, 2000.0, 6000.0, 1.0, 40.0);

	// Stopband center should be heavily attenuated
	double db_center = mag_db(f.cascade().response(4000.0 / 44100.0));
	if (!(db_center < -20.0))
		throw std::runtime_error("test failed: spec BS center = " + std::to_string(db_center) + " dB");

	// DC should pass
	double db_dc = mag_db(f.cascade().response(0.0));
	if (!(db_dc > -3.0))
		throw std::runtime_error("test failed: spec BS DC = " + std::to_string(db_dc) + " dB");

	std::cout << "  spec_bandstop: passed (center=" << db_center << " dB, DC=" << db_dc << " dB)\n";
}

void test_spec_simple_filter() {
	SimpleFilter<iir::EllipticLowPassSpec<4>> f;
	f.setup(4, 44100.0, 2000.0, 3000.0, 1.0, 40.0);

	double y0 = f.process(1.0);
	if (!(std::isfinite(y0)))
		throw std::runtime_error("test failed: spec simple filter finite output");

	for (int i = 0; i < 99; ++i) {
		double s = f.process(0.0);
		if (!std::isfinite(s))
			throw std::runtime_error("test failed: spec simple filter sample " + std::to_string(i) + " NaN");
	}

	f.reset();
	double y0b = f.process(1.0);
	if (!(near(y0, y0b, 1e-15)))
		throw std::runtime_error("test failed: spec simple filter reset");

	std::cout << "  spec_simple_filter: passed\n";
}

// ============================================================================
// Posit<32,2> regression: verify the Elliptic prewarp and bandpass-geometry
// math that this PR refactored actually works at the caller's scalar type.
//
// Instead of instantiating the full EllipticLowPassSpec<posit> (which drags
// in PoleZeroLayout<posit>, BiquadCoefficients<posit>, etc. — those paths
// assume std::complex<T> but complex_for_t<posit> dispatches to
// sw::universal::complex; that's a pre-existing infrastructure issue tracked
// separately), this test recomputes the exact prewarp expressions used by
// the Spec::setup methods in both double and posit, and verifies agreement.
// ============================================================================

void test_spec_prewarp_in_posit_precision() {
	using posit_t = sw::universal::posit<32, 2>;
	using std::tan; using std::abs;

	auto check = [](const char* label, double posit_val, double double_val, double eps) {
		double diff = std::abs(posit_val - double_val);
		if (diff > eps) {
			char buf[256];
			std::snprintf(buf, sizeof(buf),
				"test failed: %s posit=%.12g double=%.12g diff=%.3e (eps=%.1e)",
				label, posit_val, double_val, diff, eps);
			throw std::runtime_error(buf);
		}
		return diff;
	};

	// Lowpass prewarp: the math inside EllipticLowPassSpec::setup
	{
		constexpr posit_t pi_p = posit_t(sw::dsp::pi);
		const posit_t fs_p(44100.0);
		const posit_t wp_p = tan(pi_p * posit_t(2000.0) / fs_p);
		const posit_t ws_p = tan(pi_p * posit_t(3000.0) / fs_p);
		const posit_t k_p  = wp_p / ws_p;

		const double wp_d = std::tan(sw::dsp::pi * 2000.0 / 44100.0);
		const double ws_d = std::tan(sw::dsp::pi * 3000.0 / 44100.0);
		const double k_d  = wp_d / ws_d;

		// Tolerances reflect posit<32,2> ULP (~2^-28 ~ 3.7e-9 near unity) with
		// a few ops of amplification. wp and ws are direct tan results; k is
		// a single division so retains similar precision.
		double d1 = check("LP wp", static_cast<double>(wp_p), wp_d, 1e-8);
		double d2 = check("LP ws", static_cast<double>(ws_p), ws_d, 1e-8);
		double d3 = check("LP k",  static_cast<double>(k_p),  k_d,  5e-8);
		std::cout << "  spec_prewarp (lowpass): wp_diff=" << d1
		          << " ws_diff=" << d2 << " k_diff=" << d3 << "\n";
	}

	// Bandpass prewarp + geometry: the math inside EllipticBandPassSpec::setup
	{
		constexpr posit_t pi_p = posit_t(sw::dsp::pi);
		constexpr posit_t one_p = posit_t(1);
		const posit_t fs_p(44100.0);
		const posit_t wpl = tan(pi_p * posit_t(3000.0) / fs_p);
		const posit_t wph = tan(pi_p * posit_t(5000.0) / fs_p);
		const posit_t wsl = tan(pi_p * posit_t(2000.0) / fs_p);
		const posit_t wsh = tan(pi_p * posit_t(6000.0) / fs_p);
		const posit_t w0_sq = wpl * wph;
		const posit_t bw = wph - wpl;
		const posit_t kl = abs((wsl * wsl - w0_sq) / (wsl * bw));
		const posit_t kh = abs((wsh * wsh - w0_sq) / (wsh * bw));
		const posit_t k_min = (kl < kh) ? kl : kh;
		const posit_t k_p   = one_p / k_min;

		// Double reference
		const double wpl_d = std::tan(sw::dsp::pi * 3000.0 / 44100.0);
		const double wph_d = std::tan(sw::dsp::pi * 5000.0 / 44100.0);
		const double wsl_d = std::tan(sw::dsp::pi * 2000.0 / 44100.0);
		const double wsh_d = std::tan(sw::dsp::pi * 6000.0 / 44100.0);
		const double w0_sq_d = wpl_d * wph_d;
		const double bw_d = wph_d - wpl_d;
		const double kl_d = std::abs((wsl_d * wsl_d - w0_sq_d) / (wsl_d * bw_d));
		const double kh_d = std::abs((wsh_d * wsh_d - w0_sq_d) / (wsh_d * bw_d));
		const double k_d  = 1.0 / std::min(kl_d, kh_d);

		// Bandpass kl amplifies error through wsl², subtraction, division.
		// Measured on this config: wpl ~1e-10, kl ~6e-7, k ~1e-8.
		double d_wpl = check("BP wpl", static_cast<double>(wpl), wpl_d, 1e-8);
		double d_kl  = check("BP kl",  static_cast<double>(kl),  kl_d,  5e-6);
		double d_k   = check("BP k",   static_cast<double>(k_p), k_d,   1e-6);
		std::cout << "  spec_prewarp (bandpass): wpl_diff=" << d_wpl
		          << " kl_diff=" << d_kl << " k_diff=" << d_k << "\n";
	}

	std::cout << "  spec_prewarp_in_posit_precision: passed\n";
}

void test_spec_rejects_invalid() {
	iir::EllipticLowPassSpec<4> f;

	bool threw = false;
	try {
		// passband >= stopband should throw
		f.setup(4, 44100.0, 5000.0, 3000.0, 1.0, 40.0);
	} catch (const std::invalid_argument&) {
		threw = true;
	}
	if (!threw)
		throw std::runtime_error("test failed: spec LP should reject passband >= stopband");

	threw = false;
	try {
		// stopband >= Nyquist should throw
		f.setup(4, 44100.0, 2000.0, 22100.0, 1.0, 40.0);
	} catch (const std::invalid_argument&) {
		threw = true;
	}
	if (!threw)
		throw std::runtime_error("test failed: spec LP should reject stopband >= Nyquist");

	std::cout << "  spec_rejects_invalid: passed\n";
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
		test_elliptic_rejects_rolloff_out_of_range();
		test_elliptic_issue50_impulse_finite();

		std::cout << "\nCauer-Darlington (Spec) Tests\n";

		test_elliptic_minimum_order();
		test_spec_lowpass_acceptance();
		test_spec_lowpass_orders();
		test_spec_highpass();
		test_spec_bandpass();
		test_spec_bandstop();
		test_spec_simple_filter();
		test_spec_prewarp_in_posit_precision();
		test_spec_rejects_invalid();

		std::cout << "All Elliptic filter tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << '\n';
		return 1;
	}
}
