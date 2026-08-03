// test_transfer_function_pole_zero.cpp: pole/zero extraction tests.
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)`.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sw/dsp/filter/iir/elliptic.hpp>
#include <sw/dsp/filter/layout/layout.hpp>
#include <sw/dsp/transfer_function/pole_zero.hpp>

using sw::dsp::transfer_function::PoleZeroPlot;
using sw::dsp::transfer_function::butterworth_prototype;
using sw::dsp::transfer_function::chebyshev1_prototype;
using sw::dsp::transfer_function::chebyshev2_prototype;
using sw::dsp::transfer_function::bessel_prototype;
using sw::dsp::transfer_function::elliptic_prototype;
using sw::dsp::transfer_function::lp_to_hp;
using sw::dsp::transfer_function::lp_to_bp;
using sw::dsp::transfer_function::lp_to_bs;
using sw::dsp::transfer_function::apply_bilinear;

static const double pi_d = std::numbers::pi_v<double>;

// ---------------------------------------------------------------------------
// Butterworth: N poles on left half unit circle, scaled by omega_c.
//   Order N, cutoff omega_c: poles at omega_c * exp(j*pi*(2k+N-1)/(2N))
//   All poles have |s| = omega_c (radial symmetry).
// ---------------------------------------------------------------------------
static void test_butterworth_pole_radius() {
	const double cutoff = 1000.0;
	const double omega_c = 2.0 * pi_d * cutoff;
	for (int N : {2, 3, 4, 5, 8}) {
		auto p = butterworth_prototype(N, cutoff);
		if (static_cast<int>(p.s_poles.size()) != N)
			throw std::runtime_error("butterworth: wrong pole count");
		if (!p.s_zeros.empty())
			throw std::runtime_error("butterworth: should have no zeros");
		for (const auto& s : p.s_poles) {
			// LHP (real < 0) for stability.
			if (s.real() > 1e-9)
				throw std::runtime_error("butterworth: pole in RHP");
			// |s| = omega_c
			if (std::abs(std::abs(s) - omega_c) > 1e-6 * omega_c)
				throw std::runtime_error("butterworth: pole not on radius omega_c");
		}
	}
}

// ---------------------------------------------------------------------------
// Chebyshev I: N poles on an ellipse in LHP; conjugate symmetry.
// ---------------------------------------------------------------------------
static void test_chebyshev1_pole_ellipse() {
	auto p = chebyshev1_prototype(4, 1000.0, /*ripple_dB=*/1.0);
	if (p.s_poles.size() != 4)
		throw std::runtime_error("chebyshev1: wrong pole count");
	// All poles in LHP.
	for (const auto& s : p.s_poles) {
		if (s.real() > 1e-9)
			throw std::runtime_error("chebyshev1: pole in RHP");
	}
	// Conjugate symmetry check: for every pole (a, b), (a, -b) also present.
	for (const auto& s : p.s_poles) {
		bool found_conj = false;
		for (const auto& t : p.s_poles) {
			if (std::abs(t.real() - s.real()) < 1e-9
			 && std::abs(t.imag() + s.imag()) < 1e-9) {
				found_conj = true; break;
			}
		}
		if (!found_conj)
			throw std::runtime_error("chebyshev1: missing conjugate pair");
	}
}

// ---------------------------------------------------------------------------
// Chebyshev II: finite zeros on the imaginary axis; number = 2 * floor(N/2).
// ---------------------------------------------------------------------------
static void test_chebyshev2_finite_zeros() {
	auto p = chebyshev2_prototype(4, 1000.0, /*stopband_dB=*/40.0);
	if (p.s_poles.size() != 4)
		throw std::runtime_error("chebyshev2: wrong pole count");
	// Order 4 -> 2 * (4/2) = 4 finite zeros.
	if (p.s_zeros.size() != 4)
		throw std::runtime_error("chebyshev2: wrong zero count for even order");
	// All zeros purely imaginary.
	for (const auto& z : p.s_zeros) {
		if (std::abs(z.real()) > 1e-9)
			throw std::runtime_error("chebyshev2: zero not purely imaginary");
	}
	// Odd order -> 2*(3/2) = 2 finite zeros.
	auto p3 = chebyshev2_prototype(3, 1000.0, 40.0);
	if (p3.s_zeros.size() != 2)
		throw std::runtime_error("chebyshev2: wrong zero count for odd order");
}

// ---------------------------------------------------------------------------
// Bessel: all poles in LHP.
// ---------------------------------------------------------------------------
static void test_bessel_lhp() {
	for (int N : {2, 3, 4, 5, 6}) {
		auto p = bessel_prototype(N, 1000.0);
		if (static_cast<int>(p.s_poles.size()) != N)
			throw std::runtime_error("bessel: wrong pole count");
		for (const auto& s : p.s_poles) {
			if (s.real() > 1e-9)
				throw std::runtime_error("bessel: pole in RHP");
		}
		// For odd orders one real pole; conjugate pairs for the rest.
		int real_poles = 0;
		for (const auto& s : p.s_poles) {
			if (std::abs(s.imag()) < 1e-6 * std::abs(s.real()) + 1e-9)
				++real_poles;
		}
		const int expected_real = (N & 1) ? 1 : 0;
		if (real_poles != expected_real)
			throw std::runtime_error("bessel: unexpected real-pole count");
	}
}

// ---------------------------------------------------------------------------
// Elliptic: N poles all in LHP; 2*floor(N/2) finite zeros on the
// imaginary axis. Cross-check that extracted poles/zeros match what
// iir::EllipticAnalogPrototype computes directly for the same params.
// ---------------------------------------------------------------------------
static void test_elliptic_pole_zero() {
	for (int N : {3, 4, 5, 6}) {
		const double cutoff = 1000.0;
		const double ripple = 0.5;
		const double k_sel  = 0.5;
		auto p = elliptic_prototype(N, cutoff, ripple, k_sel);
		if (static_cast<int>(p.s_poles.size()) != N)
			throw std::runtime_error("elliptic: wrong pole count");
		for (const auto& s : p.s_poles) {
			if (s.real() > 1e-6 * std::abs(s))
				throw std::runtime_error("elliptic: pole in RHP");
		}
		// Finite zeros: exactly 2 * floor(N/2). Odd-N designs have one
		// zero at infinity which is omitted from the s_zeros list.
		const int expected_zeros = 2 * (N / 2);
		if (static_cast<int>(p.s_zeros.size()) != expected_zeros)
			throw std::runtime_error(
				"elliptic: unexpected finite-zero count");
		// All finite zeros purely imaginary.
		for (const auto& z : p.s_zeros) {
			if (std::abs(z.real()) > 1e-6 * std::abs(z))
				throw std::runtime_error(
					"elliptic: zero not on imaginary axis");
		}
	}
}

// Cross-check: the poles extracted by elliptic_prototype should match
// (mod ordering + omega_c scaling) the poles the library's elliptic
// filter design produces from the same inputs.
static void test_elliptic_matches_filter_design() {
	constexpr int N = 4;
	const double ripple = 0.5;
	const double k_sel  = 0.5;
	const double cutoff = 1000.0;
	auto p = elliptic_prototype(N, cutoff, ripple, k_sel);

	// Reference: run the library elliptic-prototype directly.
	sw::dsp::iir::EllipticAnalogPrototype<double, 12> proto;
	sw::dsp::PoleZeroLayout<double, 12> ref;
	proto.design_from_modulus(N, ripple, k_sel, ref);

	const double omega_c = 2.0 * pi_d * cutoff;
	// Gather reference s-plane poles.
	std::vector<std::complex<double>> ref_poles;
	for (int i = 0; i < ref.num_pairs(); ++i) {
		const auto& pair = ref[i];
		ref_poles.emplace_back(pair.poles.first.real()  * omega_c,
		                        pair.poles.first.imag() * omega_c);
		if (!pair.is_single_pole()) {
			ref_poles.emplace_back(pair.poles.second.real() * omega_c,
			                        pair.poles.second.imag() * omega_c);
		}
	}
	if (ref_poles.size() != p.s_poles.size())
		throw std::runtime_error("elliptic cross-check: pole count mismatch");

	// For each extracted pole, verify a reference pole exists within
	// numerical tolerance (order-independent match).
	for (const auto& s : p.s_poles) {
		bool found = false;
		for (const auto& r : ref_poles) {
			if (std::abs(s - r) < 1e-9 * std::max(std::abs(s), 1.0)) {
				found = true; break;
			}
		}
		if (!found)
			throw std::runtime_error(
				"elliptic cross-check: extracted pole not in reference set");
	}
}

// Validation: signature preconditions.
static void test_elliptic_input_validation() {
	bool threw;
	// selectivity_k out of range.
	threw = false;
	try { elliptic_prototype(4, 1000.0, 0.5, 0.0); }
	catch (const std::exception&) { threw = true; }
	if (!threw) throw std::runtime_error("elliptic: expected throw on k<=0");

	threw = false;
	try { elliptic_prototype(4, 1000.0, 0.5, 1.0); }
	catch (const std::exception&) { threw = true; }
	if (!threw) throw std::runtime_error("elliptic: expected throw on k>=1");

	// order too high.
	threw = false;
	try { elliptic_prototype(20, 1000.0, 0.5, 0.5); }
	catch (const std::exception&) { threw = true; }
	if (!threw) throw std::runtime_error("elliptic: expected throw on order>12");

	// ripple_dB non-positive.
	threw = false;
	try { elliptic_prototype(4, 1000.0, 0.0, 0.5); }
	catch (const std::exception&) { threw = true; }
	if (!threw) throw std::runtime_error("elliptic: expected throw on ripple=0");
}

// ---------------------------------------------------------------------------
// LP -> HP: LP prototype with no finite zeros becomes HP with `order`
// zeros at s = 0. LHP poles stay in LHP.
// ---------------------------------------------------------------------------
static void test_lp_to_hp() {
	auto p = butterworth_prototype(4, 1000.0);
	lp_to_hp(p, 500.0);
	if (p.s_zeros.size() != 4)
		throw std::runtime_error("lp_to_hp: expected 4 zeros at s=0");
	for (const auto& z : p.s_zeros) {
		if (std::abs(z) > 1e-9)
			throw std::runtime_error("lp_to_hp: zero not at origin");
	}
	if (p.kind != "highpass")
		throw std::runtime_error("lp_to_hp: kind not updated");
	// Poles still in LHP.
	for (const auto& s : p.s_poles) {
		if (s.real() > 1e-6 * std::abs(s))
			throw std::runtime_error("lp_to_hp: pole left LHP");
	}
}

// ---------------------------------------------------------------------------
// LP -> BP: LP order N produces 2N poles and 2N zeros total.
// ---------------------------------------------------------------------------
static void test_lp_to_bp() {
	auto p = butterworth_prototype(4, 1000.0);
	lp_to_bp(p, 500.0, 2000.0);
	if (p.s_poles.size() != 8)
		throw std::runtime_error("lp_to_bp: expected 2N poles");
	if (p.s_zeros.size() != 8)
		throw std::runtime_error("lp_to_bp: expected 2N zeros");
	// BP poles still in LHP (stability preserved).
	for (const auto& s : p.s_poles) {
		if (s.real() > 1e-6 * std::abs(s))
			throw std::runtime_error("lp_to_bp: pole left LHP");
	}
	if (p.kind != "bandpass")
		throw std::runtime_error("lp_to_bp: kind not updated");
}

// ---------------------------------------------------------------------------
// Bilinear: s-plane LHP poles map to inside the unit circle. |z_pole| < 1.
// ---------------------------------------------------------------------------
static void test_bilinear_maps_lhp_inside_unit_circle() {
	auto p = butterworth_prototype(4, 1000.0);
	apply_bilinear(p, /*fs=*/48000.0);
	if (p.z_poles.size() != 4)
		throw std::runtime_error("bilinear: z_pole count wrong");
	for (const auto& z : p.z_poles) {
		if (std::abs(z) >= 1.0 - 1e-9)
			throw std::runtime_error("bilinear: z_pole outside unit circle");
	}
	// Butterworth s-plane has no finite zeros -> bilinear puts zeros at z = -1.
	if (p.z_zeros.size() != 4)
		throw std::runtime_error("bilinear: z_zero count wrong");
	for (const auto& z : p.z_zeros) {
		if (std::abs(z.real() + 1.0) > 1e-9 || std::abs(z.imag()) > 1e-9)
			throw std::runtime_error("bilinear: z_zero not at z=-1");
	}
	if (p.sample_rate_hz != 48000.0)
		throw std::runtime_error("bilinear: sample_rate not stored");
}

// ---------------------------------------------------------------------------
// JSON dump round-trip: file exists, header + fields present.
// ---------------------------------------------------------------------------
static void test_dump_json() {
	auto p = butterworth_prototype(4, 1000.0);
	apply_bilinear(p, 48000.0);
	const std::string path = "/tmp/_test_pole_zero.json";
	p.dump_json(path);
	std::ifstream in(path);
	if (!in) throw std::runtime_error("dump_json: file not created");
	std::stringstream buf; buf << in.rdbuf();
	const std::string j = buf.str();
	for (const auto& needle :
	     {"butterworth", "order", "s_poles", "z_poles", "sample_rate_hz"}) {
		if (j.find(needle) == std::string::npos)
			throw std::runtime_error(
				std::string("dump_json: field missing: ") + needle);
	}
	std::remove(path.c_str());
}

int main() {
	try {
		std::cout << "test_transfer_function_pole_zero\n";
		test_butterworth_pole_radius();          std::cout << "  butterworth_radius       PASS\n";
		test_chebyshev1_pole_ellipse();          std::cout << "  chebyshev1_ellipse       PASS\n";
		test_chebyshev2_finite_zeros();          std::cout << "  chebyshev2_finite_zeros  PASS\n";
		test_bessel_lhp();                       std::cout << "  bessel_lhp               PASS\n";
		test_elliptic_pole_zero();               std::cout << "  elliptic_pole_zero       PASS\n";
		test_elliptic_matches_filter_design();   std::cout << "  elliptic_matches_filter  PASS\n";
		test_elliptic_input_validation();        std::cout << "  elliptic_input_validation PASS\n";
		test_lp_to_hp();                         std::cout << "  lp_to_hp                 PASS\n";
		test_lp_to_bp();                         std::cout << "  lp_to_bp                 PASS\n";
		test_bilinear_maps_lhp_inside_unit_circle();
		std::cout << "  bilinear_lhp_inside_unit PASS\n";
		test_dump_json();                        std::cout << "  dump_json                PASS\n";
		std::cout << "OK\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "FAIL: " << ex.what() << "\n";
		return 1;
	}
}
