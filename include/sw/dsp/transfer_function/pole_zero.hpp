#pragma once
// pole_zero.hpp: Analytical pole/zero extraction from analog-prototype
// filter designs, threaded through Constantinides transformations
// (LP -> HP/BP/BS) and the bilinear transform (s-plane -> z-plane).
//
// Complements transfer_function/bode.hpp: sweep_bode() works on ANY
// LTI block by probe-and-measure; pole_zero here works only on filters
// whose analog prototype we know analytically, but delivers the exact
// closed-form pole/zero locations without measurement noise.
//
// Supported prototype families:
//   butterworth_prototype        - closed form
//   chebyshev1_prototype         - closed form (ellipse poles)
//   chebyshev2_prototype         - closed form (inverse ellipse + finite zeros)
//   bessel_prototype             - roots of reverse Bessel polynomial
//                                  (via Laguerre root finder)
//   elliptic_prototype           - THROWS: not yet implemented in this
//                                  header. Requires Jacobi sn/cn/dn
//                                  which are already used inside
//                                  filter/iir/elliptic.hpp but not
//                                  broken out as a reusable API.
//                                  Tracked as a follow-up.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <complex>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>
#include <sw/dsp/math/root_finder.hpp>

namespace sw::dsp::transfer_function {

// ============================================================================
// PoleZeroPlot - result carrier for the closed-form pipeline
// ============================================================================

struct PoleZeroPlot {
	// Design metadata (for the JSON sidecar).
	std::string design;             // "butterworth", "chebyshev1", ...
	int         order = 0;
	std::string kind = "lowpass";   // "lowpass", "highpass", "bandpass", "bandstop"

	// s-plane (continuous-time) prototype poles + zeros.
	std::vector<std::complex<double>> s_poles;
	std::vector<std::complex<double>> s_zeros;

	// z-plane (discrete-time) locations after bilinear transform.
	// Empty until apply_bilinear() is called.
	std::vector<std::complex<double>> z_poles;
	std::vector<std::complex<double>> z_zeros;

	// The angular cutoff (rad/s) the prototype was designed at. LP
	// prototypes canonically use omega_c = 1; bandpass/bandstop
	// intermediates carry through the low/high edge frequencies for
	// the JSON viewer.
	double cutoff_hz = 1.0;
	double low_hz  = 0.0;   // used by BP/BS
	double high_hz = 0.0;   // used by BP/BS
	// Discrete-time sample rate (populated by apply_bilinear).
	double sample_rate_hz = 0.0;

	// Ripple / stopband parameters (populated by prototype builders
	// that use them; ignored by families that don't).
	double ripple_dB   = 0.0;
	double stopband_dB = 0.0;

	// Dump a JSON representation of the plot for the mp-dsp-python
	// pole/zero viewer. Complex numbers are written as [re, im] arrays.
	void dump_json(const std::string& path) const {
		std::ofstream out(path);
		if (!out) throw std::runtime_error(
			"PoleZeroPlot::dump_json: cannot open " + path);
		out << std::setprecision(17);
		auto emit = [&](const char* name,
		                 const std::vector<std::complex<double>>& v) {
			out << "  \"" << name << "\": [";
			for (std::size_t i = 0; i < v.size(); ++i) {
				out << "[" << v[i].real() << ", " << v[i].imag() << "]";
				if (i + 1 < v.size()) out << ", ";
			}
			out << "]";
		};
		out << "{\n"
		    << "  \"design\": \"" << design << "\",\n"
		    << "  \"order\": " << order << ",\n"
		    << "  \"kind\": \"" << kind << "\",\n"
		    << "  \"cutoff_hz\": " << cutoff_hz << ",\n"
		    << "  \"low_hz\": " << low_hz << ",\n"
		    << "  \"high_hz\": " << high_hz << ",\n"
		    << "  \"sample_rate_hz\": " << sample_rate_hz << ",\n"
		    << "  \"ripple_dB\": " << ripple_dB << ",\n"
		    << "  \"stopband_dB\": " << stopband_dB << ",\n";
		emit("s_poles", s_poles); out << ",\n";
		emit("s_zeros", s_zeros); out << ",\n";
		emit("z_poles", z_poles); out << ",\n";
		emit("z_zeros", z_zeros); out << "\n"
		    << "}\n";
	}
};

// ============================================================================
// Prototype builders - all return an LP prototype at cutoff_hz
// ============================================================================

// Butterworth LP: N poles equally spaced on the left half of the unit
// circle in the s-plane, scaled by 2*pi*cutoff_hz. No finite zeros.
inline PoleZeroPlot butterworth_prototype(int order, double cutoff_hz) {
	if (order < 1)
		throw std::invalid_argument("butterworth_prototype: order must be >= 1");
	PoleZeroPlot p;
	p.design = "butterworth";
	p.order = order;
	p.cutoff_hz = cutoff_hz;
	const double omega_c = 2.0 * std::numbers::pi_v<double> * cutoff_hz;
	p.s_poles.reserve(order);
	for (int k = 1; k <= order; ++k) {
		const double theta = std::numbers::pi_v<double>
			* (2.0 * k + order - 1.0) / (2.0 * order);
		p.s_poles.emplace_back(omega_c * std::cos(theta),
		                        omega_c * std::sin(theta));
	}
	return p;
}

// Chebyshev I LP: N poles on an ellipse in the s-plane. `ripple_dB` is
// the peak passband ripple (positive value).
inline PoleZeroPlot chebyshev1_prototype(int order, double cutoff_hz,
                                          double ripple_dB) {
	if (order < 1)
		throw std::invalid_argument("chebyshev1_prototype: order must be >= 1");
	if (!(ripple_dB > 0.0))
		throw std::invalid_argument(
			"chebyshev1_prototype: ripple_dB must be > 0");
	PoleZeroPlot p;
	p.design = "chebyshev1";
	p.order = order;
	p.cutoff_hz = cutoff_hz;
	p.ripple_dB = ripple_dB;

	const double omega_c = 2.0 * std::numbers::pi_v<double> * cutoff_hz;
	// epsilon = sqrt(10^(Rp/10) - 1)
	const double eps = std::sqrt(std::pow(10.0, ripple_dB / 10.0) - 1.0);
	// mu = (1/N) * asinh(1/eps)
	const double mu = std::asinh(1.0 / eps) / static_cast<double>(order);

	p.s_poles.reserve(order);
	for (int k = 1; k <= order; ++k) {
		const double theta = std::numbers::pi_v<double>
			* (2.0 * k - 1.0) / (2.0 * order);
		const double re = -std::sinh(mu) * std::sin(theta) * omega_c;
		const double im =  std::cosh(mu) * std::cos(theta) * omega_c;
		p.s_poles.emplace_back(re, im);
	}
	return p;
}

// Chebyshev II (inverse Chebyshev) LP: has both finite poles and
// finite zeros on the imaginary axis. `stopband_dB` is the minimum
// stopband attenuation (positive value; typical range 40-80 dB).
inline PoleZeroPlot chebyshev2_prototype(int order, double cutoff_hz,
                                          double stopband_dB) {
	if (order < 1)
		throw std::invalid_argument("chebyshev2_prototype: order must be >= 1");
	if (!(stopband_dB > 0.0))
		throw std::invalid_argument(
			"chebyshev2_prototype: stopband_dB must be > 0");
	PoleZeroPlot p;
	p.design = "chebyshev2";
	p.order = order;
	p.cutoff_hz = cutoff_hz;
	p.stopband_dB = stopband_dB;

	const double omega_c = 2.0 * std::numbers::pi_v<double> * cutoff_hz;
	const double eps = 1.0 / std::sqrt(std::pow(10.0, stopband_dB / 10.0) - 1.0);
	const double mu  = std::asinh(1.0 / eps) / static_cast<double>(order);

	// Chebyshev II poles: reciprocals of Chebyshev I poles (unnormalized),
	// then scaled back to omega_c cutoff.
	p.s_poles.reserve(order);
	for (int k = 1; k <= order; ++k) {
		const double theta = std::numbers::pi_v<double>
			* (2.0 * k - 1.0) / (2.0 * order);
		const std::complex<double> ch1_pole(
			-std::sinh(mu) * std::sin(theta),
			 std::cosh(mu) * std::cos(theta));
		// s_pole = omega_c / ch1_pole (reciprocal, scaled)
		const std::complex<double> pole = omega_c
			/ std::complex<double>(ch1_pole.real(), ch1_pole.imag());
		p.s_poles.push_back(pole);
	}
	// Chebyshev II finite zeros: on the imaginary axis at
	// omega_c / cos((2k-1)*pi/(2N)) for k=1..floor(N/2), each as
	// +/- pair. If N is odd, one zero is at infinity (omitted).
	const int nz = order / 2;
	p.s_zeros.reserve(2 * nz);
	for (int k = 1; k <= nz; ++k) {
		const double theta = std::numbers::pi_v<double>
			* (2.0 * k - 1.0) / (2.0 * order);
		const double omega_z = omega_c / std::cos(theta);
		p.s_zeros.emplace_back(0.0,  omega_z);
		p.s_zeros.emplace_back(0.0, -omega_z);
	}
	return p;
}

// Bessel LP: poles are the left-half-plane roots of the reverse
// Bessel polynomial theta_N(s), scaled to give the desired -3 dB
// cutoff (approximately - Bessel's cutoff is a weak-slope thing).
// Uses Laguerre's method via the library's RootFinder.
inline PoleZeroPlot bessel_prototype(int order, double cutoff_hz) {
	if (order < 1)
		throw std::invalid_argument("bessel_prototype: order must be >= 1");
	if (order > 12)
		throw std::invalid_argument(
			"bessel_prototype: order > 12 exceeds fixed root-finder capacity");
	PoleZeroPlot p;
	p.design = "bessel";
	p.order = order;
	p.cutoff_hz = cutoff_hz;

	// Reverse Bessel polynomial coefficients via recursion:
	//   theta_0(s) = 1
	//   theta_1(s) = s + 1
	//   theta_n(s) = (2n-1) * theta_{n-1}(s) + s^2 * theta_{n-2}(s)
	// theta_n has degree n. We store coefficients low-to-high.
	std::vector<double> prev{1.0};                // theta_0
	std::vector<double> curr{1.0, 1.0};           // theta_1
	if (order == 1) {
		// Single real pole at s = -1 (before scaling).
	}
	for (int n = 2; n <= order; ++n) {
		std::vector<double> next(n + 1, 0.0);
		// (2n-1) * theta_{n-1}
		for (std::size_t i = 0; i < curr.size(); ++i)
			next[i] += (2.0 * n - 1.0) * curr[i];
		// s^2 * theta_{n-2} - shift prev up by 2
		for (std::size_t i = 0; i < prev.size(); ++i)
			next[i + 2] += prev[i];
		prev = std::move(curr);
		curr = std::move(next);
	}
	// curr[k] = coefficient of s^k.
	// Delay-normalize: divide all coefficients by curr[0] so
	// theta_N(0) = 1. Then find complex roots via Laguerre.
	const double c0 = curr[0];
	for (auto& c : curr) c /= c0;

	// RootFinder<T, MaxDegree>: hand it the polynomial coefficients
	// low-to-high. MaxDegree=12 covers all supported orders.
	sw::dsp::RootFinder<double, 12> rf;
	// The RootFinder API expects coeffs to be assigned via .coef(i);
	// let's inline the equivalent by feeding raw polynomial evaluation.
	// Actually - the library expects raw indexing; grep confirmed
	// .root(i) accessors. Feed via a compatible interface:
	for (int i = 0; i <= order; ++i) rf.coef(i) = std::complex<double>(curr[i], 0.0);
	rf.solve(order);

	// Bessel roots come out at "delay-normalized" positions (delay=1
	// at DC). Scale by omega_c so the -3 dB point is roughly cutoff.
	// Standard Bessel scaling constants (Bessel/Thomson):
	//   the -3 dB frequency of the delay-normalized filter is
	//   sqrt((2N+1) * ln(2)) at DC delay, but this varies by order.
	// For an approximate cutoff match, scale each pole by
	//   omega_c = 2*pi*cutoff_hz.
	// Users wanting exact cutoff should apply a per-order correction
	// factor documented in Bessel filter tables.
	const double omega_c = 2.0 * std::numbers::pi_v<double> * cutoff_hz;
	p.s_poles.reserve(order);
	for (int i = 0; i < order; ++i) {
		auto r = rf.root(i);
		p.s_poles.emplace_back(r.real() * omega_c, r.imag() * omega_c);
	}
	return p;
}

// Elliptic prototype - not implemented in this header yet.
//
// The library's filter/iir/elliptic.hpp already computes elliptic
// pole/zero locations via a Jacobi-sn helper and a Bairstow-method
// polynomial factorer. Extracting a reusable, standalone API for the
// prototype builder here would require pulling those helpers out into
// sw::dsp::math (or a subheader). Tracked as a follow-up so this file
// stays scoped to closed-form families that don't require additional
// math infrastructure.
inline PoleZeroPlot elliptic_prototype(int /*order*/,
                                        double /*cutoff_hz*/,
                                        double /*ripple_dB*/,
                                        double /*stopband_dB*/) {
	throw std::runtime_error(
		"elliptic_prototype: not yet exposed as a standalone API. "
		"See filter/iir/elliptic.hpp for the underlying design; a "
		"follow-up will factor the Jacobi sn helper into "
		"sw::dsp::math for reuse here.");
}

// ============================================================================
// Constantinides transformations (s-plane)
//
// Transform a LP prototype (designed at cutoff_hz = omega_c) into an
// equivalent HP, BP, or BS filter at the target edges. Operates
// in-place on a PoleZeroPlot's s_poles + s_zeros arrays.
//
// LP -> HP: s' = omega_c_new / s. Poles and zeros invert; any
//   infinite zero (from an odd-order LP) becomes a finite zero at 0.
// LP -> BP: each s pole becomes two poles centered on omega_0 =
//   sqrt(low*high) with width omega_c * (high - low).
// LP -> BS: inverse of BP.
// ============================================================================

inline void lp_to_hp(PoleZeroPlot& plot, double cutoff_hz) {
	const double omega_new = 2.0 * std::numbers::pi_v<double> * cutoff_hz;
	auto invert = [&](std::vector<std::complex<double>>& v) {
		for (auto& s : v) {
			if (std::abs(s) < 1e-300) continue;
			s = std::complex<double>(omega_new, 0.0) / s;
		}
	};
	invert(plot.s_poles);
	// Existing finite zeros invert. If the LP had no finite zeros
	// (like Butterworth or Chebyshev I), HP gets `order` zeros at 0.
	const int existing_zeros = static_cast<int>(plot.s_zeros.size());
	invert(plot.s_zeros);
	const int missing_zeros = plot.order - existing_zeros;
	for (int i = 0; i < missing_zeros; ++i) {
		plot.s_zeros.emplace_back(0.0, 0.0);
	}
	plot.kind = "highpass";
	plot.cutoff_hz = cutoff_hz;
	plot.z_poles.clear();
	plot.z_zeros.clear();
}

// LP -> BP: substitution s' = (s^2 + omega_0^2) / (BW * s)
// where omega_0 = sqrt(omega_l * omega_h), BW = omega_h - omega_l.
// Each LP pole s_p produces two BP poles at the roots of
//   s^2 - BW*s_p*s + omega_0^2 = 0.
inline void lp_to_bp(PoleZeroPlot& plot, double low_hz, double high_hz) {
	if (!(low_hz > 0.0) || !(high_hz > low_hz))
		throw std::invalid_argument(
			"lp_to_bp: require 0 < low_hz < high_hz");
	const double omega_l = 2.0 * std::numbers::pi_v<double> * low_hz;
	const double omega_h = 2.0 * std::numbers::pi_v<double> * high_hz;
	const double omega_0 = std::sqrt(omega_l * omega_h);
	const double BW = omega_h - omega_l;

	auto transform = [&](std::vector<std::complex<double>>& v) {
		std::vector<std::complex<double>> out;
		out.reserve(v.size() * 2);
		for (const auto& sp : v) {
			// Discriminant: (BW*sp/2)^2 - omega_0^2
			const std::complex<double> half = 0.5 * BW * sp;
			const std::complex<double> disc = std::sqrt(
				half * half - std::complex<double>(omega_0 * omega_0, 0.0));
			out.push_back(half + disc);
			out.push_back(half - disc);
		}
		v = std::move(out);
	};
	transform(plot.s_poles);
	transform(plot.s_zeros);
	// LP-to-BP adds `order` zeros at s = 0 for filters without finite
	// LP zeros (the DC nulling in a bandpass response).
	// We accounted for pole doubling above; still need to add the DC
	// zeros. Total BP zeros = 2 * order (matching pole count for a
	// proper BP transfer function).
	const int existing_z = static_cast<int>(plot.s_zeros.size());
	const int needed_z   = 2 * plot.order;
	for (int i = 0; i < (needed_z - existing_z); ++i) {
		plot.s_zeros.emplace_back(0.0, 0.0);
	}
	plot.kind = "bandpass";
	plot.low_hz = low_hz;
	plot.high_hz = high_hz;
	plot.z_poles.clear();
	plot.z_zeros.clear();
}

// LP -> BS: substitution s' = BW*s / (s^2 + omega_0^2). Inverse of BP.
inline void lp_to_bs(PoleZeroPlot& plot, double low_hz, double high_hz) {
	if (!(low_hz > 0.0) || !(high_hz > low_hz))
		throw std::invalid_argument(
			"lp_to_bs: require 0 < low_hz < high_hz");
	const double omega_l = 2.0 * std::numbers::pi_v<double> * low_hz;
	const double omega_h = 2.0 * std::numbers::pi_v<double> * high_hz;
	const double omega_0 = std::sqrt(omega_l * omega_h);
	const double BW = omega_h - omega_l;

	auto transform = [&](std::vector<std::complex<double>>& v) {
		std::vector<std::complex<double>> out;
		out.reserve(v.size() * 2);
		for (const auto& sp : v) {
			if (std::abs(sp) < 1e-300) {
				// Zero-input maps to +/- j*omega_0.
				out.emplace_back(0.0,  omega_0);
				out.emplace_back(0.0, -omega_0);
				continue;
			}
			const std::complex<double> half = 0.5 * BW / sp;
			const std::complex<double> disc = std::sqrt(
				half * half - std::complex<double>(omega_0 * omega_0, 0.0));
			out.push_back(half + disc);
			out.push_back(half - disc);
		}
		v = std::move(out);
	};
	transform(plot.s_poles);
	transform(plot.s_zeros);
	// BS zeros include a pair at +/- j*omega_0 for each LP pole that
	// wasn't already representing a zero; already handled by transform.
	plot.kind = "bandstop";
	plot.low_hz = low_hz;
	plot.high_hz = high_hz;
	plot.z_poles.clear();
	plot.z_zeros.clear();
}

// ============================================================================
// Bilinear transform: s -> z via z = (2/T + s) / (2/T - s), T = 1/fs
// ============================================================================

inline void apply_bilinear(PoleZeroPlot& plot, double sample_rate_hz) {
	if (!(sample_rate_hz > 0.0))
		throw std::invalid_argument(
			"apply_bilinear: sample_rate_hz must be > 0");
	const double two_over_T = 2.0 * sample_rate_hz;
	const std::complex<double> k(two_over_T, 0.0);
	auto map = [&](const std::vector<std::complex<double>>& in,
	                std::vector<std::complex<double>>& out) {
		out.clear();
		out.reserve(in.size());
		for (const auto& s : in) {
			out.push_back((k + s) / (k - s));
		}
	};
	map(plot.s_poles, plot.z_poles);
	map(plot.s_zeros, plot.z_zeros);
	// Bilinear maps s = infinity -> z = -1. Any implicit s-plane
	// infinity zeros (poles > zeros in the s-plane) become z = -1 zeros.
	const std::size_t total_z_needed = plot.z_poles.size();
	while (plot.z_zeros.size() < total_z_needed) {
		plot.z_zeros.emplace_back(-1.0, 0.0);
	}
	plot.sample_rate_hz = sample_rate_hz;
}

} // namespace sw::dsp::transfer_function
