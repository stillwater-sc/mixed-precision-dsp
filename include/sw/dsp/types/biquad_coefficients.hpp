#pragma once
// biquad_coefficients.hpp: coefficients for a second-order IIR section
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <complex>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/types/pole_zero_pair.hpp>

namespace sw::dsp {

// Coefficients for a second-order (biquad) IIR section.
//
// Transfer function:
//   H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
//
// Note: a0 is normalized to 1. The denominator coefficients stored
// here are a1 and a2 (not a0*a1, a0*a2).
template <DspField CoeffScalar>
struct BiquadCoefficients {
	CoeffScalar b0{}, b1{}, b2{};  // numerator
	CoeffScalar a1{}, a2{};        // denominator (a0 = 1, normalized)

	constexpr BiquadCoefficients() = default;

	constexpr BiquadCoefficients(CoeffScalar b0_, CoeffScalar b1_, CoeffScalar b2_,
	                             CoeffScalar a1_, CoeffScalar a2_)
		: b0(b0_), b1(b1_), b2(b2_), a1(a1_), a2(a2_) {}

	// Set to identity (pass-through): H(z) = 1
	constexpr void set_identity() {
		b0 = CoeffScalar{1};
		b1 = CoeffScalar{};
		b2 = CoeffScalar{};
		a1 = CoeffScalar{};
		a2 = CoeffScalar{};
	}

	// Set coefficients from a single pole/zero pair (first-order section)
	void set_one_pole(const std::complex<CoeffScalar>& pole,
	                  const std::complex<CoeffScalar>& zero) {
		b0 = CoeffScalar{1};
		b1 = CoeffScalar{-1} * zero.real();
		b2 = CoeffScalar{};
		a1 = CoeffScalar{-1} * pole.real();
		a2 = CoeffScalar{};
	}

	// Set coefficients from a conjugate pair of poles/zeros (second-order section)
	void set_two_pole(const std::complex<CoeffScalar>& pole1,
	                  const std::complex<CoeffScalar>& zero1,
	                  const std::complex<CoeffScalar>& pole2,
	                  const std::complex<CoeffScalar>& zero2) {
		using std::real;
		using std::imag;

		// Denominator: (1 - pole1*z^-1)(1 - pole2*z^-1)
		//            = 1 - (pole1+pole2)*z^-1 + pole1*pole2*z^-2
		CoeffScalar pr = real(pole1) + real(pole2);
		CoeffScalar pp = real(pole1) * real(pole2) + imag(pole1) * imag(pole1);
		a1 = CoeffScalar{-1} * pr;
		a2 = pp;

		// Numerator: (1 - zero1*z^-1)(1 - zero2*z^-1)
		CoeffScalar zr = real(zero1) + real(zero2);
		CoeffScalar zp = real(zero1) * real(zero2) + imag(zero1) * imag(zero1);
		b0 = CoeffScalar{1};
		b1 = CoeffScalar{-1} * zr;
		b2 = zp;
	}

	// Set from a PoleZeroPair (dispatches to one_pole or two_pole)
	void set_from_pole_zero_pair(const PoleZeroPair<CoeffScalar>& pz) {
		if (pz.is_single_pole()) {
			set_one_pole(pz.poles.first, pz.zeros.first);
		} else {
			set_two_pole(pz.poles.first, pz.zeros.first,
			             pz.poles.second, pz.zeros.second);
		}
	}

	// Apply a gain scale factor to the numerator
	constexpr void apply_scale(CoeffScalar scale) {
		b0 = b0 * scale;
		b1 = b1 * scale;
		b2 = b2 * scale;
	}

	// Evaluate frequency response at normalized frequency f in [0, 0.5]
	// where f = frequency / sample_rate
	std::complex<CoeffScalar> response(double normalized_freq) const {
		using complex_t = std::complex<double>;
		const double w = 2.0 * 3.14159265358979323846 * normalized_freq;
		const complex_t z1 = std::exp(complex_t(0, -w));        // z^-1
		const complex_t z2 = std::exp(complex_t(0, -2.0 * w));  // z^-2

		complex_t num = static_cast<double>(b0)
		              + static_cast<double>(b1) * z1
		              + static_cast<double>(b2) * z2;
		complex_t den = 1.0
		              + static_cast<double>(a1) * z1
		              + static_cast<double>(a2) * z2;

		auto result = num / den;
		return std::complex<CoeffScalar>(
			static_cast<CoeffScalar>(result.real()),
			static_cast<CoeffScalar>(result.imag()));
	}
};

} // namespace sw::dsp
