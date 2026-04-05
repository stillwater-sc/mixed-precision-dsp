#pragma once
// biquad.hpp: biquad coefficient operations and pole/zero conversion
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/types/biquad_coefficients.hpp>
#include <sw/dsp/types/pole_zero_pair.hpp>
#include <sw/dsp/math/constants.hpp>

namespace sw::dsp {

// Represents a biquad as pole/zero pairs with gain, allowing
// reconstruction of exact coefficients and smooth interpolation
// in the z-plane.
template <DspField T>
struct BiquadPoleState : PoleZeroPair<T> {
	T gain{1};

	BiquadPoleState() = default;

	// Construct from biquad coefficients by extracting poles and zeros
	explicit BiquadPoleState(const BiquadCoefficients<T>& c) {
		using complex_t = std::complex<T>;

		// Extract zeros from numerator: b0 + b1*z^-1 + b2*z^-2
		// Roots of b0*z^2 + b1*z + b2 = 0
		if (c.b2 == T{}) {
			// First-order: single zero at -b1/b0
			if (c.b0 != T{}) {
				this->zeros.first = complex_t(T{-1} * c.b1 / c.b0);
			}
			this->zeros.second = complex_t{};
		} else {
			T disc_z = c.b1 * c.b1 - T{4} * c.b0 * c.b2;
			if (disc_z >= T{}) {
				T sq = std::sqrt(disc_z);
				this->zeros.first = complex_t((T{-1} * c.b1 + sq) / (T{2} * c.b0));
				this->zeros.second = complex_t((T{-1} * c.b1 - sq) / (T{2} * c.b0));
			} else {
				T sq = std::sqrt(T{-1} * disc_z);
				this->zeros.first = complex_t(T{-1} * c.b1 / (T{2} * c.b0),
				                              sq / (T{2} * c.b0));
				this->zeros.second = std::conj(this->zeros.first);
			}
		}

		// Extract poles from denominator: 1 + a1*z^-1 + a2*z^-2
		// Roots of z^2 + a1*z + a2 = 0
		if (c.a2 == T{}) {
			this->poles.first = complex_t(T{-1} * c.a1);
			this->poles.second = complex_t{};
		} else {
			T disc_p = c.a1 * c.a1 - T{4} * c.a2;
			if (disc_p >= T{}) {
				T sq = std::sqrt(disc_p);
				this->poles.first = complex_t((T{-1} * c.a1 + sq) / T{2});
				this->poles.second = complex_t((T{-1} * c.a1 - sq) / T{2});
			} else {
				T sq = std::sqrt(T{-1} * disc_p);
				this->poles.first = complex_t(T{-1} * c.a1 / T{2}, sq / T{2});
				this->poles.second = std::conj(this->poles.first);
			}
		}

		gain = c.b0;
	}
};

} // namespace sw::dsp
