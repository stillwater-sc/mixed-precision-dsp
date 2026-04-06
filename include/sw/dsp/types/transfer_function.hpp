#pragma once
// transfer_function.hpp: first-class rational transfer function H(z)
//
// H(z) = B(z) / A(z) = (b0 + b1*z^-1 + ... + bM*z^-M) /
//                       (1  + a1*z^-1 + ... + aN*z^-N)
//
// Note: a0 is normalized to 1 (not stored in denominator).
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>

namespace sw::dsp {

template <DspField T>
class TransferFunction {
public:
	// Numerator coefficients: b0, b1, ..., bM
	mtl::vec::dense_vector<T> numerator;
	// Denominator coefficients: a1, a2, ..., aN (a0 = 1, implicit)
	mtl::vec::dense_vector<T> denominator;

	TransferFunction() = default;

	TransferFunction(mtl::vec::dense_vector<T> num, mtl::vec::dense_vector<T> den)
		: numerator(std::move(num)), denominator(std::move(den)) {}

	// Evaluate H(z) at a complex point z
	complex_for_t<T> evaluate(complex_for_t<T> z) const {
		using complex_t = complex_for_t<T>;

		// Compute B(z) = b0 + b1*z^-1 + b2*z^-2 + ...
		complex_t z_inv = complex_t(T{1}) / z;
		complex_t z_pow(T{1});
		complex_t num_val{};
		for (std::size_t i = 0; i < numerator.size(); ++i) {
			num_val = num_val + complex_t(numerator[i]) * z_pow;
			z_pow = z_pow * z_inv;
		}

		// Compute A(z) = 1 + a1*z^-1 + a2*z^-2 + ...
		z_pow = z_inv;
		complex_t den_val(T{1});
		for (std::size_t i = 0; i < denominator.size(); ++i) {
			den_val = den_val + complex_t(denominator[i]) * z_pow;
			z_pow = z_pow * z_inv;
		}

		return num_val / den_val;
	}

	// Evaluate frequency response at normalized frequency f in [0, 0.5]
	complex_for_t<T> frequency_response(double f) const {
		using complex_t = complex_for_t<T>;
		double w = two_pi * f;
		complex_t z(static_cast<T>(std::cos(w)), static_cast<T>(std::sin(w)));
		return evaluate(z);
	}

	// Check stability: all poles must be inside the unit circle.
	// Simple check: evaluate denominator at many points on the unit circle
	// and verify no zeros (which would be poles of H(z)) are near the circle.
	// For a rigorous check, find actual roots of the denominator polynomial.
	bool is_stable() const {
		// For now, use the simplified check: if denominator is empty, stable
		if (denominator.size() == 0) return true;

		// Check that the denominator polynomial has no roots on or outside
		// the unit circle by evaluating at many angles
		using std::abs;
		for (int k = 0; k < 360; ++k) {
			double angle = two_pi * k / 360.0;
			auto z = complex_for_t<T>(static_cast<T>(std::cos(angle)),
			                           static_cast<T>(std::sin(angle)));
			// Evaluate denominator A(z)
			auto z_inv = complex_for_t<T>(T{1}) / z;
			auto z_pow = z_inv;
			auto den_val = complex_for_t<T>(T{1});
			for (std::size_t i = 0; i < denominator.size(); ++i) {
				den_val = den_val + complex_for_t<T>(denominator[i]) * z_pow;
				z_pow = z_pow * z_inv;
			}
			double mag = static_cast<double>(abs(den_val));
			if (mag < 1e-6) return false;  // pole near unit circle
		}
		return true;
	}

	// Cascade two transfer functions: H1(z) * H2(z)
	TransferFunction operator*(const TransferFunction& rhs) const {
		// Convolve numerators and denominators
		auto conv = [](const mtl::vec::dense_vector<T>& a,
		               const mtl::vec::dense_vector<T>& b) {
			if (a.size() == 0) return b;
			if (b.size() == 0) return a;
			mtl::vec::dense_vector<T> result(a.size() + b.size() - 1, T{});
			for (std::size_t i = 0; i < a.size(); ++i) {
				for (std::size_t j = 0; j < b.size(); ++j) {
					result[i + j] = result[i + j] + a[i] * b[j];
				}
			}
			return result;
		};

		// For denominator cascade, we need to include the implicit a0=1
		// Prepend 1 to each denominator, convolve, then strip the leading 1
		mtl::vec::dense_vector<T> den_a(denominator.size() + 1);
		den_a[0] = T{1};
		for (std::size_t i = 0; i < denominator.size(); ++i) den_a[i + 1] = denominator[i];

		mtl::vec::dense_vector<T> den_b(rhs.denominator.size() + 1);
		den_b[0] = T{1};
		for (std::size_t i = 0; i < rhs.denominator.size(); ++i) den_b[i + 1] = rhs.denominator[i];

		auto num_result = conv(numerator, rhs.numerator);
		auto den_full = conv(den_a, den_b);

		// Strip the leading 1 from denominator
		mtl::vec::dense_vector<T> den_result(den_full.size() - 1);
		for (std::size_t i = 0; i < den_result.size(); ++i) {
			den_result[i] = den_full[i + 1];
		}

		return TransferFunction(std::move(num_result), std::move(den_result));
	}
};

} // namespace sw::dsp
