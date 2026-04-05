#pragma once
// root_finder.hpp: Laguerre's method for finding complex polynomial roots
//
// Finds all roots of a polynomial with complex coefficients.
// Uses Laguerre's method with deflation and optional polishing.
//
// Ported from DSPFilters (Vinnie Falco), templated on scalar type,
// using std::array instead of raw pointers.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

// RootFinder<T, MaxDegree> finds roots of a polynomial
//   f(x) = coef[0] + coef[1]*x + coef[2]*x^2 + ... + coef[degree]*x^degree
//
// Usage:
//   RootFinder<double, 10> rf;
//   rf.coef(0) = 1.0;  // set coefficients
//   rf.coef(1) = ...;
//   rf.solve(degree);
//   auto r = rf.root(0);  // get roots
template <DspField T, int MaxDegree>
class RootFinder {
public:
	using complex_t = std::complex<T>;

	// Access input coefficients (degree+1 elements, index 0..degree)
	complex_t& coef(int i) {
		if (i < 0 || i > MaxDegree) throw std::out_of_range("RootFinder::coef index out of range");
		return coef_[i];
	}
	const complex_t& coef(int i) const {
		if (i < 0 || i > MaxDegree) throw std::out_of_range("RootFinder::coef index out of range");
		return coef_[i];
	}

	// Access roots (degree elements, index 0..degree-1)
	complex_t& root(int i) {
		if (i < 0 || i >= MaxDegree) throw std::out_of_range("RootFinder::root index out of range");
		return root_[i];
	}
	const complex_t& root(int i) const {
		if (i < 0 || i >= MaxDegree) throw std::out_of_range("RootFinder::root index out of range");
		return root_[i];
	}

	// Find all roots of the polynomial of given degree.
	// If polish=true, roots are refined using the original (un-deflated) polynomial.
	// If do_sort=true, roots are sorted by descending imaginary part.
	void solve(int degree, bool polish = true, bool do_sort = true) {
		if (degree < 1 || degree > MaxDegree)
			throw std::out_of_range("RootFinder::solve degree out of range");

		const T snap_eps = std::numeric_limits<T>::epsilon() * T{32};

		// Copy coefficients for deflation
		for (int j = 0; j <= degree; ++j) {
			deflated_[j] = coef_[j];
		}

		// Find each root via Laguerre + deflation
		for (int j = degree - 1; j >= 0; --j) {
			complex_t x(T{});
			int its;
			laguerre(j + 1, deflated_.data(), x, its);

			// Snap near-real roots to real axis
			if (std::abs(x.imag()) <= T{2} * snap_eps * std::max(T{1}, std::abs(x.real()))) {
				x = complex_t(x.real(), T{});
			}

			root_[j] = x;

			// Deflate: divide out the found root
			complex_t b = deflated_[j + 1];
			for (int jj = j; jj >= 0; --jj) {
				complex_t c = deflated_[jj];
				deflated_[jj] = b;
				b = x * b + c;
			}
		}

		// Polish roots using original coefficients
		if (polish) {
			for (int j = 0; j < degree; ++j) {
				int its;
				laguerre(degree, coef_.data(), root_[j], its);
			}
		}

		if (do_sort) {
			sort(degree);
		}
	}

	// Sort roots by descending imaginary part
	void sort(int degree) {
		if (degree < 0 || degree > MaxDegree)
			throw std::out_of_range("RootFinder::sort degree out of range");
		// Insertion sort
		for (int j = 1; j < degree; ++j) {
			complex_t x = root_[j];
			int i;
			for (i = j - 1; i >= 0; --i) {
				if (root_[i].imag() >= x.imag()) break;
				root_[i + 1] = root_[i];
			}
			root_[i + 1] = x;
		}
	}

private:
	// Laguerre's method: improve x as a root of polynomial a[] of given degree
	static void laguerre(int degree, complex_t a[], complex_t& x, int& its) {
		constexpr int MR = 8;
		constexpr int MT = 10;
		constexpr int MAXIT = MT * MR;
		const T EPS = std::numeric_limits<T>::epsilon();

		static const double frac[MR + 1] = {
			0.0, 0.5, 0.25, 0.75, 0.13, 0.38, 0.62, 0.88, 1.0
		};

		int m = degree;
		for (int iter = 1; iter <= MAXIT; ++iter) {
			its = iter;
			complex_t b = a[m];
			T err = std::abs(b);
			complex_t d(T{}), f(T{});
			T abx = std::abs(x);

			for (int j = m - 1; j >= 0; --j) {
				f = x * f + d;
				d = x * d + b;
				b = x * b + a[j];
				err = std::abs(b) + abx * err;
			}
			err *= EPS;

			if (std::abs(b) <= err) return;

			complex_t g = d / b;
			complex_t g2 = g * g;
			complex_t h = g2 - complex_t(T{2}) * f / b;

			complex_t sq = std::sqrt(
				complex_t(static_cast<T>(m - 1)) * (complex_t(static_cast<T>(m)) * h - g2));
			complex_t gp = g + sq;
			complex_t gm = g - sq;

			T abp = std::abs(gp);
			T abm = std::abs(gm);
			if (abp < abm) gp = gm;

			complex_t dx;
			if (std::max(abp, abm) > T{}) {
				dx = complex_t(static_cast<T>(m)) / gp;
			} else {
				dx = std::polar(static_cast<double>(T{1} + abx), static_cast<double>(iter));
			}

			complex_t x1 = x - dx;
			if (x == x1) return;

			if (iter % MT != 0) {
				x = x1;
			} else {
				x = x - complex_t(static_cast<T>(frac[iter / MT])) * dx;
			}
		}

		throw std::runtime_error("root_finder: Laguerre failed to converge");
	}

	std::array<complex_t, MaxDegree + 1> coef_{};
	std::array<complex_t, MaxDegree + 1> deflated_{};
	std::array<complex_t, MaxDegree> root_{};
};

} // namespace sw::dsp
