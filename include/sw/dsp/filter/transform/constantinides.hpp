#pragma once
// constantinides.hpp: bandpass and bandstop frequency transformations
//
// Constantinides transforms map a lowpass analog prototype to bandpass
// or bandstop digital filters. These are second-order transforms that
// double the filter order (each analog pole becomes a pair of digital poles).
//
// Reference: Constantinides, A.G. "Spectral Transformations for Digital
// Filters," Proc. IEEE vol 117, 1970.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <complex>
#include <limits>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/types/complex_pair.hpp>
#include <sw/dsp/filter/layout/layout.hpp>

namespace sw::dsp {

namespace detail {

// Fused multiply-add for complex numbers: c + v * c1
template <typename T>
inline complex_for_t<T> addmul(const complex_for_t<T>& c, T v, const complex_for_t<T>& c1) {
	return complex_for_t<T>(c.real() + v * c1.real(), c.imag() + v * c1.imag());
}

} // namespace detail

// Band-pass transform (Constantinides).
//
// Maps an analog lowpass prototype to a digital bandpass filter centered
// at fc with bandwidth fw (both normalized: freq/sample_rate).
// Doubles the filter order.
template <DspField T>
class BandPassTransform {
public:
	template <int MaxAnalog, int MaxDigital>
	BandPassTransform(T fc, T fw,
	                  PoleZeroLayout<T, MaxDigital>& digital,
	                  const PoleZeroLayout<T, MaxAnalog>& analog) {
		digital.reset();

		const T ww = two_pi_v<T> * fw;
		wc2_ = two_pi_v<T> * fc - ww / T{2};
		wc_ = wc2_ + ww;

		// Clamp to valid range
		const T eps = T{1e-8};
		if (wc2_ < eps) wc2_ = eps;
		if (wc_ > pi_v<T> - eps) wc_ = pi_v<T> - eps;

		a_ = std::cos((wc_ + wc2_) * T{0.5}) / std::cos((wc_ - wc2_) * T{0.5});
		b_ = T{1} / std::tan((wc_ - wc2_) * T{0.5});
		a2_ = a_ * a_;
		b2_ = b_ * b_;
		ab_ = a_ * b_;
		ab_2_ = T{2} * ab_;

		const int num_poles = analog.num_poles();
		const int pairs = num_poles / 2;

		for (int i = 0; i < pairs; ++i) {
			const auto& pair = analog[i];
			ComplexPair<T> p1 = transform(pair.poles.first);
			ComplexPair<T> z1 = transform(pair.zeros.first);

			digital.add_conjugate_pairs(p1.first, z1.first);
			digital.add_conjugate_pairs(p1.second, z1.second);
		}

		if (num_poles & 1) {
			ComplexPair<T> poles = transform(analog[pairs].poles.first);
			ComplexPair<T> zeros = transform(analog[pairs].zeros.first);
			digital.add(poles, zeros);
		}

		T wn = analog.normal_w();
		digital.set_normal(
			T{2} * std::atan(std::sqrt(
				std::tan((wc_ + wn) * T{0.5}) * std::tan((wc2_ + wn) * T{0.5}))),
			analog.normal_gain());
	}

private:
	ComplexPair<T> transform(complex_for_t<T> c) const {
		using complex_t = complex_for_t<T>;
		if (c.real() == std::numeric_limits<T>::infinity()) {
			return ComplexPair<T>(complex_t(T{-1}), complex_t(T{1}));
		}

		c = (T{1} + c) / (T{1} - c);  // bilinear

		complex_t v(T{});
		v = detail::addmul(v, T{4} * (b2_ * (a2_ - T{1}) + T{1}), c);
		v = v + complex_t(T{8} * (b2_ * (a2_ - T{1}) - T{1}));
		v = v * c;
		v = v + complex_t(T{4} * (b2_ * (a2_ - T{1}) + T{1}));
		v = std::sqrt(v);

		complex_t u = T{-1} * v;
		u = detail::addmul(u, ab_2_, c);
		u = u + complex_t(ab_2_);

		v = detail::addmul(v, ab_2_, c);
		v = v + complex_t(ab_2_);

		complex_t d(T{});
		d = detail::addmul(d, T{2} * (b_ - T{1}), c) + complex_t(T{2} * (T{1} + b_));

		return ComplexPair<T>(u / d, v / d);
	}

	T wc_, wc2_, a_, b_, a2_, b2_, ab_, ab_2_;
};

// Band-stop transform (Constantinides).
//
// Maps an analog lowpass prototype to a digital bandstop (notch) filter
// centered at fc with bandwidth fw. Doubles the filter order.
template <DspField T>
class BandStopTransform {
public:
	template <int MaxAnalog, int MaxDigital>
	BandStopTransform(T fc, T fw,
	                  PoleZeroLayout<T, MaxDigital>& digital,
	                  const PoleZeroLayout<T, MaxAnalog>& analog) {
		digital.reset();

		const T ww = two_pi_v<T> * fw;
		wc2_ = two_pi_v<T> * fc - ww / T{2};
		wc_ = wc2_ + ww;

		const T eps = T{1e-8};
		if (wc2_ < eps) wc2_ = eps;
		if (wc_ > pi_v<T> - eps) wc_ = pi_v<T> - eps;

		a_ = std::cos((wc_ + wc2_) * T{0.5}) / std::cos((wc_ - wc2_) * T{0.5});
		b_ = std::tan((wc_ - wc2_) * T{0.5});
		a2_ = a_ * a_;
		b2_ = b_ * b_;

		const int num_poles = analog.num_poles();
		const int pairs = num_poles / 2;

		for (int i = 0; i < pairs; ++i) {
			const auto& pair = analog[i];
			ComplexPair<T> p = transform(pair.poles.first);
			ComplexPair<T> z = transform(pair.zeros.first);

			// Ensure conjugate symmetry
			if (z.second == z.first) {
				z.second = std::conj(z.first);
			}

			digital.add_conjugate_pairs(p.first, z.first);
			digital.add_conjugate_pairs(p.second, z.second);
		}

		if (num_poles & 1) {
			ComplexPair<T> poles = transform(analog[pairs].poles.first);
			ComplexPair<T> zeros = transform(analog[pairs].zeros.first);
			digital.add(poles, zeros);
		}

		if (fc < T{0.25}) {
			digital.set_normal(pi_v<T>, analog.normal_gain());
		} else {
			digital.set_normal(T{}, analog.normal_gain());
		}
	}

private:
	ComplexPair<T> transform(complex_for_t<T> c) const {
		using complex_t = complex_for_t<T>;

		if (c.real() == std::numeric_limits<T>::infinity()) {
			c = complex_t(T{-1});
		} else {
			c = (T{1} + c) / (T{1} - c);  // bilinear
		}

		complex_t u(T{});
		u = detail::addmul(u, T{4} * (b2_ + a2_ - T{1}), c);
		u = u + complex_t(T{8} * (b2_ - a2_ + T{1}));
		u = u * c;
		u = u + complex_t(T{4} * (a2_ + b2_ - T{1}));
		u = std::sqrt(u);

		complex_t v = u * T{-0.5};
		v = v + complex_t(a_);
		v = detail::addmul(v, -a_, c);

		u = u * T{0.5};
		u = u + complex_t(a_);
		u = detail::addmul(u, -a_, c);

		complex_t d(b_ + T{1});
		d = detail::addmul(d, b_ - T{1}, c);

		return ComplexPair<T>(u / d, v / d);
	}

	T wc_, wc2_, a_, b_, a2_, b2_;
};

} // namespace sw::dsp
