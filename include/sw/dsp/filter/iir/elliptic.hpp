#pragma once
// elliptic.hpp: Elliptic (Cauer) IIR filter design
//
// Equiripple passband AND stopband. Specified by order, passband ripple,
// and rolloff. Has the steepest transition of any classical IIR filter
// for a given order, at the cost of ripple in both bands.
//
// All arithmetic parameterized on T. Uses std::array for temporaries.
// ADL-friendly math calls throughout.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/math/elliptic_integrals.hpp>
#include <sw/dsp/filter/layout/layout.hpp>
#include <sw/dsp/filter/layout/analog_prototype.hpp>
#include <sw/dsp/filter/biquad/cascade.hpp>
#include <sw/dsp/filter/transform/bilinear.hpp>
#include <sw/dsp/filter/transform/constantinides.hpp>

namespace sw::dsp::iir {

// Elliptic analog lowpass prototype.
// MaxOrder determines array sizing. All temporaries are std::array<T, N>.
template <DspField T, int MaxOrder>
class EllipticAnalogPrototype {
	// Array size for internal temporaries: covers up to 2*MaxOrder+2
	static constexpr int N = 2 * MaxOrder + 4;

public:
	void design(int num_poles, T ripple_db, T rolloff,
	            PoleZeroLayout<T, MaxOrder>& layout) {
		if (num_poles <= 0)
			throw std::invalid_argument("elliptic: num_poles must be > 0");
		if (num_poles > MaxOrder)
			throw std::out_of_range("elliptic: num_poles exceeds MaxOrder");

		using std::sqrt;
		using std::exp;
		using std::pow;
		using std::sin;
		using std::fabs;

		layout.reset();

		const int n = num_poles;
		T e2 = pow(T{10}, ripple_db / T{10}) - T{1};
		T xi = T{5} * exp(rolloff - T{1}) + T{1};

		T K = elliptic_K(T{1} / xi);
		T Kprime = elliptic_K(sqrt(T{1} - T{1} / (xi * xi)));

		// Compute zeros via Jacobi sn function
		int ni = ((n & 1) == 1) ? 0 : 1;
		std::array<T, MaxOrder + 1> f{};
		std::array<T, MaxOrder + 1> zeros{};

		for (int i = 1; i <= n / 2; ++i) {
			T u = static_cast<T>(2 * i - ni) * K / static_cast<T>(n);
			T sn = calcsn(u, K, Kprime);
			sn = sn * two_pi_v<T> / K;
			f[i] = T{1} / sn;
			zeros[i - 1] = f[i];
		}
		zeros[n / 2] = std::numeric_limits<T>::infinity();

		T fb = T{1} / two_pi_v<T>;
		int nin = n % 2;
		int n2 = n / 2;

		std::array<T, MaxOrder + 1> z1{};
		for (int i = 1; i <= n2; ++i) {
			T x = f[n2 + 1 - i];
			z1[i] = sqrt(T{1} - T{1} / (x * x));
		}

		T e = sqrt(e2);
		T fbb = fb * fb;
		int m = nin + 2 * n2;
		int em = 2 * (m / 2);
		T tp = two_pi_v<T>;

		// Internal temporary arrays
		std::array<T, N> s1{}, b1{}, a1{}, c1{}, d1{};
		std::array<T, MaxOrder + 1> p{}, q1{};

		// calcfz: compute f(z)
		{
			int i = 1;
			if (nin == 1) s1[i++] = T{1};
			for (; i <= nin + n2; ++i) {
				s1[i] = z1[i - nin];
				s1[i + n2] = z1[i - nin];
			}
			prodpoly(nin + 2 * n2, s1, b1, a1);
			for (int j = 0; j <= em; j += 2)
				a1[j] = e * b1[j];
			// calcfz2
			for (int j = 0; j <= 2 * em; j += 2) {
				int ji = 0, jf = 0;
				if (j < em + 2) { ji = 0; jf = j; }
				if (j > em) { ji = j - em; jf = em; }
				c1[j] = T{};
				for (int k = ji; k <= jf; k += 2)
					c1[j] = c1[j] + a1[k] * (a1[j - k] * pow(T{10}, static_cast<T>(m - j / 2)));
			}
		}

		// calcqz: compute q(z)
		{
			int i;
			for (i = 1; i <= nin; ++i)
				s1[i] = T{-10};
			for (; i <= nin + n2; ++i)
				s1[i] = T{-10} * z1[i - nin] * z1[i - nin];
			for (; i <= nin + 2 * n2; ++i)
				s1[i] = s1[i - n2];
			prodpoly(m, s1, b1, a1);
			int dd = ((nin & 1) == 1) ? -1 : 1;
			for (int j = 0; j <= 2 * m; j += 2)
				d1[j] = static_cast<T>(dd) * b1[j / 2];
		}

		if (m > em) c1[2 * m] = T{};
		for (int i = 0; i <= 2 * m; i += 2)
			a1[m - i / 2] = c1[i] + d1[i];

		T a0 = findfact(m, a1, b1, c1, p, q1);

		// Extract poles and zeros
		int r = 0;
		while (r < em / 2) {
			++r;
			p[r] = p[r] / T{10};
			q1[r] = q1[r] / T{100};
			T d = T{1} + p[r] + q1[r];
			T b1r = (T{1} + p[r] / T{2}) * fbb / d;
			T zf1r = fb / pow(d, T{0.25});
			T zq1r = T{1} / sqrt(fabs(T{2} * (T{1} - b1r / (zf1r * zf1r))));
			T zw1r = tp * zf1r;

			complex_for_t<T> pole(
				T{-0.5} * zw1r / zq1r,
				T{0.5} * sqrt(fabs(zw1r * zw1r / (zq1r * zq1r) - T{4} * zw1r * zw1r)));

			complex_for_t<T> zero(T{}, zeros[r - 1]);

			layout.add_conjugate_pairs(pole, zero);
		}

		if (a0 != T{}) {
			T real_pole = T{} - sqrt(fbb / (T{0.1} * a0 - T{1})) * tp;
			layout.add(complex_for_t<T>(real_pole), s_infinity<T>());
		}

		layout.set_normal(T{},
			(num_poles & 1) ? T{1} : static_cast<T>(pow(T{10}, T{-1} * ripple_db / T{20})));
	}

private:
	// Product of (z + s1[i]) for i = 1..sn, stored in b1
	static void prodpoly(int sn, std::array<T, N>& s1,
	                     std::array<T, N>& b1, std::array<T, N>& a1) {
		b1[0] = s1[1];
		b1[1] = T{1};
		for (int j = 2; j <= sn; ++j) {
			a1[0] = s1[j] * b1[0];
			for (int i = 1; i <= j - 1; ++i)
				a1[i] = b1[i - 1] + s1[j] * b1[i];
			for (int i = 0; i != j; ++i)
				b1[i] = a1[i];
			b1[j] = T{1};
		}
	}

	// Compute factors by Bairstow's method
	static T findfact(int t, std::array<T, N>& a1,
	                  std::array<T, N>& b1, std::array<T, N>& c1,
	                  std::array<T, MaxOrder + 1>& p, std::array<T, MaxOrder + 1>& q1) {
		using std::fabs;

		T a = T{};
		for (int i = 1; i <= t; ++i)
			a1[i] = a1[i] / a1[0];
		a1[0] = T{1};
		b1[0] = T{1};
		c1[0] = T{1};

		int i1 = 0;
		for (;;) {
			if (t <= 2) break;
			T p0 = T{}, q0 = T{};
			++i1;
			for (;;) {
				b1[1] = a1[1] - p0;
				c1[1] = b1[1] - p0;
				for (int i = 2; i <= t; ++i)
					b1[i] = a1[i] - p0 * b1[i - 1] - q0 * b1[i - 2];
				for (int i = 2; i < t; ++i)
					c1[i] = b1[i] - p0 * c1[i - 1] - q0 * c1[i - 2];
				int x1 = t - 1, x2 = t - 2, x3 = t - 3;
				T x4 = c1[x2] * c1[x2] + c1[x3] * (b1[x1] - c1[x1]);
				if (x4 == T{}) x4 = T{1e-3};
				T ddp = (b1[x1] * c1[x2] - b1[t] * c1[x3]) / x4;
				p0 = p0 + ddp;
				T dq = (b1[t] * c1[x2] - b1[x1] * (c1[x1] - b1[x1])) / x4;
				q0 = q0 + dq;
				if (fabs(ddp + dq) < T{1e-6}) break;
			}
			p[i1] = p0;
			q1[i1] = q0;
			a1[1] = a1[1] - p0;
			t -= 2;
			for (int i = 2; i <= t; ++i)
				a1[i] = a1[i] - p0 * a1[i - 1] - q0 * a1[i - 2];
			if (t <= 2) break;
		}

		if (t == 2) { ++i1; p[i1] = a1[1]; q1[i1] = a1[2]; }
		if (t == 1) a = T{} - a1[1];

		return a;
	}

	// Jacobi sn function via q-series
	static T calcsn(T u, T K, T Kprime) {
		using std::exp;
		using std::pow;
		using std::sin;

		T q = exp(T{-1} * pi_v<T> * Kprime / K);
		T v = half_pi_v<T> * u / K;
		T sn = T{};
		for (int j = 0; ; ++j) {
			T w = pow(q, static_cast<T>(j) + T{0.5});
			sn = sn + w * sin((T{2} * static_cast<T>(j) + T{1}) * v) / (T{1} - w * w);
			if (static_cast<double>(w) < 1e-7) break;
		}
		return sn;
	}
};

// Elliptic filter designs: LP, HP, BP, BS
template <int MaxOrder,
          DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class EllipticLowPass {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = (MaxOrder + 1) / 2;

	void setup(int order, double sample_rate, double cutoff_freq,
	           double ripple_db, double rolloff) {
		EllipticAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, static_cast<CoeffScalar>(ripple_db),
		             static_cast<CoeffScalar>(rolloff), analog_);
		LowPassTransform<CoeffScalar>(
			static_cast<CoeffScalar>(cutoff_freq / sample_rate), digital_, analog_);
		cascade_.set_layout(digital_);
	}

	const Cascade<CoeffScalar, max_stages>& cascade() const { return cascade_; }

private:
	PoleZeroLayout<CoeffScalar, MaxOrder> analog_;
	PoleZeroLayout<CoeffScalar, MaxOrder> digital_;
	Cascade<CoeffScalar, max_stages> cascade_;
};

template <int MaxOrder,
          DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class EllipticHighPass {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = (MaxOrder + 1) / 2;

	void setup(int order, double sample_rate, double cutoff_freq,
	           double ripple_db, double rolloff) {
		EllipticAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, static_cast<CoeffScalar>(ripple_db),
		             static_cast<CoeffScalar>(rolloff), analog_);
		HighPassTransform<CoeffScalar>(
			static_cast<CoeffScalar>(cutoff_freq / sample_rate), digital_, analog_);
		cascade_.set_layout(digital_);
	}

	const Cascade<CoeffScalar, max_stages>& cascade() const { return cascade_; }

private:
	PoleZeroLayout<CoeffScalar, MaxOrder> analog_;
	PoleZeroLayout<CoeffScalar, MaxOrder> digital_;
	Cascade<CoeffScalar, max_stages> cascade_;
};

template <int MaxOrder,
          DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class EllipticBandPass {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = MaxOrder;

	void setup(int order, double sample_rate, double center_freq, double width_freq,
	           double ripple_db, double rolloff) {
		EllipticAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, static_cast<CoeffScalar>(ripple_db),
		             static_cast<CoeffScalar>(rolloff), analog_);
		BandPassTransform<CoeffScalar>(
			static_cast<CoeffScalar>(center_freq / sample_rate),
			static_cast<CoeffScalar>(width_freq / sample_rate), digital_, analog_);
		cascade_.set_layout(digital_);
	}

	const Cascade<CoeffScalar, max_stages>& cascade() const { return cascade_; }

private:
	PoleZeroLayout<CoeffScalar, MaxOrder> analog_;
	PoleZeroLayout<CoeffScalar, MaxOrder * 2> digital_;
	Cascade<CoeffScalar, max_stages> cascade_;
};

template <int MaxOrder,
          DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class EllipticBandStop {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = MaxOrder;

	void setup(int order, double sample_rate, double center_freq, double width_freq,
	           double ripple_db, double rolloff) {
		EllipticAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, static_cast<CoeffScalar>(ripple_db),
		             static_cast<CoeffScalar>(rolloff), analog_);
		BandStopTransform<CoeffScalar>(
			static_cast<CoeffScalar>(center_freq / sample_rate),
			static_cast<CoeffScalar>(width_freq / sample_rate), digital_, analog_);
		cascade_.set_layout(digital_);
	}

	const Cascade<CoeffScalar, max_stages>& cascade() const { return cascade_; }

private:
	PoleZeroLayout<CoeffScalar, MaxOrder> analog_;
	PoleZeroLayout<CoeffScalar, MaxOrder * 2> digital_;
	Cascade<CoeffScalar, max_stages> cascade_;
};

} // namespace sw::dsp::iir
