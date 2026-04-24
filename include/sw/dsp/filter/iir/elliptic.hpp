#pragma once
// elliptic.hpp: Elliptic (Cauer) IIR filter design
//
// Two design interfaces:
//
// 1. DSPFilters-style: setup(order, fs, fc, ripple_db, rolloff)
//    `rolloff` is a selectivity parameter in [0.1, 5.0]. Internally the
//    elliptic modulus is k = 1/xi with xi = 5*exp(rolloff-1)+1.
//    Classes: EllipticLowPass, EllipticHighPass, EllipticBandPass, EllipticBandStop
//
// 2. Matlab/scipy-style (Cauer-Darlington): setup(order, fs, fp, fs_stop, ripple_db, stopband_db)
//    Explicit passband ripple Ap (dB) and stopband attenuation As (dB),
//    with both passband and stopband edge frequencies. Solves for the
//    selectivity modulus k = fp/fs_stop via the Cauer-Darlington relations.
//    Classes: EllipticLowPassSpec, EllipticHighPassSpec, EllipticBandPassSpec, EllipticBandStopSpec
//    Free function: elliptic_minimum_order(ripple_db, stopband_db, fp, fs_stop, sample_rate)
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
	// DSPFilters-style: rolloff is a selectivity parameter in [0.1, 5.0].
	// Internally maps to xi = 5*exp(rolloff-1)+1, then k = 1/xi.
	void design(int num_poles, T ripple_db, T rolloff,
	            PoleZeroLayout<T, MaxOrder>& layout) {
		if (!(rolloff >= T{0.1} && rolloff <= T{5}))
			throw std::invalid_argument(
				"elliptic: rolloff must be in [0.1, 5.0] (selectivity "
				"parameter, not stopband dB)");

		using std::exp;
		T xi = T{5} * exp(rolloff - T{1}) + T{1};
		design_core(num_poles, ripple_db, xi, layout);
	}

	// Cauer-Darlington: specify the selectivity modulus k = fp/fs directly.
	// xi = 1/k. The modulus k must be in (0, 1).
	void design_from_modulus(int num_poles, T ripple_db, T selectivity_k,
	                         PoleZeroLayout<T, MaxOrder>& layout) {
		if (!(selectivity_k > T{0} && selectivity_k < T{1}))
			throw std::invalid_argument(
				"elliptic: selectivity_k must be in (0, 1)");
		T xi = T{1} / selectivity_k;
		design_core(num_poles, ripple_db, xi, layout);
	}

private:
	void design_core(int num_poles, T ripple_db, T xi,
	                 PoleZeroLayout<T, MaxOrder>& layout) {
		if (num_poles <= 0)
			throw std::invalid_argument("elliptic: num_poles must be > 0");
		if (num_poles > MaxOrder)
			throw std::out_of_range("elliptic: num_poles exceeds MaxOrder");
		if (!(ripple_db > T{0}))
			throw std::invalid_argument("elliptic: ripple_db must be > 0");

		using std::sqrt;
		using std::exp;
		using std::pow;
		using std::sin;
		using std::fabs;

		layout.reset();

		const int n = num_poles;
		T e2 = pow(T{10}, ripple_db / T{10}) - T{1};

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
		// Convergence threshold: use T-scaled epsilon to avoid
		// premature termination at high precision or excessive
		// iterations at low precision
		T tol = std::numeric_limits<T>::epsilon() * T{1000};
		T sn = T{};
		for (int j = 0; ; ++j) {
			T w = pow(q, static_cast<T>(j) + T{0.5});
			sn = sn + w * sin((T{2} * static_cast<T>(j) + T{1}) * v) / (T{1} - w * w);
			if (w < tol) break;
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

// ============================================================================
// Cauer-Darlington (Matlab/scipy-style) design: (Ap, As, fp, fs_stop)
// ============================================================================

// Validation helpers
inline void validate_lowpass(double passband_freq, double stopband_freq,
                             double sample_rate, double ripple_db, double stopband_db) {
	if (ripple_db <= 0 || stopband_db <= 0)
		throw std::invalid_argument("elliptic: ripple_db and stopband_db must be > 0");
	if (passband_freq <= 0 || stopband_freq <= 0)
		throw std::invalid_argument("elliptic: frequencies must be > 0");
	if (passband_freq >= stopband_freq)
		throw std::invalid_argument("elliptic: passband_freq must be < stopband_freq");
	if (stopband_freq >= sample_rate / 2.0)
		throw std::invalid_argument("elliptic: stopband_freq must be < Nyquist");
}

inline void validate_highpass(double passband_freq, double stopband_freq,
                              double sample_rate, double ripple_db, double stopband_db) {
	if (ripple_db <= 0 || stopband_db <= 0)
		throw std::invalid_argument("elliptic: ripple_db and stopband_db must be > 0");
	if (passband_freq <= 0 || stopband_freq <= 0)
		throw std::invalid_argument("elliptic: frequencies must be > 0");
	if (stopband_freq >= passband_freq)
		throw std::invalid_argument("elliptic highpass: stopband_freq must be < passband_freq");
	if (passband_freq >= sample_rate / 2.0)
		throw std::invalid_argument("elliptic highpass: passband_freq must be < Nyquist");
}

// Minimum filter order needed to meet both passband and stopband specs.
// Returns the smallest integer n such that the elliptic filter achieves
// at least stopband_db attenuation with at most ripple_db passband ripple.
//
// passband_freq and stopband_freq are in Hz; sample_rate in Hz.
// passband_freq < stopband_freq (lowpass orientation).
inline int elliptic_minimum_order(double ripple_db, double stopband_db,
                                  double passband_freq, double stopband_freq,
                                  double sample_rate) {
	if (ripple_db <= 0)
		throw std::invalid_argument("elliptic_minimum_order: ripple_db must be > 0");
	if (stopband_db <= 0)
		throw std::invalid_argument("elliptic_minimum_order: stopband_db must be > 0");
	if (passband_freq <= 0 || stopband_freq <= 0)
		throw std::invalid_argument("elliptic_minimum_order: frequencies must be > 0");
	if (passband_freq >= stopband_freq)
		throw std::invalid_argument("elliptic_minimum_order: passband_freq must be < stopband_freq");
	if (stopband_freq >= sample_rate / 2.0)
		throw std::invalid_argument("elliptic_minimum_order: stopband_freq must be < Nyquist");

	double wp = std::tan(sw::dsp::pi * passband_freq / sample_rate);
	double ws = std::tan(sw::dsp::pi * stopband_freq / sample_rate);
	double k = wp / ws;
	double kp = std::sqrt(1.0 - k * k);

	double eps_p = std::sqrt(std::pow(10.0, ripple_db / 10.0) - 1.0);
	double eps_s = std::sqrt(std::pow(10.0, stopband_db / 10.0) - 1.0);
	double k1 = eps_p / eps_s;
	double k1p = std::sqrt(1.0 - k1 * k1);

	double Kk = sw::dsp::elliptic_K(k);
	double Kkp = sw::dsp::elliptic_K(kp);
	double Kk1 = sw::dsp::elliptic_K(k1);
	double Kk1p = sw::dsp::elliptic_K(k1p);

	double n_exact = (Kk * Kk1p) / (Kkp * Kk1);
	return static_cast<int>(std::ceil(n_exact));
}

// Lowpass with explicit passband/stopband specification.
// setup(order, sample_rate, passband_freq, stopband_freq, ripple_db, stopband_db)
template <int MaxOrder,
          DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class EllipticLowPassSpec {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = (MaxOrder + 1) / 2;

	void setup(int order, double sample_rate, double passband_freq,
	           double stopband_freq, double ripple_db, double stopband_db) {
		using std::tan;
		validate_lowpass(passband_freq, stopband_freq, sample_rate, ripple_db, stopband_db);

		// Bilinear prewarp in CoeffScalar so posit/cfloat users get design-time
		// math at their declared precision (required for embedded deployments).
		constexpr CoeffScalar pi_T = CoeffScalar(sw::dsp::pi);
		const CoeffScalar fs = CoeffScalar(sample_rate);
		const CoeffScalar wp = tan(pi_T * CoeffScalar(passband_freq) / fs);
		const CoeffScalar ws = tan(pi_T * CoeffScalar(stopband_freq) / fs);
		const CoeffScalar k  = wp / ws;

		EllipticAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design_from_modulus(order, CoeffScalar(ripple_db), k, analog_);
		LowPassTransform<CoeffScalar>(
			CoeffScalar(passband_freq) / fs, digital_, analog_);
		cascade_.set_layout(digital_);
	}

	const Cascade<CoeffScalar, max_stages>& cascade() const { return cascade_; }

private:
	PoleZeroLayout<CoeffScalar, MaxOrder> analog_;
	PoleZeroLayout<CoeffScalar, MaxOrder> digital_;
	Cascade<CoeffScalar, max_stages> cascade_;
};

// Highpass with explicit passband/stopband specification.
// setup(order, sample_rate, passband_freq, stopband_freq, ripple_db, stopband_db)
// passband_freq > stopband_freq (highpass: passband is above stopband edge).
template <int MaxOrder,
          DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class EllipticHighPassSpec {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = (MaxOrder + 1) / 2;

	void setup(int order, double sample_rate, double passband_freq,
	           double stopband_freq, double ripple_db, double stopband_db) {
		using std::tan;
		validate_highpass(passband_freq, stopband_freq, sample_rate, ripple_db, stopband_db);

		constexpr CoeffScalar pi_T = CoeffScalar(sw::dsp::pi);
		const CoeffScalar fs = CoeffScalar(sample_rate);
		const CoeffScalar wp = tan(pi_T * CoeffScalar(passband_freq) / fs);
		const CoeffScalar ws = tan(pi_T * CoeffScalar(stopband_freq) / fs);
		const CoeffScalar k  = ws / wp;

		EllipticAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design_from_modulus(order, CoeffScalar(ripple_db), k, analog_);
		HighPassTransform<CoeffScalar>(
			CoeffScalar(passband_freq) / fs, digital_, analog_);
		cascade_.set_layout(digital_);
	}

	const Cascade<CoeffScalar, max_stages>& cascade() const { return cascade_; }

private:
	PoleZeroLayout<CoeffScalar, MaxOrder> analog_;
	PoleZeroLayout<CoeffScalar, MaxOrder> digital_;
	Cascade<CoeffScalar, max_stages> cascade_;
};

// Bandpass with explicit passband/stopband specification.
// setup(order, sample_rate, pass_low, pass_high, stop_low, stop_high, ripple_db, stopband_db)
template <int MaxOrder,
          DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class EllipticBandPassSpec {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = MaxOrder;

	void setup(int order, double sample_rate,
	           double pass_low, double pass_high,
	           double stop_low, double stop_high,
	           double ripple_db, double stopband_db) {
		using std::tan; using std::abs;
		if (ripple_db <= 0 || stopband_db <= 0)
			throw std::invalid_argument("elliptic bandpass: ripple and stopband must be > 0");
		if (!(stop_low < pass_low && pass_low < pass_high && pass_high < stop_high))
			throw std::invalid_argument("elliptic bandpass: need stop_low < pass_low < pass_high < stop_high");
		if (stop_high >= sample_rate / 2.0)
			throw std::invalid_argument("elliptic bandpass: stop_high must be < Nyquist");

		constexpr CoeffScalar pi_T = CoeffScalar(sw::dsp::pi);
		constexpr CoeffScalar one  = CoeffScalar(1);
		constexpr CoeffScalar two  = CoeffScalar(2);
		const CoeffScalar fs = CoeffScalar(sample_rate);
		const CoeffScalar wpl = tan(pi_T * CoeffScalar(pass_low)  / fs);
		const CoeffScalar wph = tan(pi_T * CoeffScalar(pass_high) / fs);
		const CoeffScalar wsl = tan(pi_T * CoeffScalar(stop_low)  / fs);
		const CoeffScalar wsh = tan(pi_T * CoeffScalar(stop_high) / fs);

		const CoeffScalar w0_sq = wpl * wph;
		const CoeffScalar bw = wph - wpl;
		const CoeffScalar kl = abs((wsl * wsl - w0_sq) / (wsl * bw));
		const CoeffScalar kh = abs((wsh * wsh - w0_sq) / (wsh * bw));
		// min() via explicit compare; std::min on posit is ambiguous in some
		// overload sets. Comparison operators are well-defined on DspField.
		const CoeffScalar k_min = (kl < kh) ? kl : kh;
		const CoeffScalar k = one / k_min;

		const CoeffScalar center_freq = (CoeffScalar(pass_low) + CoeffScalar(pass_high)) / two;
		const CoeffScalar width_freq  =  CoeffScalar(pass_high) - CoeffScalar(pass_low);

		EllipticAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design_from_modulus(order, CoeffScalar(ripple_db), k, analog_);
		BandPassTransform<CoeffScalar>(center_freq / fs, width_freq / fs, digital_, analog_);
		cascade_.set_layout(digital_);
	}

	const Cascade<CoeffScalar, max_stages>& cascade() const { return cascade_; }

private:
	PoleZeroLayout<CoeffScalar, MaxOrder> analog_;
	PoleZeroLayout<CoeffScalar, MaxOrder * 2> digital_;
	Cascade<CoeffScalar, max_stages> cascade_;
};

// Bandstop with explicit passband/stopband specification.
// setup(order, sample_rate, stop_low, stop_high, pass_low, pass_high, ripple_db, stopband_db)
template <int MaxOrder,
          DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class EllipticBandStopSpec {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = MaxOrder;

	void setup(int order, double sample_rate,
	           double stop_low, double stop_high,
	           double pass_low, double pass_high,
	           double ripple_db, double stopband_db) {
		using std::tan; using std::abs;
		if (ripple_db <= 0 || stopband_db <= 0)
			throw std::invalid_argument("elliptic bandstop: ripple and stopband must be > 0");
		if (!(pass_low < stop_low && stop_low < stop_high && stop_high < pass_high))
			throw std::invalid_argument("elliptic bandstop: need pass_low < stop_low < stop_high < pass_high");
		if (pass_high >= sample_rate / 2.0)
			throw std::invalid_argument("elliptic bandstop: pass_high must be < Nyquist");

		constexpr CoeffScalar pi_T = CoeffScalar(sw::dsp::pi);
		constexpr CoeffScalar one  = CoeffScalar(1);
		constexpr CoeffScalar two  = CoeffScalar(2);
		const CoeffScalar fs = CoeffScalar(sample_rate);
		const CoeffScalar wpl = tan(pi_T * CoeffScalar(pass_low)  / fs);
		const CoeffScalar wph = tan(pi_T * CoeffScalar(pass_high) / fs);
		const CoeffScalar wsl = tan(pi_T * CoeffScalar(stop_low)  / fs);
		const CoeffScalar wsh = tan(pi_T * CoeffScalar(stop_high) / fs);

		// Bandstop selectivity: for each passband edge, compute its
		// equivalent lowpass prototype frequency, then take the tightest.
		const CoeffScalar w0_sq = wsl * wsh;
		const CoeffScalar bw = wsh - wsl;
		const CoeffScalar kl = abs((wpl * wpl - w0_sq) / (wpl * bw));
		const CoeffScalar kh = abs((wph * wph - w0_sq) / (wph * bw));
		const CoeffScalar k_max = (kl > kh) ? kl : kh;
		const CoeffScalar k = one / k_max;

		const CoeffScalar center_freq = (CoeffScalar(stop_low) + CoeffScalar(stop_high)) / two;
		const CoeffScalar width_freq  =  CoeffScalar(stop_high) - CoeffScalar(stop_low);

		EllipticAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design_from_modulus(order, CoeffScalar(ripple_db), k, analog_);
		BandStopTransform<CoeffScalar>(center_freq / fs, width_freq / fs, digital_, analog_);
		cascade_.set_layout(digital_);
	}

	const Cascade<CoeffScalar, max_stages>& cascade() const { return cascade_; }

private:
	PoleZeroLayout<CoeffScalar, MaxOrder> analog_;
	PoleZeroLayout<CoeffScalar, MaxOrder * 2> digital_;
	Cascade<CoeffScalar, max_stages> cascade_;
};

} // namespace sw::dsp::iir
