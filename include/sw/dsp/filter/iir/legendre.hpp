#pragma once
// legendre.hpp: Optimum-L (Legendre) IIR filter design
//
// Steepest possible transition band with monotonic passband.
// Based on Papoulis' method using Legendre polynomial recursion.
//
// Reference: Kuo, "Network Analysis and Synthesis", pp. 379-383.
// Original method: Papoulis, "On Monotonic Response Filters", Proc. IRE, 1959.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <complex>
#include <vector>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/math/root_finder.hpp>
#include <sw/dsp/filter/layout/layout.hpp>
#include <sw/dsp/filter/layout/analog_prototype.hpp>
#include <sw/dsp/filter/biquad/cascade.hpp>
#include <sw/dsp/filter/transform/bilinear.hpp>
#include <sw/dsp/filter/transform/constantinides.hpp>

namespace sw::dsp::iir {

namespace detail {

// Compute Legendre polynomial coefficients using recursion:
//   (n+1)P_{n+1}(x) = (2n+1)*x*P_n(x) - n*P_{n-1}(x)
inline void legendre_poly(std::vector<double>& p, int n,
                           std::vector<double>& aa, std::vector<double>& bb) {
	if (n == 0) { p[0] = 1.0; return; }
	if (n == 1) { p[0] = 0.0; p[1] = 1.0; return; }

	p[0] = -0.5; p[1] = 0.0; p[2] = 1.5;
	if (n == 2) return;

	for (int i = 0; i <= n; ++i) { aa[i] = bb[i] = 0.0; }
	bb[1] = 1.0;

	for (int i = 3; i <= n; ++i) {
		for (int j = 0; j <= i; ++j) {
			aa[j] = bb[j];
			bb[j] = p[j];
			p[j] = 0.0;
		}
		for (int j = i - 2; j >= 0; j -= 2) {
			p[j] -= static_cast<double>(i - 1) * aa[j] / static_cast<double>(i);
		}
		for (int j = i - 1; j >= 0; j -= 2) {
			p[j + 1] += static_cast<double>(2 * i - 1) * bb[j] / static_cast<double>(i);
		}
	}
}

// Solve for the Legendre polynomial-based transfer function coefficients.
// Produces coefficients w[] of degree n+1 that define |H(jw)|^2.
inline std::vector<double> legendre_solve(int n) {
	const int k = (n - 1) / 2;
	const double sqrt2 = 1.41421356237309504880;

	std::vector<double> a(k + 1);
	std::vector<double> p(n + 3, 0.0);
	std::vector<double> s(n + 3, 0.0);
	std::vector<double> w(n + 3, 0.0);
	std::vector<double> v(n + 3, 0.0);  // integration step writes through v[n+2]
	std::vector<double> aa(n + 3, 0.0);
	std::vector<double> bb(n + 3, 0.0);

	// Form vector of 'a' constants
	if (n & 1) {  // odd
		for (int i = 0; i <= k; ++i) {
			a[i] = (2.0 * i + 1.0) / (sqrt2 * (k + 1.0));
		}
	} else {  // even
		for (int i = 0; i <= k; ++i) a[i] = 0.0;
		if (k & 1) {
			for (int i = 1; i <= k; i += 2) {
				a[i] = (2 * i + 1) / std::sqrt(static_cast<double>((k + 1) * (k + 2)));
			}
		} else {
			for (int i = 0; i <= k; i += 2) {
				a[i] = (2 * i + 1) / std::sqrt(static_cast<double>((k + 1) * (k + 2)));
			}
		}
	}

	// Form s[] = sum of a[i]*P_i
	s[0] = a[0];
	s[1] = (k >= 1) ? a[1] : 0.0;
	for (int i = 2; i <= k; ++i) {
		legendre_poly(p, i, aa, bb);
		for (int j = 0; j <= i; ++j) {
			s[j] += a[i] * p[j];
		}
	}

	// Form v[] = square of s[]
	for (int i = 0; i <= 2 * k + 2; ++i) v[i] = 0.0;
	for (int i = 0; i <= k; ++i) {
		for (int j = 0; j <= k; ++j) {
			v[i + j] += s[i] * s[j];
		}
	}

	// Modify integrand for even n
	v[2 * k + 1] = 0.0;
	if ((n & 1) == 0) {
		for (int i = n; i >= 0; --i) {
			v[i + 1] += v[i];
		}
	}

	// Form integral of v[]
	for (int i = n + 1; i >= 0; --i) {
		v[i + 1] = v[i] / static_cast<double>(i + 1);
	}
	v[0] = 0.0;

	// Compute definite integral
	for (int i = 0; i < n + 2; ++i) s[i] = 0.0;
	s[0] = -1.0;
	s[1] = 2.0;

	for (int i = 1; i <= n; ++i) {
		if (i > 1) {
			double c0 = -s[0];
			for (int j = 1; j < i + 1; ++j) {
				double c1 = -s[j] + 2.0 * s[j - 1];
				s[j - 1] = c0;
				c0 = c1;
			}
			double c1 = 2.0 * s[i];
			s[i] = c0;
			s[i + 1] = c1;
		}
		for (int j = i; j > 0; --j) {
			w[j] += v[i] * s[j];
		}
	}
	if ((n & 1) == 0) w[1] = 0.0;

	w.resize(n + 2);
	return w;
}

} // namespace detail

// Legendre (Optimum-L) analog lowpass prototype.
template <DspField T, int MaxOrder>
class LegendreAnalogPrototype {
public:
	void design(int num_poles, PoleZeroLayout<T, MaxOrder>& layout) {
		layout.reset();
		layout.set_normal(T{}, T{1});

		auto w = detail::legendre_solve(num_poles);
		int degree = num_poles * 2;

		// Build polynomial for root finding: 1 + w(s^2)
		// The polynomial in s has degree 2*num_poles
		RootFinder<T, MaxOrder * 2> solver;
		solver.coef(0) = std::complex<T>(static_cast<T>(1.0 + w[0]));
		solver.coef(1) = std::complex<T>(T{});
		for (int i = 1; i <= num_poles; ++i) {
			// w[i] * (s^2)^i = w[i] * s^(2i), alternating sign for s^2 -> -omega^2
			T sign = (i & 1) ? T{-1} : T{1};
			solver.coef(2 * i) = std::complex<T>(static_cast<T>(w[i]) * sign);
			if (2 * i + 1 <= degree) {
				solver.coef(2 * i + 1) = std::complex<T>(T{});
			}
		}
		solver.solve(degree);

		// Keep only left-half-plane poles
		int j = 0;
		for (int i = 0; i < degree; ++i) {
			if (solver.root(i).real() <= T{}) {
				solver.root(j++) = solver.root(i);
			}
		}
		if (j < num_poles) {
			throw std::runtime_error("legendre: failed to isolate all left-half-plane poles");
		}
		// Sort by descending imaginary part
		solver.sort(j);

		const int pairs = num_poles / 2;
		for (int i = 0; i < pairs; ++i) {
			layout.add_conjugate_pairs(solver.root(i), s_infinity<T>());
		}

		if (num_poles & 1) {
			layout.add(
				std::complex<T>(solver.root(pairs).real(), T{}),
				s_infinity<T>());
		}
	}
};

// Legendre filter designs: LP, HP, BP, BS
template <int MaxOrder,
          DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class LegendreLowPass {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = (MaxOrder + 1) / 2;

	void setup(int order, double sample_rate, double cutoff_freq) {
		LegendreAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, analog_);
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
class LegendreHighPass {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = (MaxOrder + 1) / 2;

	void setup(int order, double sample_rate, double cutoff_freq) {
		LegendreAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, analog_);
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
class LegendreBandPass {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = MaxOrder;

	void setup(int order, double sample_rate, double center_freq, double width_freq) {
		LegendreAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, analog_);
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
class LegendreBandStop {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = MaxOrder;

	void setup(int order, double sample_rate, double center_freq, double width_freq) {
		LegendreAnalogPrototype<CoeffScalar, MaxOrder> proto;
		proto.design(order, analog_);
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
