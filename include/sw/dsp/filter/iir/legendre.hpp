#pragma once
// legendre.hpp: Optimum-L (Legendre) IIR filter design
//
// Steepest possible transition band with monotonic passband.
// Based on Papoulis' method using Legendre polynomial recursion.
//
// All polynomial computations are parameterized on CoeffScalar to
// preserve precision when using high-precision arithmetic types.
// Uses std::array for all temporary storage (sizes derived from MaxOrder).
//
// Reference: Kuo, "Network Analysis and Synthesis", pp. 379-383.
// Original method: Papoulis, "On Monotonic Response Filters", Proc. IRE, 1959.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <stdexcept>
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
//
// All arithmetic in type T. N is the max polynomial degree supported.
template <DspField T, int N>
void legendre_poly(std::array<T, N + 3>& p, int n,
                   std::array<T, N + 3>& aa, std::array<T, N + 3>& bb) {
	if (n == 0) { p[0] = T{1}; return; }
	if (n == 1) { p[0] = T{}; p[1] = T{1}; return; }

	p[0] = T{-0.5}; p[1] = T{}; p[2] = T{1.5};
	if (n == 2) return;

	for (int i = 0; i <= n; ++i) { aa[i] = T{}; bb[i] = T{}; }
	bb[1] = T{1};

	for (int i = 3; i <= n; ++i) {
		for (int j = 0; j <= i; ++j) {
			aa[j] = bb[j];
			bb[j] = p[j];
			p[j] = T{};
		}
		T inv_i = T{1} / static_cast<T>(i);
		for (int j = i - 2; j >= 0; j -= 2) {
			p[j] = p[j] - static_cast<T>(i - 1) * aa[j] * inv_i;
		}
		for (int j = i - 1; j >= 0; j -= 2) {
			p[j + 1] = p[j + 1] + static_cast<T>(2 * i - 1) * bb[j] * inv_i;
		}
	}
}

// Solve for the Legendre polynomial-based transfer function coefficients.
// Produces coefficients w[] of degree n+1 that define |H(jw)|^2.
// All arithmetic in type T. N is the max filter order supported.
template <DspField T, int N>
void legendre_solve(int n, std::array<T, N + 3>& w) {
	const int k = (n - 1) / 2;

	// Size all temporaries from N (max order)
	std::array<T, N + 3> a{};
	std::array<T, N + 3> p{};
	std::array<T, N + 3> s{};
	std::array<T, N + 3> v{};  // needs up to index n+2
	std::array<T, N + 3> aa{};
	std::array<T, N + 3> bb{};

	// Form vector of 'a' constants
	if (n & 1) {  // odd
		T denom = sqrt2_v<T> * static_cast<T>(k + 1);
		for (int i = 0; i <= k; ++i) {
			a[i] = static_cast<T>(2 * i + 1) / denom;
		}
	} else {  // even
		T denom = static_cast<T>(std::sqrt(static_cast<double>(
			static_cast<T>(k + 1) * static_cast<T>(k + 2))));
		if (k & 1) {
			for (int i = 1; i <= k; i += 2) {
				a[i] = static_cast<T>(2 * i + 1) / denom;
			}
		} else {
			for (int i = 0; i <= k; i += 2) {
				a[i] = static_cast<T>(2 * i + 1) / denom;
			}
		}
	}

	// Form s[] = sum of a[i]*P_i
	s[0] = a[0];
	if (k >= 1) s[1] = a[1];
	for (int i = 2; i <= k; ++i) {
		legendre_poly<T, N>(p, i, aa, bb);
		for (int j = 0; j <= i; ++j) {
			s[j] = s[j] + a[i] * p[j];
		}
	}

	// Form v[] = square of s[]
	for (int i = 0; i <= k; ++i) {
		for (int j = 0; j <= k; ++j) {
			v[i + j] = v[i + j] + s[i] * s[j];
		}
	}

	// Modify integrand for even n
	v[2 * k + 1] = T{};
	if ((n & 1) == 0) {
		for (int i = n; i >= 0; --i) {
			v[i + 1] = v[i + 1] + v[i];
		}
	}

	// Form integral of v[]
	for (int i = n + 1; i >= 0; --i) {
		v[i + 1] = v[i] / static_cast<T>(i + 1);
	}
	v[0] = T{};

	// Compute definite integral
	for (int i = 0; i < n + 2; ++i) s[i] = T{};
	s[0] = T{-1};
	s[1] = T{2};

	for (auto& wi : w) wi = T{};

	for (int i = 1; i <= n; ++i) {
		if (i > 1) {
			T c0 = T{} - s[0];
			for (int j = 1; j < i + 1; ++j) {
				T c1 = T{} - s[j] + T{2} * s[j - 1];
				s[j - 1] = c0;
				c0 = c1;
			}
			T c1 = T{2} * s[i];
			s[i] = c0;
			s[i + 1] = c1;
		}
		for (int j = i; j > 0; --j) {
			w[j] = w[j] + v[i] * s[j];
		}
	}
	if ((n & 1) == 0) w[1] = T{};
}

} // namespace detail

// Legendre (Optimum-L) analog lowpass prototype.
template <DspField T, int MaxOrder>
class LegendreAnalogPrototype {
public:
	void design(int num_poles, PoleZeroLayout<T, MaxOrder>& layout) {
		layout.reset();
		layout.set_normal(T{}, T{1});

		std::array<T, MaxOrder + 3> w{};
		detail::legendre_solve<T, MaxOrder>(num_poles, w);
		int degree = num_poles * 2;

		// Build polynomial for root finding: 1 + w(s^2)
		RootFinder<T, MaxOrder * 2> solver;
		solver.coef(0) = std::complex<T>(T{1} + w[0]);
		solver.coef(1) = std::complex<T>(T{});
		for (int i = 1; i <= num_poles; ++i) {
			T sign = (i & 1) ? T{-1} : T{1};
			solver.coef(2 * i) = std::complex<T>(w[i] * sign);
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
