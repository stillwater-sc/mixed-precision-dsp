#pragma once
// dolph_chebyshev.hpp: Dolph-Chebyshev window
//
// Equiripple sidelobes at a specified attenuation level. Optimal in the
// sense of narrowest main lobe for a given sidelobe level.
//
// Construction:
//   W[k] = (-1)^k * T_{N-1}(x0 * cos(pi*k/N)) / atten
// where T_n is the Chebyshev polynomial, x0 = cosh(acosh(atten)/(N-1)),
// and atten = 10^(attenuation_db / 20). The time-domain window w[n] is
// the inverse DFT of W.
//
// All intermediate math runs in T so non-native CoeffScalar callers get
// design-time computation at their declared precision. For |arg| <= 1
// the Chebyshev polynomial uses the stable three-term recurrence
//   T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
// which avoids trig entirely. For |arg| > 1 the cosh/acosh form is used
// (ADL picks up sw::universal::{cosh,acosh} for Universal types); the
// recurrence grows exponentially there and would overflow narrow types.
//
// The IDFT kernel is a direct-form O(N^2) sum — acceptable for
// design-time window generation.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>

namespace sw::dsp {

namespace detail {

// Chebyshev polynomial of the first kind, T_n(x), evaluated in T.
// |x| <= 1: three-term recurrence (no trig, numerically stable)
// |x| >  1: cosh(n * acosh(|x|)) (stable for large x; recurrence would overflow)
template <DspField T>
T chebyshev_poly(int n, const T& x) {
	using std::abs; using std::cosh; using std::acosh;
	constexpr T zero = T(0);
	constexpr T one  = T(1);
	constexpr T two  = T(2);
	if (n == 0) return one;
	if (n == 1) return x;

	if (abs(x) <= one) {
		// Stable recurrence T_n = 2x*T_{n-1} - T_{n-2}
		T tkm2 = one;
		T tkm1 = x;
		T tk   = zero;
		for (int k = 2; k <= n; ++k) {
			tk   = two * x * tkm1 - tkm2;
			tkm2 = tkm1;
			tkm1 = tk;
		}
		return tk;
	}

	// |x| > 1: use the hyperbolic form (stable for large magnitudes)
	const T ax  = abs(x);
	const T val = cosh(T(n) * acosh(ax));
	return (x < zero && (n % 2 != 0)) ? -val : val;
}

} // namespace detail

template <DspField T>
mtl::vec::dense_vector<T> dolph_chebyshev_window(std::size_t length,
                                                  double attenuation_db = 100.0) {
	using std::cos; using std::cosh; using std::acosh; using std::pow; using std::abs;
	// attenuation_db must be finite and strictly positive: at 0 the design is
	// degenerate (all sidelobes equal to the main lobe), a negative value
	// makes atten_linear < 1 and passes an out-of-domain argument to acosh,
	// and NaN/inf would silently produce NaN taps downstream.
	if (!(std::isfinite(attenuation_db) && attenuation_db > 0.0))
		throw std::invalid_argument(
			"dolph_chebyshev_window: attenuation_db must be finite and > 0");
	mtl::vec::dense_vector<T> w(length);
	if (length <= 1) { if (length == 1) w[0] = T(1); return w; }
	// length flows through int N (used as the order argument to
	// chebyshev_poly and as a loop bound). Guard the narrowing cast so
	// pathologically large inputs fail loudly instead of silently
	// overflowing N and corrupting allocations.
	if (length > static_cast<std::size_t>(std::numeric_limits<int>::max()))
		throw std::invalid_argument(
			"dolph_chebyshev_window: length exceeds INT_MAX");

	const int N = static_cast<int>(length);
	const int order = N - 1;

	constexpr T zero = T(0);
	constexpr T half = T(0.5);
	constexpr T ten  = T(10);
	constexpr T twenty = T(20);
	constexpr T pi_T = T(pi);
	constexpr T two_pi_T = T(two_pi);

	const T atten_db     = T(attenuation_db);
	const T atten_linear = pow(ten, atten_db / twenty);
	const T x0 = cosh(acosh(atten_linear) / T(order));

	// Frequency-domain window W[k] = |T_{N-1}(x0 * cos(πk/N))| / atten_linear.
	//
	// Two subtleties packed into that formula:
	//
	//   1. The absolute value makes W SYMMETRIC around k=N/2 for any N.
	//      Without it, W[N-k] = T_{N-1}(-x0 cos(πk/N)) = (-1)^{N-1} W[k],
	//      which is symmetric only for odd N. Taking |·| forces symmetry
	//      universally — the DSP interpretation is that we care about the
	//      magnitude spectrum |W(f)| with equiripple sidelobes (the standard
	//      Dolph-Chebyshev design goal), not the signed polynomial.
	//
	//   2. NO (-1)^k twist here. Earlier versions of this file applied a
	//      sign = (-1)^k factor intending to shift the IDFT output from
	//      n=0 to n=N/2 (fftshift-in-frequency-domain). That combined with
	//      the T_{N-1} parity property gave W[N-k] = -W[k] (anti-symmetric),
	//      and the cos-only IDFT of an anti-symmetric spectrum cancels
	//      term-by-term at every n — producing a near-constant output that
	//      normalized to 1.0 (issue #200). The fix here is to build a
	//      symmetric magnitude W and fftshift AFTER the transform.
	using std::abs;
	mtl::vec::dense_vector<T> W(static_cast<std::size_t>(N));
	for (int k = 0; k < N; ++k) {
		const T arg = x0 * cos(pi_T * T(k) / T(N));
		W[static_cast<std::size_t>(k)] =
			abs(detail::chebyshev_poly(order, arg)) / atten_linear;
	}

	// Inverse DFT (direct-form O(N^2) sum) — real-valued because W is
	// real and symmetric. Peak lands at n=0; we fftshift below.
	mtl::vec::dense_vector<T> wn_raw(static_cast<std::size_t>(N));
	for (int n = 0; n < N; ++n) {
		T sum = zero;
		for (int k = 0; k < N; ++k) {
			sum = sum + W[static_cast<std::size_t>(k)] *
				cos(two_pi_T * T(k) * T(n) / T(N));
		}
		wn_raw[static_cast<std::size_t>(n)] = sum;
	}

	// fftshift: rotate wn_raw by N/2 so the peak lands at the middle
	// index (n = N/2 for even N, n = (N-1)/2 for odd N). For any n:
	//   wn[n] = wn_raw[(n + N/2) mod N]
	// This is the standard numpy.fft.fftshift for a length-N array.
	const std::size_t shift = static_cast<std::size_t>(N) / 2;
	mtl::vec::dense_vector<T> wn(static_cast<std::size_t>(N));
	for (int n = 0; n < N; ++n) {
		const std::size_t src =
			(static_cast<std::size_t>(n) + shift) % static_cast<std::size_t>(N);
		wn[static_cast<std::size_t>(n)] = wn_raw[src];
	}

	// Enforce symmetry (a small IDFT rounding asymmetry can accumulate)
	// then renormalize to unity peak.
	for (int n = 0; n < N / 2; ++n) {
		const std::size_t i_lo = static_cast<std::size_t>(n);
		const std::size_t i_hi = static_cast<std::size_t>(N - 1 - n);
		const T avg = half * (wn[i_lo] + wn[i_hi]);
		wn[i_lo] = avg;
		wn[i_hi] = avg;
	}
	T max_val = zero;
	for (int n = 0; n < N; ++n) {
		const T v = abs(wn[static_cast<std::size_t>(n)]);
		if (v > max_val) max_val = v;
	}
	// Defensive: a peak of zero (or NaN-via-comparison-failure) means the
	// IDFT collapsed — extreme attenuation/length combinations could
	// underflow narrow types. Surface that as a clear error rather than
	// produce all-NaN/inf output.
	if (!(max_val > zero))
		throw std::runtime_error(
			"dolph_chebyshev_window: normalization peak is zero or invalid");
	for (int n = 0; n < N; ++n) {
		w[static_cast<std::size_t>(n)] =
			wn[static_cast<std::size_t>(n)] / max_val;
	}
	return w;
}

} // namespace sw::dsp
