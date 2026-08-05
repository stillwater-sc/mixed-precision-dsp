#pragma once
// remez.hpp: Parks-McClellan (Remez exchange) equiripple FIR design
//
// Optimal equiripple FIR filter design using the Remez exchange
// algorithm with barycentric Lagrange interpolation. Supports
// bandpass, differentiator, and Hilbert transformer modes.
//
// The algorithm is implemented internally in double precision
// (design-time computation); output taps are projected to T.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>

namespace sw::dsp {

enum class RemezBandType { bandpass, differentiator, hilbert };

// How far the worst weighted grid error may exceed the converged ripple
// before remez() declares the design a failure. A converged exchange lands
// at a ratio of 1.0 to within rounding; the failures this guards against
// overshoot by many orders of magnitude, so the threshold only has to
// separate "converged" from "diverged", not grade near-misses.
inline constexpr double remez_convergence_tol = 1.05;

// Worst weighted error, relative to the problem scale, below which a design
// is accepted without proving it equiripple. Specifications the exchange can
// solve almost exactly drive delta into rounding noise, and the alternation
// test then compares against a meaningless number. 1e-4 is roughly -80 dB:
// well past anything a design that is actually broken could reach — the
// non-convergence this guards against overshoots by four or more orders of
// magnitude — while still admitting the good-but-not-provably-optimal
// designs the exchange returns in that regime.
inline constexpr double remez_exactness_floor = 1e-4;

namespace detail {

// The four linear-phase FIR types.
//
// Every type realizes its zero-phase amplitude as
//
//     A(f) = q(f) * P(cos 2*pi*f)
//
// for a type-specific factor q and a cosine polynomial P. Only Type I has
// q == 1. The Remez exchange solves for a cosine polynomial, so the other
// three are made tractable by folding q into the problem statement:
//
//     W(f)*(D(f) - A(f)) = (W(f)*q(f)) * (D(f)/q(f) - P(f))
//
// i.e. run the exchange on W' = W*q and D' = D/q, recover P, then reapply
// q analytically when converting P's coefficients to taps.
//
//   type  taps  symmetry       q(f)          deg P        center
//   I     odd   symmetric      1             (N-1)/2      sample
//   II    even  symmetric      cos(pi*f)     N/2 - 1      half-sample
//   III   odd   antisymmetric  sin(2*pi*f)   (N-1)/2 - 1  sample (zero)
//   IV    even  antisymmetric  sin(pi*f)     N/2 - 1      half-sample
//
// (issue #205)
enum class LinearPhaseType { I, II, III, IV };

inline double basis_factor(LinearPhaseType t, double f) {
	switch (t) {
		case LinearPhaseType::I:   return 1.0;
		case LinearPhaseType::II:  return std::cos(pi * f);
		case LinearPhaseType::III: return std::sin(two_pi * f);
		default:                   return std::sin(pi * f);
	}
}

// q vanishes at f = 0 for the antisymmetric types and at f = 0.5 for
// Types II and III. D/q is singular there, so those endpoints must be kept
// off the grid.
inline bool basis_vanishes_at_zero(LinearPhaseType t) {
	return t == LinearPhaseType::III || t == LinearPhaseType::IV;
}
inline bool basis_vanishes_at_nyquist(LinearPhaseType t) {
	return t == LinearPhaseType::II || t == LinearPhaseType::III;
}

// Build a dense frequency grid covering all specified bands.
// Grid density controls the number of points per band per tap.
//
// The grid is a concatenation of per-band segments and is DISCONTINUOUS
// across transition bands. `band_start` (optional out-param) receives the
// grid index at which each band begins, plus a trailing sentinel equal to
// grid.size(), so callers can walk band [band_start[b], band_start[b+1])
// without straddling a transition gap.
inline std::vector<double> build_grid(
    const std::vector<double>& bands,
    std::size_t num_taps,
    int grid_density,
    RemezBandType type,
    LinearPhaseType lp_type,
    std::vector<std::size_t>* band_start = nullptr) {

	std::size_t num_bands = bands.size() / 2;
	std::vector<double> grid;
	grid.reserve(num_taps * static_cast<std::size_t>(grid_density) * num_bands);
	if (band_start) {
		band_start->clear();
		band_start->reserve(num_bands + 1);
	}

	// One grid step, used to pull the grid off the frequencies where the
	// type's basis factor vanishes and D/q would blow up.
	const double delf = 0.5 / static_cast<double>(
	    static_cast<std::size_t>(grid_density) * num_taps);

	for (std::size_t b = 0; b < num_bands; ++b) {
		double f_start = bands[2 * b];
		double f_end   = bands[2 * b + 1];

		// Avoid exact 0 for differentiator (weight function has 1/f singularity)
		if (type == RemezBandType::differentiator && f_start < 1e-6)
			f_start = 1e-6;

		// Keep the basis-factor zeros off the grid (issue #205)
		if (basis_vanishes_at_zero(lp_type) && f_start < delf)
			f_start = delf;
		if (basis_vanishes_at_nyquist(lp_type) && f_end > 0.5 - delf)
			f_end = 0.5 - delf;
		if (!(f_end > f_start)) continue;   // band collapsed by the clamp

		std::size_t npts = std::max<std::size_t>(
		    static_cast<std::size_t>(grid_density * num_taps * (f_end - f_start) / 0.5), 4);

		if (band_start) band_start->push_back(grid.size());
		for (std::size_t i = 0; i < npts; ++i) {
			double f = f_start + static_cast<double>(i) / static_cast<double>(npts - 1) * (f_end - f_start);
			grid.push_back(f);
		}
	}
	if (band_start) band_start->push_back(grid.size());
	return grid;
}

// Evaluate the desired response and weight at a frequency for a given band.
inline void eval_desired_weight(
    double freq,
    const std::vector<double>& bands,
    const std::vector<double>& desired,
    const std::vector<double>& weights,
    RemezBandType type,
    double& des_val,
    double& wt_val) {

	std::size_t num_bands = bands.size() / 2;
	for (std::size_t b = 0; b < num_bands; ++b) {
		if (freq >= bands[2 * b] - 1e-10 && freq <= bands[2 * b + 1] + 1e-10) {
			double f0 = bands[2 * b];
			double f1 = bands[2 * b + 1];
			double d0 = desired[2 * b];
			double d1 = desired[2 * b + 1];

			// Linear interpolation of desired response within band
			double t = (f1 > f0) ? (freq - f0) / (f1 - f0) : 0.0;
			des_val = d0 + t * (d1 - d0);
			wt_val = weights[b];

			if (type == RemezBandType::differentiator) {
				// Weight function is 1/f for differentiator
				if (freq > 1e-10)
					wt_val /= freq;
			}
			return;
		}
	}
	des_val = 0.0;
	wt_val = 1.0;
}

// Barycentric Lagrange interpolation at frequency x using the
// current extremal set. Returns the interpolated value.
// Also computes delta (the equiripple deviation).
inline double lagrange_interp(
    const std::vector<double>& extremal_freqs,
    const std::vector<double>& extremal_des,
    const std::vector<double>& extremal_wt,
    double x,
    double delta) {

	std::size_t n = extremal_freqs.size();
	double numer = 0.0, denom = 0.0;

	for (std::size_t i = 0; i < n; ++i) {
		double cos_ext = std::cos(two_pi * extremal_freqs[i]);
		double cos_x   = std::cos(two_pi * x);
		double diff = cos_x - cos_ext;

		if (std::abs(diff) < 1e-15)
			return extremal_des[i] + (((i & 1) == 0) ? 1.0 : -1.0) * delta / extremal_wt[i];

		// Barycentric weight: alternating sign / product of differences
		double bary = 1.0;
		for (std::size_t j = 0; j < n; ++j) {
			if (j != i) {
				double cos_j = std::cos(two_pi * extremal_freqs[j]);
				bary *= (cos_x - cos_j);
			}
		}
		if (std::abs(bary) < 1e-300) continue;

		double c = 1.0 / bary;
		double val = extremal_des[i] + (((i & 1) == 0) ? 1.0 : -1.0) * delta / extremal_wt[i];
		numer += c * val;
		denom += c;
	}

	return (std::abs(denom) > 1e-300) ? numer / denom : 0.0;
}

// Scale factor folded into every node difference when forming barycentric
// weights.
//
// A weight is 1 / prod_{j!=i} (x_i - x_j) over n-1 factors. On nodes spread
// over an interval of width W those products run like (W/4)^(n-1), so the
// raw weights grow or shrink exponentially in n: 65 nodes packed into a
// narrow band drive them past 1e80, and any quantity formed as a ratio of
// alternating-sign sums of numbers that large is pure cancellation noise.
// Scaling each difference by 4/W normalizes the product to O(1) regardless
// of n and of how wide the nodes actually spread. The classic
// Parks-McClellan factor of 2 is this expression for W = 2, the full range
// of cos over [0, 0.5]; using the actual node range instead is what keeps
// narrow-band and high-tap designs conditioned. The factor is common to
// every weight and appears only in ratios, so it cancels exactly and
// changes nothing but the exponent range. (issues #203, #205)
inline double barycentric_scale(const std::vector<double>& x, std::size_t n) {
	if (n < 2) return 2.0;
	double lo = x[0], hi = x[0];
	for (std::size_t i = 1; i < n; ++i) {
		lo = std::min(lo, x[i]);
		hi = std::max(hi, x[i]);
	}
	const double width = hi - lo;
	return (width > 1e-12) ? 4.0 / width : 2.0;
}

// Compute delta (equiripple deviation) from the current extremal set
// using the Remez formula.
//
// Convention: eval_approx() below builds the interpolant through the
// reference values
//
//     v_i = D_i + (-1)^i * delta / W_i .
//
// A polynomial of degree <= n-2 interpolates all n reference points only
// if the order-(n-1) divided difference of those values vanishes, i.e.
// sum_i b_i v_i = 0 with b_i the barycentric weights over all n points:
//
//     sum_i b_i D_i + delta * sum_i (-1)^i b_i / W_i = 0
//     delta = -sum_i b_i D_i / sum_i (-1)^i b_i / W_i .
//
// The leading minus sign is what makes delta consistent with the "+"
// in eval_approx. Returning +numer/denom instead yields a delta of the
// wrong sign, so the interpolant misses the LAST reference point: the
// error curve then alternates only n-1 times instead of n, the extremal
// search can never assemble a full alternating set, and the exchange
// freezes on its first reference set for every remaining iteration.
// (issue #203)
inline double compute_delta(
    const std::vector<double>& extremal_des,
    const std::vector<double>& extremal_wt,
    const std::vector<double>& extremal_cos,
    std::size_t n_extremals) {

	// Compute barycentric weights for the Chebyshev interpolation.
	// The per-difference scaling is required, not cosmetic — see the note on
	// bary_scale in remez() below.
	const double bary_scale = barycentric_scale(extremal_cos, n_extremals);
	std::vector<double> bary(n_extremals);
	for (std::size_t i = 0; i < n_extremals; ++i) {
		double prod = 1.0;
		for (std::size_t j = 0; j < n_extremals; ++j) {
			if (j != i) {
				double diff = extremal_cos[i] - extremal_cos[j];
				if (std::abs(diff) < 1e-15) diff = 1e-15;
				prod *= bary_scale * diff;
			}
		}
		bary[i] = 1.0 / prod;
	}

	double numer = 0.0, denom = 0.0;
	for (std::size_t i = 0; i < n_extremals; ++i) {
		numer += bary[i] * extremal_des[i];
		double sign = ((i & 1) == 0) ? 1.0 : -1.0;
		denom += sign * bary[i] / extremal_wt[i];
	}

	return -numer / denom;
}

// Evaluate the current polynomial approximation at cos(2*pi*f)
// using barycentric interpolation from the extremal set.
inline double eval_approx(
    double freq,
    const std::vector<double>& /*extremal_freqs*/,
    const std::vector<double>& extremal_des,
    const std::vector<double>& extremal_wt,
    const std::vector<double>& extremal_cos,
    const std::vector<double>& bary_weights,
    double delta,
    std::size_t n_poly) {

	double cos_f = std::cos(two_pi * freq);
	double numer = 0.0, denom = 0.0;

	for (std::size_t i = 0; i < n_poly; ++i) {
		double diff = cos_f - extremal_cos[i];
		if (std::abs(diff) < 1e-15) {
			double sign = ((i & 1) == 0) ? 1.0 : -1.0;
			return extremal_des[i] + sign * delta / extremal_wt[i];
		}
		double c = bary_weights[i] / diff;
		double sign = ((i & 1) == 0) ? 1.0 : -1.0;
		double val = extremal_des[i] + sign * delta / extremal_wt[i];
		numer += c * val;
		denom += c;
	}

	return (std::abs(denom) > 1e-300) ? numer / denom : 0.0;
}

} // namespace detail

// remez: Parks-McClellan optimal equiripple FIR filter design.
//
// Parameters:
//   num_taps      — number of filter taps (filter length)
//   bands         — band edge frequencies, normalized [0, 0.5]
//                   pairs: [f1,f2, f3,f4, ...] (even number of values)
//   desired       — desired gain at each band edge [d1,d2, d3,d4, ...]
//   weights       — weight per band [w1, w2, ...] (one per band)
//   type          — bandpass (default), differentiator, or hilbert
//   max_iterations — convergence limit
//   grid_density  — grid points per tap per band
//
// Returns: dense_vector<T> of filter tap coefficients.
template <DspField T>
mtl::vec::dense_vector<T> remez(
    std::size_t num_taps,
    const std::vector<T>& bands,
    const std::vector<T>& desired,
    const std::vector<T>& weights,
    RemezBandType type = RemezBandType::bandpass,
    int max_iterations = 40,
    int grid_density = 16) {

	if (num_taps < 3)
		throw std::invalid_argument("remez: num_taps must be >= 3");
	if (bands.size() < 2 || (bands.size() & 1) != 0)
		throw std::invalid_argument("remez: bands must have even number of elements");
	if (desired.size() != bands.size())
		throw std::invalid_argument("remez: desired must have same size as bands");
	if (weights.size() * 2 != bands.size())
		throw std::invalid_argument("remez: weights must have one entry per band");
	if (max_iterations <= 0)
		throw std::invalid_argument("remez: max_iterations must be > 0");
	if (grid_density <= 0)
		throw std::invalid_argument("remez: grid_density must be > 0");
	for (std::size_t i = 0; i < bands.size(); ++i) {
		if (bands[i] < T{0} || bands[i] > T(0.5))
			throw std::invalid_argument("remez: band edges must be in [0, 0.5]");
		if (i > 0 && bands[i] < bands[i - 1])
			throw std::invalid_argument("remez: band edges must be nondecreasing");
	}
	for (const auto& w : weights) {
		if (!(w > T{0}))
			throw std::invalid_argument("remez: weights must be > 0");
	}

	// The Remez exchange is a design-time computation requiring high
	// dynamic range for the barycentric interpolation. Internal math
	// uses double; output taps are projected to T.
	// (See v0.5.0-implementation-plan.md: "must be implemented in
	// double internally")
	std::vector<double> d_bands(bands.size()), d_desired(desired.size()), d_weights(weights.size());
	for (std::size_t i = 0; i < bands.size(); ++i) d_bands[i] = static_cast<double>(bands[i]);
	for (std::size_t i = 0; i < desired.size(); ++i) d_desired[i] = static_cast<double>(desired[i]);
	for (std::size_t i = 0; i < weights.size(); ++i) d_weights[i] = static_cast<double>(weights[i]);

	// Determine symmetry
	// Type I (odd taps) and Type II (even taps) for bandpass
	// Type III (odd taps) and Type IV (even taps) for differentiator/hilbert
	bool is_symmetric = (type == RemezBandType::bandpass);
	bool is_odd = (num_taps & 1) != 0;

	// Linear-phase type, and with it the basis factor q(f) and the degree
	// of the cosine polynomial P the exchange actually solves for. See the
	// LinearPhaseType comment above. (issue #205)
	const detail::LinearPhaseType lp_type =
	    is_symmetric ? (is_odd ? detail::LinearPhaseType::I   : detail::LinearPhaseType::II)
	                 : (is_odd ? detail::LinearPhaseType::III : detail::LinearPhaseType::IV);

	std::size_t L; // degree of the cosine polynomial P
	switch (lp_type) {
		case detail::LinearPhaseType::I:   L = (num_taps - 1) / 2;     break;
		case detail::LinearPhaseType::II:  L = num_taps / 2 - 1;       break;
		case detail::LinearPhaseType::III: L = (num_taps - 1) / 2 - 1; break;
		default:                           L = num_taps / 2 - 1;       break;
	}
	std::size_t n_extremals = L + 2;

	// Build dense frequency grid
	std::vector<std::size_t> band_start;
	auto grid = detail::build_grid(d_bands, num_taps, grid_density, type,
	                               lp_type, &band_start);
	std::size_t grid_size = grid.size();

	if (grid_size < n_extremals)
		throw std::invalid_argument("remez: grid too sparse for the given num_taps");

	// Compute desired and weight for each grid point, then fold the type's
	// basis factor in: the exchange approximates P = A/q, so it must see
	// D' = D/q against W' = W*q. q is bounded away from zero on the grid
	// because build_grid() excluded the frequencies where it vanishes.
	// (issue #205)
	std::vector<double> grid_des(grid_size), grid_wt(grid_size);
	for (std::size_t i = 0; i < grid_size; ++i) {
		detail::eval_desired_weight(grid[i], d_bands, d_desired, d_weights, type,
		                            grid_des[i], grid_wt[i]);
		const double q = detail::basis_factor(lp_type, grid[i]);
		grid_des[i] /= q;
		grid_wt[i]  *= q;
	}

	// Scale of the problem: the largest weighted desired magnitude on the
	// grid. Absolute error thresholds below are expressed relative to it so
	// they mean the same thing for a unit-gain lowpass and for a
	// differentiator whose desired response runs to 0.5.
	double problem_scale = 0.0;
	for (std::size_t i = 0; i < grid_size; ++i)
		problem_scale = std::max(problem_scale, std::abs(grid_wt[i] * grid_des[i]));
	if (!(problem_scale > 0.0)) problem_scale = 1.0;

	// Initialize extremal set with uniform spacing across grid
	std::vector<std::size_t> extremal_idx(n_extremals);
	for (std::size_t i = 0; i < n_extremals; ++i) {
		extremal_idx[i] = i * (grid_size - 1) / (n_extremals - 1);
	}

	// Remez exchange iteration
	double delta = 0.0;

	// Best iterate seen, by worst weighted grid error.
	//
	// The exchange is not monotone. On specifications it can solve almost
	// exactly — a wide-band Hilbert transformer with a generous tap budget,
	// say — delta collapses toward rounding noise, after which the error
	// curve the extremal search reads is partly noise, the reference set can
	// stampede into a narrow cluster, and the interpolation over
	// near-coincident nodes blows up. Keeping the best iterate means a later
	// divergence cannot destroy an already-good answer, which is what lets
	// the loop keep running in that regime instead of having to bail out
	// early and settle for a worse design. (issue #205)
	// (No best_delta: the final delta is re-derived from whichever reference
	// set is chosen, so storing it here would be redundant.)
	std::vector<std::size_t> best_idx = extremal_idx;
	double best_max_err = std::numeric_limits<double>::infinity();
	bool   have_best    = false;

	for (int iter = 0; iter < max_iterations; ++iter) {
		// Extract extremal frequencies, desired values, weights, and cosines
		std::vector<double> ext_freq(n_extremals), ext_des(n_extremals),
		                    ext_wt(n_extremals), ext_cos(n_extremals);
		for (std::size_t i = 0; i < n_extremals; ++i) {
			ext_freq[i] = grid[extremal_idx[i]];
			ext_des[i]  = grid_des[extremal_idx[i]];
			ext_wt[i]   = grid_wt[extremal_idx[i]];
			ext_cos[i]  = std::cos(two_pi * ext_freq[i]);
		}
		// The (reference set, delta) pair measured below is this one, so
		// snapshot it before the exchange replaces it further down.
		std::vector<std::size_t> this_idx = extremal_idx;

		// Compute delta (equiripple deviation)
		double new_delta = detail::compute_delta(ext_des, ext_wt, ext_cos, n_extremals);

		// Compute barycentric weights for the polynomial (excluding last
		// extremal). See barycentric_scale() for why the scaling is required.
		std::size_t n_poly = n_extremals - 1;
		const double bary_scale = detail::barycentric_scale(ext_cos, n_poly);
		std::vector<double> bary_weights(n_poly);
		for (std::size_t i = 0; i < n_poly; ++i) {
			double prod = 1.0;
			for (std::size_t j = 0; j < n_poly; ++j) {
				if (j != i) {
					double diff = ext_cos[i] - ext_cos[j];
					if (std::abs(diff) < 1e-15) diff = (i < j) ? -1e-15 : 1e-15;
					prod *= bary_scale * diff;
				}
			}
			bary_weights[i] = 1.0 / prod;
		}

		// Evaluate error on entire grid and find new extremals
		std::vector<double> error(grid_size);
		double max_err = 0.0;
		for (std::size_t i = 0; i < grid_size; ++i) {
			double approx = detail::eval_approx(grid[i], ext_freq, ext_des, ext_wt,
			                                     ext_cos, bary_weights, new_delta, n_poly);
			error[i] = grid_wt[i] * (grid_des[i] - approx);
			max_err = std::max(max_err, std::abs(error[i]));
		}

		if (std::isfinite(max_err) && std::isfinite(new_delta) && max_err < best_max_err) {
			best_idx     = this_idx;
			best_max_err = max_err;
			have_best    = true;
		}

		// Find local extrema of the error function, BAND BY BAND.
		//
		// The grid is discontinuous across transition bands, so comparing a
		// band's last point against the next band's first point is comparing
		// across a gap and says nothing about either being extremal. Every
		// band edge is therefore an unconditional candidate — in a
		// Parks-McClellan solution the error is extremal at every band edge,
		// and the alternation filter below discards any candidate that loses
		// on magnitude to a same-sign neighbour. Comparing across the gap
		// instead is what let the band edges escape the extremal set, leaving
		// the largest errors in the design exactly where they were never
		// controlled. (issue #203)
		std::vector<std::size_t> new_extremals;
		new_extremals.reserve(n_extremals * 2);

		for (std::size_t b = 0; b + 1 < band_start.size(); ++b) {
			std::size_t s = band_start[b];
			std::size_t e = band_start[b + 1];   // one past the band's last point
			if (e <= s) continue;
			--e;                                 // band's last grid index
			new_extremals.push_back(s);          // band start: always a candidate
			for (std::size_t i = s + 1; i < e; ++i) {
				if ((error[i] >= error[i-1] && error[i] >= error[i+1]) ||
				    (error[i] <= error[i-1] && error[i] <= error[i+1]))
					new_extremals.push_back(i);
			}
			if (e > s) new_extremals.push_back(e);  // band end: always a candidate
		}

		// Select n_extremals extrema with alternating sign, sorted by frequency.
		// The standard Remez approach: keep all local extrema sorted by
		// frequency, then trim from the ends or interior to get exactly
		// n_extremals with alternating signs.
		if (new_extremals.size() >= n_extremals) {
			// Already sorted by frequency (grid index)
			// Ensure alternating sign: walk through and remove violations
			std::vector<std::size_t> alt;
			alt.reserve(new_extremals.size());
			alt.push_back(new_extremals[0]);

			for (std::size_t i = 1; i < new_extremals.size(); ++i) {
				bool same_sign = (error[new_extremals[i]] > 0) == (error[alt.back()] > 0);
				if (same_sign) {
					// Keep the larger-magnitude one
					if (std::abs(error[new_extremals[i]]) > std::abs(error[alt.back()]))
						alt.back() = new_extremals[i];
				} else {
					alt.push_back(new_extremals[i]);
				}
			}

			// If we have more than n_extremals, trim the smallest from ends
			while (alt.size() > n_extremals) {
				if (std::abs(error[alt.front()]) < std::abs(error[alt.back()]))
					alt.erase(alt.begin());
				else
					alt.pop_back();
			}

			if (alt.size() >= n_extremals) {
				extremal_idx = alt;
			}
		}

		// Check convergence.
		//
		// The criterion is the Chebyshev alternation theorem: the design is
		// optimal exactly when no point on the grid has weighted error larger
		// than the ripple |delta| carried by the extremal set. Testing
		// "delta stopped changing" instead reports success at any fixed point
		// of the exchange, including the one it lands on when the extremal
		// search fails to supply a fresh set. (issue #203)
		delta = new_delta;
		if (max_err <= std::abs(delta) * (1.0 + 1e-9)) break;
	}

	// Continue from the best iterate rather than the last one.
	if (have_best) extremal_idx = best_idx;

	// Final: evaluate the converged approximation on a dense grid
	// and extract tap coefficients via inverse cosine/sine transform.

	// Re-extract final extremal set
	std::vector<double> ext_freq(n_extremals), ext_des(n_extremals),
	                    ext_wt(n_extremals), ext_cos(n_extremals);
	for (std::size_t i = 0; i < n_extremals; ++i) {
		ext_freq[i] = grid[extremal_idx[i]];
		ext_des[i]  = grid_des[extremal_idx[i]];
		ext_wt[i]   = grid_wt[extremal_idx[i]];
		ext_cos[i]  = std::cos(two_pi * ext_freq[i]);
	}

	// Re-derive delta for THIS reference set.
	//
	// The exchange loop computes delta from the set it starts the iteration
	// with, then replaces that set before the iteration ends. Carrying the
	// old delta into the extraction pairs a reference set with the ripple of
	// a different one, and eval_approx then interpolates values the set does
	// not satisfy — the polynomial it produces is not the converged design.
	// On well-behaved specs the two sets are identical by the final
	// iteration and the mismatch is invisible; on specs where the last
	// exchange still moves points it produced filters whose measured
	// stopband bore no relation to the reported delta (45 dB claimed,
	// 9 dB delivered). (issue #203)
	delta = detail::compute_delta(ext_des, ext_wt, ext_cos, n_extremals);

	// Compute barycentric weights for final polynomial
	// (same scaling as inside the exchange loop)
	std::size_t n_poly = n_extremals - 1;
	const double bary_scale = detail::barycentric_scale(ext_cos, n_poly);
	std::vector<double> bary_weights(n_poly);
	for (std::size_t i = 0; i < n_poly; ++i) {
		double prod = 1.0;
		for (std::size_t j = 0; j < n_poly; ++j) {
			if (j != i) {
				double diff = ext_cos[i] - ext_cos[j];
				if (std::abs(diff) < 1e-15) diff = (i < j) ? -1e-15 : 1e-15;
				prod *= bary_scale * diff;
			}
		}
		bary_weights[i] = 1.0 / prod;
	}

	// Post-condition: the design that is about to be returned must actually
	// equioscillate. Re-measure the weighted error of the FINAL (reference
	// set, delta) pair over the whole grid; by the alternation theorem no
	// grid point may exceed |delta|.
	//
	// Some specifications — notably half-band band edges placed symmetrically
	// about 0.25 with many taps and a narrow transition — are degenerate for
	// the exchange and it does not converge. Reference implementations report
	// this ("failure to converge; try reducing the transition band width").
	// Returning the non-converged iterate instead hands back a filter whose
	// stopband can sit ABOVE its passband, which is indistinguishable from a
	// working filter until something downstream measures it. Fail loudly.
	// (issue #203)
	//
	// Applies to all four types: since #205 folded each type's basis factor
	// into the weight, delta describes the filter every type returns.
	//
	// The second acceptance route covers specifications the exchange solves
	// so nearly exactly that delta underflows into rounding noise. There the
	// alternation test cannot be applied — it is comparing against a number
	// that is no longer meaningful — but the design can still be judged on
	// its own terms: a worst weighted error this far below the problem scale
	// is a filter no realizable arithmetic will distinguish from optimal.
	// Without this route a 64-tap Hilbert transformer over [0.10, 0.5], which
	// lands near 130 dB, would be rejected for not being provably optimal.
	// (issue #205)
	{
		double final_max_err = 0.0;
		for (std::size_t i = 0; i < grid_size; ++i) {
			double approx = detail::eval_approx(grid[i], ext_freq, ext_des, ext_wt,
			                                    ext_cos, bary_weights, delta, n_poly);
			final_max_err = std::max(final_max_err, std::abs(grid_wt[i] * (grid_des[i] - approx)));
		}
		const bool finite      = std::isfinite(delta) && std::isfinite(final_max_err);
		const bool equiripples = std::abs(delta) > 0.0 &&
		                         final_max_err <= std::abs(delta) * remez_convergence_tol;
		const bool exact_enough = final_max_err <= remez_exactness_floor * problem_scale;
		if (!finite || !(equiripples || exact_enough)) {
			throw std::runtime_error(
			    "remez: failed to converge for num_taps=" + std::to_string(num_taps) +
			    " (ripple " + std::to_string(std::abs(delta)) +
			    ", worst grid error " + std::to_string(final_max_err) +
			    "); widen the transition band, change num_taps, or raise max_iterations");
		}
	}

	// Evaluate the converged P(f) on a dense uniform grid and recover its
	// cosine-polynomial coefficients via an inverse DCT-I.
	//
	// This is now the same computation for all four types: the exchange
	// approximates P, not A, and the type's basis factor q is reapplied
	// analytically by the conversion below. (issue #205)
	std::size_t M_eval = std::max<std::size_t>(4 * (L + 1), 128);

	// Evaluate P on the CLOSED interval [0, 0.5], i = 0..M_eval. The closed
	// grid is required for the inverse transform to be orthogonal; see the
	// DCT-I note below. P itself is a polynomial in cos(2*pi*f) and has no
	// singularity at the endpoints, so it is safe to evaluate there even
	// though the exchange grid excluded them.
	std::vector<double> Pv(M_eval + 1);
	for (std::size_t i = 0; i <= M_eval; ++i) {
		double f = 0.5 * static_cast<double>(i) / static_cast<double>(M_eval);
		Pv[i] = detail::eval_approx(f, ext_freq, ext_des, ext_wt,
		                            ext_cos, bary_weights, delta, n_poly);
	}

	// Inverse DCT-I: p[k] = (2/M) * sum_i'' P(f_i) * cos(2*pi*f_i*k),
	// where sum'' half-weights the two endpoints i = 0 and i = M_eval.
	// With f_i = 0.5*i/M the basis is cos(pi*i*k/M), which is orthogonal
	// on the closed grid ONLY under those endpoint weights. Sampling the
	// half-open [0, 0.5) with uniform weights instead leaves every p[k]
	// with an O(1/M) error; with the weights the recovery is exact to
	// machine precision at any M >= L. (issue #203)
	// p[0] uses (1/M) since cos(0) = 1 for all terms (DC has no alternation)
	std::vector<double> p(L + 1, 0.0);
	for (std::size_t k = 0; k <= L; ++k) {
		double sum = 0.0;
		for (std::size_t i = 0; i <= M_eval; ++i) {
			double f = 0.5 * static_cast<double>(i) / static_cast<double>(M_eval);
			double w = (i == 0 || i == M_eval) ? 0.5 : 1.0;
			sum += w * Pv[i] * std::cos(two_pi * f * static_cast<double>(k));
		}
		double scale = (k == 0) ? 1.0 : 2.0;
		p[k] = sum * scale / static_cast<double>(M_eval);
	}

	// Reapply q(f) analytically and convert to taps.
	//
	// Each identity below expands q(f)*cos(2*pi*f*k) back onto the type's
	// natural basis, so the conversion is exact rather than a resampling:
	//
	//   II   cos(pi*f)   * cos(k*w) = 1/2 [cos((k+1/2)w) + cos((k-1/2)w)]
	//   III  sin(2*pi*f) * cos(k*w) = 1/2 [sin((k+1)w)   - sin((k-1)w)]
	//   IV   sin(pi*f)   * cos(k*w) = 1/2 [sin((k+1/2)w) - sin((k-1/2)w)]
	//
	// with w = 2*pi*f. The k = 0 term of each folds onto itself (cos and
	// sin of a negated argument), which is why the first coefficient of
	// each list carries a whole p[0] rather than half of one.
	// (issue #205)
	mtl::vec::dense_vector<T> taps(num_taps);
	auto p_at = [&](std::size_t k) -> double { return (k <= L) ? p[k] : 0.0; };

	switch (lp_type) {
	case detail::LinearPhaseType::I: {
		// A(f) = sum_{k=0}^{L} p[k] cos(2*pi*f*k)
		// h[L] = p[0], h[L-k] = h[L+k] = p[k]/2
		taps[L] = static_cast<T>(p[0]);
		for (std::size_t k = 1; k <= L; ++k) {
			T val = static_cast<T>(p[k] / 2.0);
			taps[L - k] = val;
			taps[L + k] = val;
		}
		break;
	}
	case detail::LinearPhaseType::II: {
		// A(f) = sum_{j=1}^{M} b[j] cos(2*pi*f*(j-1/2)),  M = N/2, L = M-1
		// h[M-j] = h[M+j-1] = b[j]/2
		const std::size_t M = num_taps / 2;
		std::vector<double> b(M + 1, 0.0);
		b[1] = p_at(0) + 0.5 * p_at(1);
		for (std::size_t j = 2; j <= M; ++j) b[j] = 0.5 * (p_at(j - 1) + p_at(j));
		for (std::size_t j = 1; j <= M; ++j) {
			T val = static_cast<T>(b[j] / 2.0);
			taps[M - j]     = val;
			taps[M + j - 1] = val;
		}
		break;
	}
	case detail::LinearPhaseType::III: {
		// A(f) = sum_{m=1}^{Lf} c[m] sin(2*pi*f*m),  Lf = (N-1)/2, L = Lf-1
		// h[Lf] = 0, h[Lf-m] = -h[Lf+m] = c[m]/2
		const std::size_t Lf = (num_taps - 1) / 2;
		std::vector<double> c(Lf + 1, 0.0);
		c[1] = p_at(0) - 0.5 * p_at(2);
		for (std::size_t m = 2; m <= Lf; ++m) c[m] = 0.5 * (p_at(m - 1) - p_at(m + 1));
		taps[Lf] = T{0};
		for (std::size_t m = 1; m <= Lf; ++m) {
			double val = c[m] / 2.0;
			taps[Lf - m] = static_cast<T>(val);
			taps[Lf + m] = static_cast<T>(-val);
		}
		break;
	}
	default: {
		// Type IV: A(f) = sum_{m=1}^{M} d[m] sin(2*pi*f*(m-1/2)), M = N/2, L = M-1
		// h[M-m] = -h[M+m-1] = d[m]/2
		const std::size_t M = num_taps / 2;
		std::vector<double> d(M + 1, 0.0);
		d[1] = p_at(0) - 0.5 * p_at(1);
		for (std::size_t m = 2; m <= M; ++m) d[m] = 0.5 * (p_at(m - 1) - p_at(m));
		for (std::size_t m = 1; m <= M; ++m) {
			double val = d[m] / 2.0;
			taps[M - m]     = static_cast<T>(val);
			taps[M + m - 1] = static_cast<T>(-val);
		}
		break;
	}
	}

	return taps;
}

// Convenience: equiripple lowpass FIR design
template <DspField T>
mtl::vec::dense_vector<T> design_fir_equiripple_lowpass(
    std::size_t num_taps,
    T passband_edge,
    T stopband_edge,
    T passband_weight = T{1},
    T stopband_weight = T{1}) {

	std::vector<T> bands    = {T{0}, passband_edge, stopband_edge, T(0.5)};
	std::vector<T> desired_  = {T{1}, T{1}, T{0}, T{0}};
	std::vector<T> wts      = {passband_weight, stopband_weight};
	return remez<T>(num_taps, bands, desired_, wts);
}

// Convenience: equiripple bandpass FIR design
template <DspField T>
mtl::vec::dense_vector<T> design_fir_equiripple_bandpass(
    std::size_t num_taps,
    T stop1, T pass1, T pass2, T stop2,
    T stopband_weight = T{1},
    T passband_weight = T{1}) {

	std::vector<T> bands    = {T{0}, stop1, pass1, pass2, stop2, T(0.5)};
	std::vector<T> desired_  = {T{0}, T{0}, T{1}, T{1}, T{0}, T{0}};
	std::vector<T> wts      = {stopband_weight, passband_weight, stopband_weight};
	return remez<T>(num_taps, bands, desired_, wts);
}

} // namespace sw::dsp
