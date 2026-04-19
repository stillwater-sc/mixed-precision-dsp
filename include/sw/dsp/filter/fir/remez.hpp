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
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>

namespace sw::dsp {

enum class RemezBandType { bandpass, differentiator, hilbert };

namespace detail {

// Build a dense frequency grid covering all specified bands.
// Grid density controls the number of points per band per tap.
inline std::vector<double> build_grid(
    const std::vector<double>& bands,
    std::size_t num_taps,
    int grid_density,
    RemezBandType type) {

	std::size_t num_bands = bands.size() / 2;
	std::vector<double> grid;
	grid.reserve(num_taps * static_cast<std::size_t>(grid_density) * num_bands);

	for (std::size_t b = 0; b < num_bands; ++b) {
		double f_start = bands[2 * b];
		double f_end   = bands[2 * b + 1];

		// Avoid exact 0 for differentiator (weight function has 1/f singularity)
		if (type == RemezBandType::differentiator && f_start < 1e-6)
			f_start = 1e-6;

		std::size_t npts = std::max<std::size_t>(
		    static_cast<std::size_t>(grid_density * num_taps * (f_end - f_start) / 0.5), 4);

		for (std::size_t i = 0; i < npts; ++i) {
			double f = f_start + static_cast<double>(i) / static_cast<double>(npts - 1) * (f_end - f_start);
			grid.push_back(f);
		}
	}
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

// Compute delta (equiripple deviation) from the current extremal set
// using the Remez formula.
inline double compute_delta(
    const std::vector<double>& extremal_des,
    const std::vector<double>& extremal_wt,
    const std::vector<double>& extremal_cos,
    std::size_t n_extremals) {

	// Compute barycentric weights for the Chebyshev interpolation
	std::vector<double> bary(n_extremals);
	for (std::size_t i = 0; i < n_extremals; ++i) {
		double prod = 1.0;
		for (std::size_t j = 0; j < n_extremals; ++j) {
			if (j != i) {
				double diff = extremal_cos[i] - extremal_cos[j];
				if (std::abs(diff) < 1e-15) diff = 1e-15;
				prod *= diff;
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

	return numer / denom;
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

	// Convert inputs to double for internal computation
	std::vector<double> d_bands(bands.size()), d_desired(desired.size()), d_weights(weights.size());
	for (std::size_t i = 0; i < bands.size(); ++i) d_bands[i] = static_cast<double>(bands[i]);
	for (std::size_t i = 0; i < desired.size(); ++i) d_desired[i] = static_cast<double>(desired[i]);
	for (std::size_t i = 0; i < weights.size(); ++i) d_weights[i] = static_cast<double>(weights[i]);

	// Determine symmetry
	// Type I (odd taps) and Type II (even taps) for bandpass
	// Type III (odd taps) and Type IV (even taps) for differentiator/hilbert
	bool is_symmetric = (type == RemezBandType::bandpass);
	bool is_odd = (num_taps & 1) != 0;

	// Effective half-length for the cosine polynomial
	std::size_t L; // order of cosine polynomial
	if (is_symmetric) {
		L = is_odd ? (num_taps - 1) / 2 : num_taps / 2;
	} else {
		L = is_odd ? (num_taps - 1) / 2 : num_taps / 2 - 1;
	}
	std::size_t n_extremals = L + 2;

	// Build dense frequency grid
	auto grid = detail::build_grid(d_bands, num_taps, grid_density, type);
	std::size_t grid_size = grid.size();

	if (grid_size < n_extremals)
		throw std::invalid_argument("remez: grid too sparse for the given num_taps");

	// Compute desired and weight for each grid point
	std::vector<double> grid_des(grid_size), grid_wt(grid_size);
	for (std::size_t i = 0; i < grid_size; ++i) {
		detail::eval_desired_weight(grid[i], d_bands, d_desired, d_weights, type,
		                            grid_des[i], grid_wt[i]);
	}

	// Initialize extremal set with uniform spacing across grid
	std::vector<std::size_t> extremal_idx(n_extremals);
	for (std::size_t i = 0; i < n_extremals; ++i) {
		extremal_idx[i] = i * (grid_size - 1) / (n_extremals - 1);
	}

	// Remez exchange iteration
	double delta = 0.0;

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

		// Compute delta (equiripple deviation)
		double new_delta = detail::compute_delta(ext_des, ext_wt, ext_cos, n_extremals);

		// Compute barycentric weights for the polynomial (excluding last extremal)
		std::size_t n_poly = n_extremals - 1;
		std::vector<double> bary_weights(n_poly);
		for (std::size_t i = 0; i < n_poly; ++i) {
			double prod = 1.0;
			for (std::size_t j = 0; j < n_poly; ++j) {
				if (j != i) {
					double diff = ext_cos[i] - ext_cos[j];
					if (std::abs(diff) < 1e-15) diff = (i < j) ? -1e-15 : 1e-15;
					prod *= diff;
				}
			}
			bary_weights[i] = 1.0 / prod;
		}

		// Evaluate error on entire grid and find new extremals
		std::vector<double> error(grid_size);
		for (std::size_t i = 0; i < grid_size; ++i) {
			double approx = detail::eval_approx(grid[i], ext_freq, ext_des, ext_wt,
			                                     ext_cos, bary_weights, new_delta, n_poly);
			error[i] = grid_wt[i] * (grid_des[i] - approx);
		}

		// Find local extrema of the error function
		std::vector<std::size_t> new_extremals;
		new_extremals.reserve(n_extremals * 2);

		// Check first point
		if (grid_size > 1) {
			if ((error[0] > 0 && error[0] >= error[1]) ||
			    (error[0] < 0 && error[0] <= error[1]))
				new_extremals.push_back(0);
		}

		// Interior points
		for (std::size_t i = 1; i + 1 < grid_size; ++i) {
			if ((error[i] >= error[i-1] && error[i] >= error[i+1] && error[i] > 0) ||
			    (error[i] <= error[i-1] && error[i] <= error[i+1] && error[i] < 0))
				new_extremals.push_back(i);
		}

		// Check last point
		if (grid_size > 1) {
			std::size_t last = grid_size - 1;
			if ((error[last] > 0 && error[last] >= error[last-1]) ||
			    (error[last] < 0 && error[last] <= error[last-1]))
				new_extremals.push_back(last);
		}

		// Select the n_extremals largest-magnitude extrema with alternating sign
		if (new_extremals.size() >= n_extremals) {
			// Sort by error magnitude (descending)
			std::sort(new_extremals.begin(), new_extremals.end(),
			          [&error](std::size_t a, std::size_t b) {
				          return std::abs(error[a]) > std::abs(error[b]);
			          });

			// Greedy selection: pick alternating-sign extremals
			std::vector<std::size_t> selected;
			selected.reserve(n_extremals);

			// Start with the largest
			selected.push_back(new_extremals[0]);

			// Try to fill from sorted list, maintaining alternating sign
			for (std::size_t i = 1; i < new_extremals.size() && selected.size() < n_extremals; ++i) {
				// Just collect enough, we'll sort by frequency and fix sign later
				selected.push_back(new_extremals[i]);
			}

			if (selected.size() >= n_extremals) {
				selected.resize(n_extremals);
				// Sort by frequency (grid index)
				std::sort(selected.begin(), selected.end());
				extremal_idx = selected;
			}
		}

		// Check convergence
		if (iter > 0 && std::abs(std::abs(new_delta) - std::abs(delta)) <
		    std::abs(delta) * 1e-12) {
			delta = new_delta;
			break;
		}
		delta = new_delta;
	}

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

	// Compute barycentric weights for final polynomial
	std::size_t n_poly = n_extremals - 1;
	std::vector<double> bary_weights(n_poly);
	for (std::size_t i = 0; i < n_poly; ++i) {
		double prod = 1.0;
		for (std::size_t j = 0; j < n_poly; ++j) {
			if (j != i) {
				double diff = ext_cos[i] - ext_cos[j];
				if (std::abs(diff) < 1e-15) diff = (i < j) ? -1e-15 : 1e-15;
				prod *= diff;
			}
		}
		bary_weights[i] = 1.0 / prod;
	}

	// Evaluate the converged H(f) on a dense uniform grid and
	// recover cosine polynomial coefficients via inverse DCT.
	// Use a large grid for accurate coefficient recovery.
	std::size_t M_eval = std::max<std::size_t>(4 * (L + 1), 128);

	mtl::vec::dense_vector<T> taps(num_taps);

	if (is_symmetric) {
		// Type I (odd N=2L+1) or Type II (even N=2L):
		// H(f) = sum_{k=0}^{L} a[k] * cos(2*pi*f*k)
		// Taps: h[L-k] = h[L+k] = a[k]/2, h[L] = a[0]

		// Evaluate H at M_eval uniformly spaced points in [0, 0.5)
		std::vector<double> H(M_eval);
		for (std::size_t i = 0; i < M_eval; ++i) {
			double f = 0.5 * static_cast<double>(i) / static_cast<double>(M_eval);
			H[i] = detail::eval_approx(f, ext_freq, ext_des, ext_wt,
			                           ext_cos, bary_weights, delta, n_poly);
		}

		// Inverse DCT: a[k] = (2/M) * sum_i H(f_i) * cos(2*pi*f_i*k)
		// a[0] uses (1/M) since cos(0) = 1 for all terms (DC has no alternation)
		std::vector<double> a(L + 1, 0.0);
		for (std::size_t k = 0; k <= L; ++k) {
			double sum = 0.0;
			for (std::size_t i = 0; i < M_eval; ++i) {
				double f = 0.5 * static_cast<double>(i) / static_cast<double>(M_eval);
				sum += H[i] * std::cos(two_pi * f * static_cast<double>(k));
			}
			double scale = (k == 0) ? 1.0 : 2.0;
			a[k] = sum * scale / static_cast<double>(M_eval);
		}

		// Convert cosine coefficients to symmetric tap coefficients
		if (is_odd) {
			// Type I: N = 2L+1, center tap at index L
			// h[L] = a[0], h[L±k] = a[k]/2
			taps[L] = static_cast<T>(a[0]);
			for (std::size_t k = 1; k <= L; ++k) {
				T val = static_cast<T>(a[k] / 2.0);
				taps[L - k] = val;
				taps[L + k] = val;
			}
		} else {
			// Type II: N = 2L, center between indices L-1 and L
			// h[n] computed via IDFT from the cos polynomial with half-sample shift
			// b[k] = a[k] for even, but cos((k+0.5)*...) basis
			// Simpler: directly evaluate h[n] = (1/N)*sum H(f)*e^{j2pi*f*n} via real IDFT
			double half = static_cast<double>(num_taps - 1) / 2.0;
			for (std::size_t n = 0; n < num_taps; ++n) {
				double val = a[0];
				for (std::size_t k = 1; k <= L; ++k) {
					val += a[k] * std::cos(pi * static_cast<double>(k) *
					       (2.0 * static_cast<double>(n) - 2.0 * half) /
					       static_cast<double>(num_taps));
				}
				taps[n] = static_cast<T>(val / static_cast<double>(num_taps));
			}
		}
	} else {
		// Type III (odd) / Type IV (even): antisymmetric
		std::vector<double> H(M_eval);
		for (std::size_t i = 0; i < M_eval; ++i) {
			double f = 0.5 * static_cast<double>(i) / static_cast<double>(M_eval);
			H[i] = detail::eval_approx(f, ext_freq, ext_des, ext_wt,
			                           ext_cos, bary_weights, delta, n_poly);
		}

		// Inverse DST: b[k] = (2/M) * sum_i H(f_i) * sin(2*pi*f_i*(k+1))
		std::vector<double> b(L + 1, 0.0);
		for (std::size_t k = 0; k <= L; ++k) {
			double sum = 0.0;
			for (std::size_t i = 0; i < M_eval; ++i) {
				double f = 0.5 * static_cast<double>(i) / static_cast<double>(M_eval);
				sum += H[i] * std::sin(two_pi * f * static_cast<double>(k + 1));
			}
			b[k] = sum * 2.0 / static_cast<double>(M_eval);
		}

		if (is_odd) {
			// Type III: h[L] = 0, h[L±k] = ±b[k-1]/2
			taps[L] = T{0};
			for (std::size_t k = 1; k <= L; ++k) {
				T val = static_cast<T>(b[k - 1] / 2.0);
				taps[L + k] = val;
				taps[L - k] = static_cast<T>(-static_cast<double>(val));
			}
		} else {
			double half = static_cast<double>(num_taps - 1) / 2.0;
			for (std::size_t n = 0; n < num_taps; ++n) {
				double val = 0.0;
				for (std::size_t k = 0; k <= L; ++k) {
					val += b[k] * std::sin(pi * static_cast<double>(k + 1) *
					       (2.0 * static_cast<double>(n) - 2.0 * half) /
					       static_cast<double>(num_taps));
				}
				taps[n] = static_cast<T>(val / static_cast<double>(num_taps));
			}
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
