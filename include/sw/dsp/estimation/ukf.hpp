#pragma once
// ukf.hpp: Unscented Kalman Filter with sigma-point sampling
//
// The UKF propagates a deterministic set of 2n+1 "sigma points" through
// the nonlinear state-transition f(x) and observation h(x) functions,
// then reconstructs the predicted mean and covariance from the propagated
// points. Unlike the EKF, no Jacobians are required.
//
// Sigma-point scaling follows the Julier/Merwe parameterization
// (alpha, beta, kappa). Defaults are alpha=0.5, beta=2, kappa=3-n
// per Wan & van der Merwe 2001 recommendations for moderate nonlinearity.
//
// The matrix square root for sigma-point generation uses LDL^T
// decomposition (mtl::ldlt_factor) instead of standard Cholesky (LL^T).
// LDL^T avoids intermediate square roots, giving better numerical
// stability for ill-conditioned covariances — especially important
// for mixed-precision arithmetic where P can lose positive-definiteness
// after informative observations.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>
#include <vector>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/operators.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/vec/operators.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/operation/inv.hpp>
#include <mtl/operation/ldlt.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

template <DspField T>
class UnscentedKalmanFilter {
public:
	using matrix_t   = mtl::mat::dense2D<T>;
	using vector_t   = mtl::vec::dense_vector<T>;
	using state_func = std::function<vector_t(const vector_t&)>;
	using obs_func   = std::function<vector_t(const vector_t&)>;

	// alpha: spread of sigma points around the mean (0 < alpha <= 1).
	// beta:  prior knowledge of distribution (2 is optimal for Gaussian).
	// kappa: secondary scaling parameter. Default (nullopt) uses 3-n
	//        (Julier convention). Passing 0 explicitly is valid and gives
	//        the original non-scaled unscented transform.
	UnscentedKalmanFilter(std::size_t state_dim, std::size_t meas_dim,
	                      T alpha = T{0.5}, T beta = T{2},
	                      std::optional<T> kappa = std::nullopt)
		: n_(state_dim), m_(meas_dim),
		  alpha_(alpha), beta_(beta),
		  x_(state_dim, T{}),
		  P_(state_dim, state_dim),
		  Q_(state_dim, state_dim),
		  R_(meas_dim, meas_dim)
	{
		if (state_dim == 0)
			throw std::invalid_argument("UnscentedKalmanFilter: state_dim must be > 0");
		if (meas_dim == 0)
			throw std::invalid_argument("UnscentedKalmanFilter: meas_dim must be > 0");

		kappa_ = kappa.value_or(T{3} - static_cast<T>(state_dim));

		compute_weights();
		identity_matrix(P_);
		identity_matrix(Q_);
		identity_matrix(R_);
	}

	void set_state_function(state_func f) { f_ = std::move(f); }
	void set_observation_function(obs_func h) { h_ = std::move(h); }

	// Predict: generate sigma points from (x, P), propagate through f,
	// reconstruct predicted mean and covariance.
	void predict() {
		if (!f_)
			throw std::logic_error("UnscentedKalmanFilter::predict: state function not set");

		auto sigma = generate_sigma_points(x_, P_);

		// Propagate each sigma point through f.
		std::size_t num_pts = 2 * n_ + 1;
		std::vector<vector_t> chi_pred(num_pts);
		for (std::size_t i = 0; i < num_pts; ++i) {
			chi_pred[i] = f_(column(sigma, i));
		}

		// Reconstruct predicted mean.
		x_ = weighted_mean(chi_pred);

		// Reconstruct predicted covariance + process noise.
		P_ = weighted_covariance(chi_pred, x_, chi_pred, x_);
		P_ = P_ + Q_;
	}

	// Update: generate sigma points from predicted (x, P), propagate
	// through h, compute cross-covariance, Kalman gain, and update.
	void update(const vector_t& z) {
		if (!h_)
			throw std::logic_error("UnscentedKalmanFilter::update: observation function not set");
		if (z.size() != m_)
			throw std::invalid_argument("UnscentedKalmanFilter::update: measurement size mismatch");

		auto sigma = generate_sigma_points(x_, P_);

		std::size_t num_pts = 2 * n_ + 1;
		std::vector<vector_t> gamma(num_pts);
		std::vector<vector_t> chi(num_pts);
		for (std::size_t i = 0; i < num_pts; ++i) {
			chi[i] = column(sigma, i);
			gamma[i] = h_(chi[i]);
		}

		vector_t z_pred = weighted_mean(gamma);

		// Innovation covariance P_zz + R.
		matrix_t P_zz = weighted_covariance(gamma, z_pred, gamma, z_pred);
		P_zz = P_zz + R_;

		// Cross-covariance P_xz.
		matrix_t P_xz = weighted_covariance(chi, x_, gamma, z_pred);

		// Kalman gain.
		using mtl::inv;
		matrix_t K = P_xz * inv(P_zz);

		// Update state and covariance.
		using mtl::trans;
		vector_t innovation = z - z_pred;
		x_ = x_ + K * innovation;
		P_ = P_ - K * P_zz * trans(K);
	}

	// Accessors
	matrix_t& Q() { return Q_; }
	matrix_t& R() { return R_; }
	matrix_t& P() { return P_; }
	vector_t& state() { return x_; }

	const matrix_t& Q() const { return Q_; }
	const matrix_t& R() const { return R_; }
	const matrix_t& P() const { return P_; }
	const vector_t& state() const { return x_; }

	std::size_t state_dim() const { return n_; }
	std::size_t meas_dim() const { return m_; }

private:
	// --- Weight computation (Julier/Merwe) ---
	void compute_weights() {
		T n = static_cast<T>(n_);
		lambda_ = alpha_ * alpha_ * (n + kappa_) - n;
		T denom = n + lambda_;

		std::size_t num_pts = 2 * n_ + 1;
		Wm_.resize(num_pts);
		Wc_.resize(num_pts);

		Wm_[0] = lambda_ / denom;
		Wc_[0] = lambda_ / denom + (T{1} - alpha_ * alpha_ + beta_);
		for (std::size_t i = 1; i < num_pts; ++i) {
			Wm_[i] = T{1} / (T{2} * denom);
			Wc_[i] = Wm_[i];
		}
	}

	// --- Sigma-point generation via LDL^T ---
	// Returns an n × (2n+1) matrix whose columns are the sigma points.
	matrix_t generate_sigma_points(const vector_t& x, const matrix_t& P) const {
		std::size_t num_pts = 2 * n_ + 1;
		matrix_t sigma(n_, num_pts);

		// Compute S = L * sqrt(D) from LDL^T of (n + lambda) * P.
		T scale = static_cast<T>(n_) + lambda_;
		matrix_t A(n_, n_);
		for (std::size_t i = 0; i < n_; ++i)
			for (std::size_t j = 0; j < n_; ++j)
				A(i, j) = scale * P(i, j);

		int info = mtl::ldlt_factor(A);
		if (info != 0)
			throw std::runtime_error(
				"UnscentedKalmanFilter: LDL^T failed — covariance not positive definite "
				"(pivot " + std::to_string(info - 1) + " is zero)");

		// A now has: diagonal = D, strict lower triangle = L (unit diagonal).
		// Compute S = L * sqrt(D): column j of S = L[:,j] * sqrt(D[j]).
		using std::sqrt;
		matrix_t S(n_, n_);
		for (std::size_t j = 0; j < n_; ++j) {
			if (!(A(j, j) > T{}))
				throw std::runtime_error(
					"UnscentedKalmanFilter: non-positive diagonal D["
					+ std::to_string(j) + "] in LDL^T "
					"-- covariance not positive definite");
			T sj = sqrt(A(j, j));  // sqrt(D[j])
			S(j, j) = sj;          // L has unit diagonal
			for (std::size_t i = j + 1; i < n_; ++i) {
				S(i, j) = A(i, j) * sj;  // L(i,j) * sqrt(D[j])
			}
			for (std::size_t i = 0; i < j; ++i) {
				S(i, j) = T{};  // upper triangle = 0
			}
		}

		// chi[0] = x
		for (std::size_t i = 0; i < n_; ++i) sigma(i, 0) = x[i];

		// chi[1..n] = x + S[:,i-1],  chi[n+1..2n] = x - S[:,i-1]
		for (std::size_t j = 0; j < n_; ++j) {
			for (std::size_t i = 0; i < n_; ++i) {
				sigma(i, j + 1)      = x[i] + S(i, j);
				sigma(i, j + 1 + n_) = x[i] - S(i, j);
			}
		}
		return sigma;
	}

	// Extract column j from a matrix as a vector.
	static vector_t column(const matrix_t& M, std::size_t j) {
		std::size_t rows = M.num_rows();
		vector_t v(rows);
		for (std::size_t i = 0; i < rows; ++i) v[i] = M(i, j);
		return v;
	}

	// Weighted mean of a set of vectors.
	vector_t weighted_mean(const std::vector<vector_t>& pts) const {
		std::size_t dim = pts[0].size();
		vector_t mu(dim, T{});
		for (std::size_t i = 0; i < pts.size(); ++i) {
			for (std::size_t d = 0; d < dim; ++d) {
				mu[d] = mu[d] + Wm_[i] * pts[i][d];
			}
		}
		return mu;
	}

	// Weighted cross-covariance: sum_i W_c[i] * (a[i]-mu_a) * (b[i]-mu_b)^T
	matrix_t weighted_covariance(const std::vector<vector_t>& a, const vector_t& mu_a,
	                             const std::vector<vector_t>& b, const vector_t& mu_b) const {
		std::size_t da = mu_a.size(), db = mu_b.size();
		matrix_t C(da, db);
		for (std::size_t i = 0; i < da; ++i)
			for (std::size_t j = 0; j < db; ++j)
				C(i, j) = T{};

		for (std::size_t k = 0; k < a.size(); ++k) {
			for (std::size_t i = 0; i < da; ++i) {
				T diff_a = a[k][i] - mu_a[i];
				for (std::size_t j = 0; j < db; ++j) {
					C(i, j) = C(i, j) + Wc_[k] * diff_a * (b[k][j] - mu_b[j]);
				}
			}
		}
		return C;
	}

	static void identity_matrix(matrix_t& m) {
		std::size_t r = m.num_rows(), c = m.num_cols();
		for (std::size_t i = 0; i < r; ++i)
			for (std::size_t j = 0; j < c; ++j)
				m(i, j) = (i == j) ? T{1} : T{};
	}

	std::size_t n_;   // state dimension
	std::size_t m_;   // measurement dimension
	T alpha_, beta_, kappa_, lambda_;
	std::vector<T> Wm_, Wc_;  // mean and covariance weights
	vector_t    x_;
	matrix_t    P_, Q_, R_;
	state_func  f_;
	obs_func    h_;
};

} // namespace sw::dsp
