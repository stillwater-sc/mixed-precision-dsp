#pragma once
// ekf.hpp: Extended Kalman Filter for nonlinear state estimation
//
// The EKF linearizes nonlinear state-transition f(x) and observation h(x)
// functions around the current estimate by evaluating their Jacobians F(x)
// and H(x) at each step.
//
// Predict:
//   x_pred = f(x)                    (nonlinear state propagation)
//   F      = df/dx evaluated at x    (state-transition Jacobian)
//   P_pred = F * P * F^T + Q
//
// Update:
//   y = z - h(x_pred)                (nonlinear innovation)
//   H = dh/dx evaluated at x_pred    (observation Jacobian)
//   S = H * P_pred * H^T + R
//   K = P_pred * H^T * S^-1
//   x = x_pred + K * y
//   P = (I - K * H) * P_pred
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/operators.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/vec/operators.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/operation/inv.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

template <DspField T>
class ExtendedKalmanFilter {
public:
	using matrix_t  = mtl::mat::dense2D<T>;
	using vector_t  = mtl::vec::dense_vector<T>;
	using state_func = std::function<vector_t(const vector_t&)>;
	using obs_func   = std::function<vector_t(const vector_t&)>;
	using state_jac  = std::function<matrix_t(const vector_t&)>;
	using obs_jac    = std::function<matrix_t(const vector_t&)>;

	ExtendedKalmanFilter(std::size_t state_dim, std::size_t meas_dim)
		: state_dim_(state_dim), meas_dim_(meas_dim),
		  x_(state_dim, T{}),
		  P_(state_dim, state_dim),
		  Q_(state_dim, state_dim),
		  R_(meas_dim, meas_dim)
	{
		if (state_dim == 0)
			throw std::invalid_argument("ExtendedKalmanFilter: state_dim must be > 0");
		if (meas_dim == 0)
			throw std::invalid_argument("ExtendedKalmanFilter: meas_dim must be > 0");
		identity_matrix(P_);
		identity_matrix(Q_);
		identity_matrix(R_);
	}

	// Set the nonlinear state transition f(x) and its Jacobian F = df/dx.
	void set_state_function(state_func f, state_jac F) {
		f_ = std::move(f);
		F_ = std::move(F);
	}

	// Set the nonlinear observation h(x) and its Jacobian H = dh/dx.
	void set_observation_function(obs_func h, obs_jac H) {
		h_ = std::move(h);
		H_ = std::move(H);
	}

	// Predict step: propagate state through f(x), linearize via F(x).
	void predict() {
		using mtl::trans;
		if (!f_ || !F_)
			throw std::logic_error("ExtendedKalmanFilter::predict: state function not set");

		matrix_t F_eval = F_(x_);
		x_ = f_(x_);
		P_ = F_eval * P_ * trans(F_eval) + Q_;
	}

	// Update step: correct state using measurement z.
	void update(const vector_t& z) {
		using mtl::trans;
		using mtl::inv;
		if (!h_ || !H_)
			throw std::logic_error("ExtendedKalmanFilter::update: observation function not set");
		if (z.size() != meas_dim_)
			throw std::invalid_argument("ExtendedKalmanFilter::update: measurement size mismatch");

		matrix_t H_eval = H_(x_);
		vector_t y = z - h_(x_);

		matrix_t S = H_eval * P_ * trans(H_eval) + R_;
		matrix_t K = P_ * trans(H_eval) * inv(S);

		x_ = x_ + K * y;

		matrix_t I_n(state_dim_, state_dim_);
		identity_matrix(I_n);
		P_ = (I_n - K * H_eval) * P_;
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

	std::size_t state_dim() const { return state_dim_; }
	std::size_t meas_dim() const { return meas_dim_; }

private:
	static void identity_matrix(matrix_t& m) {
		std::size_t n = m.num_rows();
		for (std::size_t i = 0; i < n; ++i) {
			for (std::size_t j = 0; j < m.num_cols(); ++j) {
				m(i, j) = (i == j) ? T{1} : T{};
			}
		}
	}

	std::size_t state_dim_;
	std::size_t meas_dim_;
	vector_t    x_;
	matrix_t    P_;
	matrix_t    Q_;
	matrix_t    R_;
	state_func  f_;
	state_jac   F_;
	obs_func    h_;
	obs_jac     H_;
};

} // namespace sw::dsp
