#pragma once
// kalman.hpp: linear Kalman filter
//
// State estimation via predict/update cycle. The user provides
// system matrices F, H, Q, R (and optionally B for control input).
//
// Predict:
//   x_pred = F * x + B * u
//   P_pred = F * P * F^T + Q
//
// Update:
//   y = z - H * x_pred           (innovation)
//   S = H * P_pred * H^T + R     (innovation covariance)
//   K = P_pred * H^T * S^-1      (Kalman gain)
//   x = x_pred + K * y
//   P = (I - K * H) * P_pred
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <stdexcept>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/operators.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/vec/operators.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/operation/inv.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

// Linear Kalman filter.
//
// Template parameters:
//   T:        scalar type (DspField)
//
// State dimension, measurement dimension, and control dimension are
// runtime parameters set via setup().
template <DspField T>
class KalmanFilter {
public:
	using matrix_t = mtl::mat::dense2D<T>;
	using vector_t = mtl::vec::dense_vector<T>;

	// Initialize with state dimension and measurement dimension.
	// Control dimension defaults to 0 (no control input).
	KalmanFilter(std::size_t state_dim, std::size_t meas_dim, std::size_t ctrl_dim = 0)
		: state_dim_(state_dim), meas_dim_(meas_dim), ctrl_dim_(ctrl_dim),
		  x_(state_dim, T{}),
		  P_(state_dim, state_dim),
		  F_(state_dim, state_dim),
		  H_(meas_dim, state_dim),
		  Q_(state_dim, state_dim),
		  R_(meas_dim, meas_dim),
		  B_(state_dim, ctrl_dim > 0 ? ctrl_dim : 1)
	{
		if (state_dim == 0) throw std::invalid_argument("KalmanFilter: state_dim must be > 0");
		if (meas_dim == 0) throw std::invalid_argument("KalmanFilter: meas_dim must be > 0");
		// Initialize P, F, Q to identity by default
		identity_matrix(P_);
		identity_matrix(F_);
		identity_matrix(Q_);
		identity_matrix(R_);
	}

	// System matrix accessors
	matrix_t& F() { return F_; }
	matrix_t& H() { return H_; }
	matrix_t& Q() { return Q_; }
	matrix_t& R() { return R_; }
	matrix_t& B() { return B_; }
	matrix_t& P() { return P_; }
	vector_t& state() { return x_; }

	const matrix_t& F() const { return F_; }
	const matrix_t& H() const { return H_; }
	const matrix_t& Q() const { return Q_; }
	const matrix_t& R() const { return R_; }
	const matrix_t& B() const { return B_; }
	const matrix_t& P() const { return P_; }
	const vector_t& state() const { return x_; }

	// Predict step (no control input)
	void predict() {
		using mtl::trans;
		// x = F * x
		x_ = F_ * x_;
		// P = F * P * F^T + Q
		P_ = F_ * P_ * trans(F_) + Q_;
	}

	// Predict step with control input
	void predict(const vector_t& u) {
		using mtl::trans;
		if (u.size() != ctrl_dim_)
			throw std::invalid_argument("KalmanFilter::predict: control vector size mismatch");
		x_ = F_ * x_ + B_ * u;
		P_ = F_ * P_ * trans(F_) + Q_;
	}

	// Update step with measurement z
	void update(const vector_t& z) {
		using mtl::trans;
		using mtl::inv;
		if (z.size() != meas_dim_)
			throw std::invalid_argument("KalmanFilter::update: measurement size mismatch");

		// Innovation: y = z - H * x
		vector_t y = z - H_ * x_;

		// Innovation covariance: S = H * P * H^T + R
		matrix_t S = H_ * P_ * trans(H_) + R_;

		// Kalman gain: K = P * H^T * S^-1
		matrix_t K = P_ * trans(H_) * inv(S);

		// Update state: x = x + K * y
		x_ = x_ + K * y;

		// Update covariance: P = (I - K * H) * P
		matrix_t I_n(state_dim_, state_dim_);
		identity_matrix(I_n);
		P_ = (I_n - K * H_) * P_;
	}

	std::size_t state_dim() const { return state_dim_; }
	std::size_t meas_dim() const { return meas_dim_; }
	std::size_t ctrl_dim() const { return ctrl_dim_; }

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
	std::size_t ctrl_dim_;
	vector_t x_;     // state estimate
	matrix_t P_;     // error covariance
	matrix_t F_;     // state transition
	matrix_t H_;     // observation
	matrix_t Q_;     // process noise covariance
	matrix_t R_;     // measurement noise covariance
	matrix_t B_;     // control input matrix
};

} // namespace sw::dsp
