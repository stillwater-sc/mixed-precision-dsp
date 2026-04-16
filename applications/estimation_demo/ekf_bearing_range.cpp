// ekf_bearing_range.cpp: track a 2D target using range/bearing measurements
//
// Demonstrates the Extended Kalman Filter on a classic nonlinear estimation
// problem: a sensor at the origin measures range r = sqrt(x^2+y^2) and
// bearing theta = atan2(y,x) of a target moving at constant velocity.
//
// The state vector is [x, y, vx, vy]. The observation model h(x) =
// [r, theta] is nonlinear; the EKF linearizes it via its Jacobian at each
// step to maintain a Gaussian state estimate.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/estimation/ekf.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

using namespace sw::dsp;

int main() {
	using vec = mtl::vec::dense_vector<double>;
	using mat = mtl::mat::dense2D<double>;
	constexpr double dt = 1.0;
	constexpr int    N  = 80;

	ExtendedKalmanFilter<double> ekf(4, 2);

	// State transition: constant velocity.
	ekf.set_state_function(
		[](const vec& s) -> vec {
			constexpr double dt = 1.0;
			vec s_new(4);
			s_new[0] = s[0] + dt * s[2];
			s_new[1] = s[1] + dt * s[3];
			s_new[2] = s[2];
			s_new[3] = s[3];
			return s_new;
		},
		[](const vec&) -> mat {
			constexpr double dt = 1.0;
			mat F(4, 4);
			for (std::size_t i = 0; i < 4; ++i)
				for (std::size_t j = 0; j < 4; ++j)
					F(i, j) = (i == j) ? 1.0 : 0.0;
			F(0, 2) = dt;
			F(1, 3) = dt;
			return F;
		}
	);

	// Observation: range and bearing.
	ekf.set_observation_function(
		[](const vec& s) -> vec {
			vec z(2);
			z[0] = std::sqrt(s[0] * s[0] + s[1] * s[1]);
			z[1] = std::atan2(s[1], s[0]);
			return z;
		},
		[](const vec& s) -> mat {
			double x = s[0], y = s[1];
			double r = std::sqrt(x * x + y * y);
			double r2 = r * r;
			mat H(2, 4);
			for (std::size_t i = 0; i < 2; ++i)
				for (std::size_t j = 0; j < 4; ++j)
					H(i, j) = 0.0;
			if (r > 1e-12) {
				H(0, 0) = x / r;   H(0, 1) = y / r;
				H(1, 0) = -y / r2; H(1, 1) = x / r2;
			}
			return H;
		}
	);

	// Process and measurement noise.
	ekf.Q()(0, 0) = 0.1;  ekf.Q()(1, 1) = 0.1;
	ekf.Q()(2, 2) = 0.01; ekf.Q()(3, 3) = 0.01;
	ekf.R()(0, 0) = 4.0;     // range noise variance (m^2)
	ekf.R()(1, 1) = 0.0025;  // bearing noise variance (rad^2, ~2.9 deg)

	// Initial estimate.
	ekf.state()[0] = 90.0;
	ekf.state()[1] = 10.0;
	ekf.state()[2] = 0.0;
	ekf.state()[3] = 0.0;
	ekf.P()(0, 0) = 200.0; ekf.P()(1, 1) = 200.0;
	ekf.P()(2, 2) = 20.0;  ekf.P()(3, 3) = 20.0;

	// True trajectory.
	double true_x = 100.0, true_y = 0.0;
	double true_vx = -0.5, true_vy = 10.0;

	std::mt19937 gen(42);
	std::normal_distribution<double> range_noise(0.0, 2.0);
	std::normal_distribution<double> bearing_noise(0.0, 0.05);

	std::cout << "=== EKF Bearing-Range Tracking Demo ===\n";
	std::cout << "Target starts at (100, 0) with velocity (-0.5, 10)\n";
	std::cout << "Sensor at origin measures range and bearing\n\n";
	std::cout << std::left
	          << std::setw(5) << "t"
	          << std::setw(14) << "true_x"
	          << std::setw(14) << "true_y"
	          << std::setw(14) << "est_x"
	          << std::setw(14) << "est_y"
	          << std::setw(14) << "err_pos"
	          << "\n";
	std::cout << std::string(75, '-') << "\n";

	vec z(2);
	for (int t = 1; t <= N; ++t) {
		true_x += true_vx * dt;
		true_y += true_vy * dt;

		double r_true = std::sqrt(true_x * true_x + true_y * true_y);
		double b_true = std::atan2(true_y, true_x);
		z[0] = r_true + range_noise(gen);
		z[1] = b_true + bearing_noise(gen);

		ekf.predict();
		ekf.update(z);

		double ex = ekf.state()[0], ey = ekf.state()[1];
		double err = std::sqrt((ex - true_x) * (ex - true_x) +
		                       (ey - true_y) * (ey - true_y));

		if (t <= 10 || t % 10 == 0) {
			std::cout << std::setw(5) << t
			          << std::setw(14) << std::fixed << std::setprecision(2) << true_x
			          << std::setw(14) << true_y
			          << std::setw(14) << ex
			          << std::setw(14) << ey
			          << std::setw(14) << err
			          << "\n";
		}
	}

	double final_err = std::sqrt(
		(ekf.state()[0] - true_x) * (ekf.state()[0] - true_x) +
		(ekf.state()[1] - true_y) * (ekf.state()[1] - true_y));
	std::cout << "\nFinal position error: " << std::fixed << std::setprecision(3)
	          << final_err << " m\n";
	std::cout << "Estimated velocity:   (" << ekf.state()[2] << ", " << ekf.state()[3]
	          << ") (true: " << true_vx << ", " << true_vy << ")\n";

	return 0;
}
