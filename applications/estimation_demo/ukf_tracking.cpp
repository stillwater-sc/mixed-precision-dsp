// ukf_tracking.cpp: 2D target tracking with the Unscented Kalman Filter
//
// Same bearing-range scenario as the EKF demo, but the UKF needs no
// Jacobians — it propagates sigma points directly through the nonlinear
// observation model h(x) = [range, bearing].
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/estimation/ukf.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

using namespace sw::dsp;

int main() {
	using vec = mtl::vec::dense_vector<double>;
	constexpr int N = 80;

	UnscentedKalmanFilter<double> ukf(4, 2);

	ukf.set_state_function([](const vec& s) -> vec {
		constexpr double dt = 1.0;
		vec s_new(4);
		s_new[0] = s[0] + dt * s[2];
		s_new[1] = s[1] + dt * s[3];
		s_new[2] = s[2];
		s_new[3] = s[3];
		return s_new;
	});

	ukf.set_observation_function([](const vec& s) -> vec {
		vec z(2);
		z[0] = std::sqrt(s[0] * s[0] + s[1] * s[1]);
		z[1] = std::atan2(s[1], s[0]);
		return z;
	});

	ukf.Q()(0, 0) = 0.1;  ukf.Q()(1, 1) = 0.1;
	ukf.Q()(2, 2) = 0.01; ukf.Q()(3, 3) = 0.01;
	ukf.R()(0, 0) = 4.0;
	ukf.R()(1, 1) = 0.0025;

	ukf.state()[0] = 90.0;
	ukf.state()[1] = 10.0;
	ukf.state()[2] = 0.0;
	ukf.state()[3] = 0.0;
	ukf.P()(0, 0) = 200.0; ukf.P()(1, 1) = 200.0;
	ukf.P()(2, 2) = 20.0;  ukf.P()(3, 3) = 20.0;

	double true_x = 100.0, true_y = 0.0;
	double true_vx = -0.5, true_vy = 10.0;

	std::mt19937 gen(42);
	std::normal_distribution<double> range_noise(0.0, 2.0);
	std::normal_distribution<double> bearing_noise(0.0, 0.05);

	std::cout << "=== UKF Bearing-Range Tracking Demo ===\n";
	std::cout << "Sigma-point sampling (no Jacobians required)\n";
	std::cout << "Target: (100, 0) velocity (-0.5, 10)\n\n";
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
		true_x += true_vx;
		true_y += true_vy;
		z[0] = std::sqrt(true_x * true_x + true_y * true_y) + range_noise(gen);
		z[1] = std::atan2(true_y, true_x) + bearing_noise(gen);

		ukf.predict();
		ukf.update(z);

		double ex = ukf.state()[0], ey = ukf.state()[1];
		double err = std::sqrt((ex - true_x) * (ex - true_x) +
		                       (ey - true_y) * (ey - true_y));
		if (t <= 10 || t % 10 == 0) {
			std::cout << std::setw(5) << t
			          << std::setw(14) << std::fixed << std::setprecision(2) << true_x
			          << std::setw(14) << true_y
			          << std::setw(14) << ex
			          << std::setw(14) << ey
			          << std::setw(14) << err << "\n";
		}
	}

	double final_err = std::sqrt(
		(ukf.state()[0] - true_x) * (ukf.state()[0] - true_x) +
		(ukf.state()[1] - true_y) * (ukf.state()[1] - true_y));
	std::cout << "\nFinal position error: " << std::fixed << std::setprecision(3)
	          << final_err << " m\n";
	std::cout << "Estimated velocity:   (" << ukf.state()[2] << ", " << ukf.state()[3]
	          << ") (true: " << true_vx << ", " << true_vy << ")\n";

	return 0;
}
