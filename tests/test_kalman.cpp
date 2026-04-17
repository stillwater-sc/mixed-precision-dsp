// test_kalman.cpp: test Kalman filter, LMS, and RLS adaptive filters
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/estimation/estimation.hpp>
#include <sw/dsp/signals/generators.hpp>

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace sw::dsp;

bool near(double a, double b, double eps = 1e-3) {
	return std::abs(a - b) < eps;
}

// ========== Kalman Filter Tests ==========

void test_kalman_construction() {
	KalmanFilter<double> kf(2, 1);
	if (!(kf.state_dim() == 2)) throw std::runtime_error("test failed: kalman state_dim");
	if (!(kf.meas_dim() == 1)) throw std::runtime_error("test failed: kalman meas_dim");
	if (!(kf.state().size() == 2)) throw std::runtime_error("test failed: kalman state size");

	std::cout << "  kalman_construction: passed\n";
}

void test_kalman_constant_velocity() {
	// 1D constant-velocity tracking:
	// State = [position, velocity], dt = 1
	// F = [[1, 1], [0, 1]]  (position += velocity * dt)
	// H = [[1, 0]]          (we only measure position)
	// Q = small process noise
	// R = measurement noise variance

	KalmanFilter<double> kf(2, 1);

	// State transition: position += velocity (dt=1)
	kf.F()(0, 0) = 1.0; kf.F()(0, 1) = 1.0;
	kf.F()(1, 0) = 0.0; kf.F()(1, 1) = 1.0;

	// Observation: measure position only
	kf.H()(0, 0) = 1.0; kf.H()(0, 1) = 0.0;

	// Process noise (small)
	kf.Q()(0, 0) = 0.001; kf.Q()(0, 1) = 0.0;
	kf.Q()(1, 0) = 0.0;   kf.Q()(1, 1) = 0.001;

	// Measurement noise
	kf.R()(0, 0) = 0.5;

	// Initial state covariance (uncertain)
	kf.P()(0, 0) = 100.0; kf.P()(0, 1) = 0.0;
	kf.P()(1, 0) = 0.0;   kf.P()(1, 1) = 100.0;

	// True trajectory: position starts at 0, velocity = 1
	// Generate noisy measurements
	std::mt19937 gen(42);
	std::normal_distribution<double> noise(0.0, 0.5);

	mtl::vec::dense_vector<double> z(1);
	for (int t = 0; t < 50; ++t) {
		double true_pos = static_cast<double>(t);
		z[0] = true_pos + noise(gen);

		kf.predict();
		kf.update(z);
	}

	// After 50 steps, estimated position and velocity should be close to truth
	double est_pos = kf.state()[0];
	double est_vel = kf.state()[1];

	if (!(near(est_pos, 49.0, 2.0)))
		throw std::runtime_error("test failed: kalman position estimate");
	if (!(near(est_vel, 1.0, 0.3)))
		throw std::runtime_error("test failed: kalman velocity estimate");

	std::cout << "  kalman_constant_velocity: passed (pos=" << est_pos
	          << ", vel=" << est_vel << ")\n";
}

void test_kalman_validation() {
	bool caught = false;
	try { KalmanFilter<double> kf(0, 1); }
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: kalman should reject state_dim=0");

	caught = false;
	try { KalmanFilter<double> kf(2, 0); }
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: kalman should reject meas_dim=0");

	std::cout << "  kalman_validation: passed\n";
}

// ========== EKF Tests ==========

void test_ekf_construction() {
	ExtendedKalmanFilter<double> ekf(4, 2);
	if (!(ekf.state_dim() == 4)) throw std::runtime_error("test failed: ekf state_dim");
	if (!(ekf.meas_dim() == 2))  throw std::runtime_error("test failed: ekf meas_dim");
	if (!(ekf.state().size() == 4)) throw std::runtime_error("test failed: ekf state size");

	std::cout << "  ekf_construction: passed\n";
}

void test_ekf_linear_equivalence() {
	// For a linear system, the EKF must produce the same results as the
	// standard KF. This verifies the Jacobian pathway is wired correctly.
	KalmanFilter<double> kf(2, 1);
	ExtendedKalmanFilter<double> ekf(2, 1);

	using vec = mtl::vec::dense_vector<double>;
	using mat = mtl::mat::dense2D<double>;

	// 1D constant-velocity: F = [[1,1],[0,1]], H = [[1,0]]
	kf.F()(0, 0) = 1.0; kf.F()(0, 1) = 1.0;
	kf.F()(1, 0) = 0.0; kf.F()(1, 1) = 1.0;
	kf.H()(0, 0) = 1.0; kf.H()(0, 1) = 0.0;
	kf.Q()(0, 0) = 0.001; kf.Q()(0, 1) = 0.0;
	kf.Q()(1, 0) = 0.0;   kf.Q()(1, 1) = 0.001;
	kf.R()(0, 0) = 0.5;
	kf.P()(0, 0) = 100.0; kf.P()(0, 1) = 0.0;
	kf.P()(1, 0) = 0.0;   kf.P()(1, 1) = 100.0;

	// Set EKF to identical system via function callbacks.
	mat F_mat(2, 2);
	F_mat(0, 0) = 1.0; F_mat(0, 1) = 1.0;
	F_mat(1, 0) = 0.0; F_mat(1, 1) = 1.0;
	mat H_mat(1, 2);
	H_mat(0, 0) = 1.0; H_mat(0, 1) = 0.0;

	ekf.set_state_function(
		[&](const vec& x) -> vec { return F_mat * x; },
		[&](const vec&)   -> mat { return F_mat; }
	);
	ekf.set_observation_function(
		[&](const vec& x) -> vec { return H_mat * x; },
		[&](const vec&)   -> mat { return H_mat; }
	);
	ekf.Q()(0, 0) = 0.001; ekf.Q()(0, 1) = 0.0;
	ekf.Q()(1, 0) = 0.0;   ekf.Q()(1, 1) = 0.001;
	ekf.R()(0, 0) = 0.5;
	ekf.P()(0, 0) = 100.0; ekf.P()(0, 1) = 0.0;
	ekf.P()(1, 0) = 0.0;   ekf.P()(1, 1) = 100.0;

	std::mt19937 gen(42);
	std::normal_distribution<double> noise(0.0, 0.5);

	vec z(1);
	for (int t = 0; t < 50; ++t) {
		double true_pos = static_cast<double>(t);
		z[0] = true_pos + noise(gen);
		kf.predict();  kf.update(z);
		ekf.predict(); ekf.update(z);
	}

	if (!near(kf.state()[0], ekf.state()[0], 1e-10))
		throw std::runtime_error("test failed: EKF vs KF position diverged");
	if (!near(kf.state()[1], ekf.state()[1], 1e-10))
		throw std::runtime_error("test failed: EKF vs KF velocity diverged");

	std::cout << "  ekf_linear_equivalence: passed (pos diff="
	          << std::abs(kf.state()[0] - ekf.state()[0]) << ")\n";
}

void test_ekf_bearing_range_tracking() {
	// 2D target tracking with nonlinear observation model.
	// State: [x, y, vx, vy]  (position + velocity in 2D)
	// Observation: h(x) = [range, bearing] = [sqrt(x^2+y^2), atan2(y,x)]
	//
	// True trajectory: constant velocity from (100, 0) heading (0, 10).

	using vec = mtl::vec::dense_vector<double>;
	using mat = mtl::mat::dense2D<double>;
	constexpr double dt = 1.0;

	ExtendedKalmanFilter<double> ekf(4, 2);

	// State transition: constant velocity (linear, but we use EKF form).
	ekf.set_state_function(
		[](const vec& s) -> vec {
			constexpr double dt = 1.0;
			vec s_new(4);
			s_new[0] = s[0] + dt * s[2];  // x += vx*dt
			s_new[1] = s[1] + dt * s[3];  // y += vy*dt
			s_new[2] = s[2];              // vx constant
			s_new[3] = s[3];              // vy constant
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
			z[0] = std::sqrt(s[0] * s[0] + s[1] * s[1]);  // range
			z[1] = std::atan2(s[1], s[0]);                  // bearing
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
				H(0, 0) = x / r;   H(0, 1) = y / r;     // d(range)/d(x,y)
				H(1, 0) = -y / r2; H(1, 1) = x / r2;    // d(bearing)/d(x,y)
			}
			return H;
		}
	);

	// Noise parameters
	ekf.Q()(0, 0) = 0.1;  ekf.Q()(1, 1) = 0.1;
	ekf.Q()(2, 2) = 0.01; ekf.Q()(3, 3) = 0.01;
	ekf.R()(0, 0) = 1.0;   // range noise variance (m^2)
	ekf.R()(1, 1) = 0.001; // bearing noise variance (rad^2)

	// Initial estimate: roughly correct position, unknown velocity.
	ekf.state()[0] = 95.0;  // x
	ekf.state()[1] = 5.0;   // y
	ekf.state()[2] = 0.0;   // vx (unknown)
	ekf.state()[3] = 0.0;   // vy (unknown)
	ekf.P()(0, 0) = 100.0; ekf.P()(1, 1) = 100.0;
	ekf.P()(2, 2) = 10.0;  ekf.P()(3, 3) = 10.0;

	// Simulate true trajectory and noisy measurements.
	std::mt19937 gen(123);
	std::normal_distribution<double> range_noise(0.0, 1.0);
	std::normal_distribution<double> bearing_noise(0.0, std::sqrt(0.001));

	double true_x = 100.0, true_y = 0.0;
	double true_vx = 0.0,  true_vy = 10.0;

	vec z(2);
	for (int t = 0; t < 50; ++t) {
		true_x += true_vx * dt;
		true_y += true_vy * dt;

		double true_range   = std::sqrt(true_x * true_x + true_y * true_y);
		double true_bearing = std::atan2(true_y, true_x);
		z[0] = true_range   + range_noise(gen);
		z[1] = true_bearing + bearing_noise(gen);

		ekf.predict();
		ekf.update(z);
	}

	double est_x  = ekf.state()[0], est_y  = ekf.state()[1];
	double est_vx = ekf.state()[2], est_vy = ekf.state()[3];

	// After 50 steps the estimate should converge near the truth.
	if (!near(est_x, true_x, 5.0))
		throw std::runtime_error("test failed: EKF bearing-range x estimate off by "
			+ std::to_string(std::abs(est_x - true_x)));
	if (!near(est_y, true_y, 5.0))
		throw std::runtime_error("test failed: EKF bearing-range y estimate off by "
			+ std::to_string(std::abs(est_y - true_y)));
	if (!near(est_vx, true_vx, 2.0))
		throw std::runtime_error("test failed: EKF bearing-range vx estimate");
	if (!near(est_vy, true_vy, 2.0))
		throw std::runtime_error("test failed: EKF bearing-range vy estimate");

	std::cout << "  ekf_bearing_range_tracking: passed (pos=[" << est_x << ", " << est_y
	          << "], vel=[" << est_vx << ", " << est_vy << "])\n";
}

void test_ekf_validation() {
	bool caught = false;
	try { ExtendedKalmanFilter<double> ekf(0, 1); }
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: ekf should reject state_dim=0");

	caught = false;
	try { ExtendedKalmanFilter<double> ekf(2, 0); }
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: ekf should reject meas_dim=0");

	// Predict without setting functions should throw logic_error.
	caught = false;
	try {
		ExtendedKalmanFilter<double> ekf(2, 1);
		ekf.predict();
	} catch (const std::logic_error&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: ekf predict without functions should throw");

	std::cout << "  ekf_validation: passed\n";
}

// ========== UKF Tests ==========

void test_ukf_construction() {
	UnscentedKalmanFilter<double> ukf(4, 2);
	if (!(ukf.state_dim() == 4)) throw std::runtime_error("test failed: ukf state_dim");
	if (!(ukf.meas_dim() == 2))  throw std::runtime_error("test failed: ukf meas_dim");
	if (!(ukf.state().size() == 4)) throw std::runtime_error("test failed: ukf state size");

	std::cout << "  ukf_construction: passed\n";
}

void test_ukf_sigma_identity() {
	// For the identity function f(x) = x, predict should leave
	// mean unchanged and covariance = P + Q.
	UnscentedKalmanFilter<double> ukf(2, 1);

	using vec = mtl::vec::dense_vector<double>;
	ukf.set_state_function([](const vec& x) -> vec { return x; });
	ukf.set_observation_function([](const vec& x) -> vec {
		vec z(1); z[0] = x[0]; return z;
	});

	ukf.state()[0] = 5.0;
	ukf.state()[1] = -3.0;
	ukf.P()(0, 0) = 4.0; ukf.P()(0, 1) = 0.0;
	ukf.P()(1, 0) = 0.0; ukf.P()(1, 1) = 9.0;
	ukf.Q()(0, 0) = 0.01; ukf.Q()(0, 1) = 0.0;
	ukf.Q()(1, 0) = 0.0;  ukf.Q()(1, 1) = 0.01;

	ukf.predict();

	if (!near(ukf.state()[0], 5.0, 1e-10))
		throw std::runtime_error("test failed: ukf identity mean[0]");
	if (!near(ukf.state()[1], -3.0, 1e-10))
		throw std::runtime_error("test failed: ukf identity mean[1]");
	// P should be original P + Q
	if (!near(ukf.P()(0, 0), 4.01, 0.05))
		throw std::runtime_error("test failed: ukf identity P(0,0)");
	if (!near(ukf.P()(1, 1), 9.01, 0.05))
		throw std::runtime_error("test failed: ukf identity P(1,1)");

	std::cout << "  ukf_sigma_identity: passed\n";
}

void test_ukf_bearing_range_tracking() {
	// Same 2D bearing-range tracking as the EKF test, for comparison.
	using vec = mtl::vec::dense_vector<double>;
	constexpr double dt = 1.0;

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
	ukf.R()(0, 0) = 1.0;
	ukf.R()(1, 1) = 0.001;

	ukf.state()[0] = 95.0;
	ukf.state()[1] = 5.0;
	ukf.state()[2] = 0.0;
	ukf.state()[3] = 0.0;
	ukf.P()(0, 0) = 100.0; ukf.P()(1, 1) = 100.0;
	ukf.P()(2, 2) = 10.0;  ukf.P()(3, 3) = 10.0;

	std::mt19937 gen(123);
	std::normal_distribution<double> range_noise(0.0, 1.0);
	std::normal_distribution<double> bearing_noise(0.0, std::sqrt(0.001));

	double true_x = 100.0, true_y = 0.0;
	double true_vx = 0.0,  true_vy = 10.0;

	vec z(2);
	for (int t = 0; t < 50; ++t) {
		true_x += true_vx * dt;
		true_y += true_vy * dt;
		z[0] = std::sqrt(true_x * true_x + true_y * true_y) + range_noise(gen);
		z[1] = std::atan2(true_y, true_x) + bearing_noise(gen);
		ukf.predict();
		ukf.update(z);
	}

	double est_x = ukf.state()[0], est_y = ukf.state()[1];
	double est_vx = ukf.state()[2], est_vy = ukf.state()[3];

	if (!near(est_x, true_x, 5.0))
		throw std::runtime_error("test failed: UKF bearing-range x off by "
			+ std::to_string(std::abs(est_x - true_x)));
	if (!near(est_y, true_y, 5.0))
		throw std::runtime_error("test failed: UKF bearing-range y off by "
			+ std::to_string(std::abs(est_y - true_y)));
	if (!near(est_vx, true_vx, 2.0))
		throw std::runtime_error("test failed: UKF bearing-range vx");
	if (!near(est_vy, true_vy, 2.0))
		throw std::runtime_error("test failed: UKF bearing-range vy");

	std::cout << "  ukf_bearing_range_tracking: passed (pos=[" << est_x << ", " << est_y
	          << "], vel=[" << est_vx << ", " << est_vy << "])\n";
}

void test_ukf_validation() {
	bool caught = false;
	try { UnscentedKalmanFilter<double> ukf(0, 1); }
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: ukf should reject state_dim=0");

	caught = false;
	try {
		UnscentedKalmanFilter<double> ukf(2, 1);
		ukf.predict();
	} catch (const std::logic_error&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: ukf predict without function should throw");

	std::cout << "  ukf_validation: passed\n";
}

// ========== LMS Tests ==========

void test_lms_converges_to_known_filter() {
	// Setup: an "unknown" filter that we'll try to learn via LMS.
	// Unknown impulse response: [0.5, -0.3, 0.2]
	mtl::vec::dense_vector<double> unknown({0.5, -0.3, 0.2});

	LMSFilter<double> lms(3, 0.1);

	// Generate training data: random input, desired = unknown filter applied
	std::mt19937 gen(42);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);

	mtl::vec::dense_vector<double> delay(3, 0.0);
	for (int n = 0; n < 5000; ++n) {
		double x = dist(gen);
		// Compute desired output from unknown filter
		delay[2] = delay[1]; delay[1] = delay[0]; delay[0] = x;
		double d = unknown[0] * delay[0] + unknown[1] * delay[1] + unknown[2] * delay[2];
		lms.process(x, d);
	}

	// Weights should converge close to the unknown filter
	const auto& w = lms.weights();
	if (!(near(w[0], 0.5, 0.05))) throw std::runtime_error("test failed: LMS w[0]");
	if (!(near(w[1], -0.3, 0.05))) throw std::runtime_error("test failed: LMS w[1]");
	if (!(near(w[2], 0.2, 0.05))) throw std::runtime_error("test failed: LMS w[2]");

	std::cout << "  lms_converges: passed (w=[" << w[0] << ", " << w[1] << ", " << w[2] << "])\n";
}

void test_lms_reset() {
	LMSFilter<double> lms(4, 0.1);
	for (int i = 0; i < 100; ++i) lms.process(0.5, 1.0);
	if (!(lms.weights()[0] != 0.0))
		throw std::runtime_error("test failed: LMS weights should be non-zero after training");
	lms.reset();
	if (!(lms.weights()[0] == 0.0))
		throw std::runtime_error("test failed: LMS reset weights to zero");

	std::cout << "  lms_reset: passed\n";
}

void test_nlms() {
	NLMSFilter<double> nlms(3, 0.5, 1e-6);

	mtl::vec::dense_vector<double> unknown({0.5, -0.3, 0.2});
	std::mt19937 gen(42);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);

	mtl::vec::dense_vector<double> delay(3, 0.0);
	for (int n = 0; n < 2000; ++n) {
		double x = dist(gen);
		delay[2] = delay[1]; delay[1] = delay[0]; delay[0] = x;
		double d = unknown[0] * delay[0] + unknown[1] * delay[1] + unknown[2] * delay[2];
		nlms.process(x, d);
	}

	// NLMS should converge faster than LMS
	const auto& w = nlms.weights();
	if (!(near(w[0], 0.5, 0.05))) throw std::runtime_error("test failed: NLMS w[0]");
	if (!(near(w[1], -0.3, 0.05))) throw std::runtime_error("test failed: NLMS w[1]");
	if (!(near(w[2], 0.2, 0.05))) throw std::runtime_error("test failed: NLMS w[2]");

	std::cout << "  nlms: passed\n";
}

// ========== RLS Tests ==========

void test_rls_converges_fast() {
	// RLS should converge much faster than LMS for the same problem
	mtl::vec::dense_vector<double> unknown({0.5, -0.3, 0.2});

	RLSFilter<double> rls(3, 0.99, 1000.0);

	std::mt19937 gen(42);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);

	mtl::vec::dense_vector<double> delay(3, 0.0);
	for (int n = 0; n < 100; ++n) {  // RLS needs far fewer samples than LMS
		double x = dist(gen);
		delay[2] = delay[1]; delay[1] = delay[0]; delay[0] = x;
		double d = unknown[0] * delay[0] + unknown[1] * delay[1] + unknown[2] * delay[2];
		rls.process(x, d);
	}

	const auto& w = rls.weights();
	if (!(near(w[0], 0.5, 0.01))) throw std::runtime_error("test failed: RLS w[0]");
	if (!(near(w[1], -0.3, 0.01))) throw std::runtime_error("test failed: RLS w[1]");
	if (!(near(w[2], 0.2, 0.01))) throw std::runtime_error("test failed: RLS w[2]");

	std::cout << "  rls_converges_fast: passed (100 samples, w=[" << w[0] << ", "
	          << w[1] << ", " << w[2] << "])\n";
}

void test_rls_validation() {
	bool caught = false;
	try { RLSFilter<double> rls(0, 0.99); }
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: RLS should reject num_taps=0");

	caught = false;
	try { RLSFilter<double> rls(4, 1.5); }
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: RLS should reject lambda > 1");

	caught = false;
	try { RLSFilter<double> rls(4, 0.0); }
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: RLS should reject lambda = 0");

	std::cout << "  rls_validation: passed\n";
}

int main() {
	try {
		std::cout << "Estimation Tests\n";

		test_kalman_construction();
		test_kalman_constant_velocity();
		test_kalman_validation();
		test_ekf_construction();
		test_ekf_linear_equivalence();
		test_ekf_bearing_range_tracking();
		test_ekf_validation();
		test_ukf_construction();
		test_ukf_sigma_identity();
		test_ukf_bearing_range_tracking();
		test_ukf_validation();
		test_lms_converges_to_known_filter();
		test_lms_reset();
		test_nlms();
		test_rls_converges_fast();
		test_rls_validation();

		std::cout << "All estimation tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << '\n';
		return 1;
	}
}
