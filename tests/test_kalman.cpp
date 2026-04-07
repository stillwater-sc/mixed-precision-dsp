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
