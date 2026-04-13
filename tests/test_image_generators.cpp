// test_image_generators.cpp: test synthetic image generators
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/image/image.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace sw::dsp;

bool near(double a, double b, double eps = 1e-4) {
	return std::abs(a - b) < eps;
}

// ========== Geometric Patterns ==========

void test_checkerboard() {
	auto img = checkerboard<float>(8, 8, 2);
	if (!(img.num_rows() == 8 && img.num_cols() == 8))
		throw std::runtime_error("test failed: checkerboard dimensions");

	// Block (0,0) should be high (1), block (0,1) should be low (0)
	if (!(near(img(0, 0), 1.0f))) throw std::runtime_error("test failed: checkerboard (0,0)");
	if (!(near(img(0, 2), 0.0f))) throw std::runtime_error("test failed: checkerboard (0,2)");
	if (!(near(img(2, 0), 0.0f))) throw std::runtime_error("test failed: checkerboard (2,0)");
	if (!(near(img(2, 2), 1.0f))) throw std::runtime_error("test failed: checkerboard (2,2)");

	std::cout << "  checkerboard: passed\n";
}

void test_stripes_horizontal() {
	auto img = stripes_horizontal<float>(8, 4, 2);
	// Rows 0-1: high, rows 2-3: low, rows 4-5: high, rows 6-7: low
	if (!(near(img(0, 0), 1.0f))) throw std::runtime_error("test failed: h-stripes row 0");
	if (!(near(img(2, 0), 0.0f))) throw std::runtime_error("test failed: h-stripes row 2");
	// All columns in a row should be the same
	if (!(near(img(0, 0), img(0, 3)))) throw std::runtime_error("test failed: h-stripes uniform cols");

	std::cout << "  stripes_horizontal: passed\n";
}

void test_stripes_vertical() {
	auto img = stripes_vertical<float>(4, 8, 2);
	if (!(near(img(0, 0), 1.0f))) throw std::runtime_error("test failed: v-stripes col 0");
	if (!(near(img(0, 2), 0.0f))) throw std::runtime_error("test failed: v-stripes col 2");
	// All rows in a column should be the same
	if (!(near(img(0, 0), img(3, 0)))) throw std::runtime_error("test failed: v-stripes uniform rows");

	std::cout << "  stripes_vertical: passed\n";
}

void test_grid() {
	auto img = grid<float>(10, 10, 5);
	// Row 0 and col 0 should be lines
	if (!(near(img(0, 3), 1.0f))) throw std::runtime_error("test failed: grid row 0");
	if (!(near(img(3, 0), 1.0f))) throw std::runtime_error("test failed: grid col 0");
	// Interior point not on grid
	if (!(near(img(1, 1), 0.0f))) throw std::runtime_error("test failed: grid interior");
	// Row 5 should be a line
	if (!(near(img(5, 3), 1.0f))) throw std::runtime_error("test failed: grid row 5");

	std::cout << "  grid: passed\n";
}

// ========== Gradients ==========

void test_gradient_horizontal() {
	auto img = gradient_horizontal<float>(4, 11);
	// First column should be 0, last column should be 1
	if (!(near(img(0, 0), 0.0, 1e-3))) throw std::runtime_error("test failed: h-gradient start");
	if (!(near(img(0, 10), 1.0, 1e-3))) throw std::runtime_error("test failed: h-gradient end");
	// Middle should be ~0.5
	if (!(near(img(0, 5), 0.5, 1e-2))) throw std::runtime_error("test failed: h-gradient middle");
	// All rows should be identical
	if (!(near(img(0, 5), img(3, 5), 1e-6))) throw std::runtime_error("test failed: h-gradient row invariant");

	std::cout << "  gradient_horizontal: passed\n";
}

void test_gradient_vertical() {
	auto img = gradient_vertical<float>(11, 4);
	if (!(near(img(0, 0), 0.0, 1e-3))) throw std::runtime_error("test failed: v-gradient start");
	if (!(near(img(10, 0), 1.0, 1e-3))) throw std::runtime_error("test failed: v-gradient end");
	if (!(near(img(5, 0), 0.5, 1e-2))) throw std::runtime_error("test failed: v-gradient middle");

	std::cout << "  gradient_vertical: passed\n";
}

void test_gradient_radial() {
	auto img = gradient_radial<float>(11, 11);
	// Center should be center_val (1.0)
	if (!(near(img(5, 5), 1.0, 1e-2))) throw std::runtime_error("test failed: radial center");
	// Corners should be near edge_val (0.0)
	if (!(near(img(0, 0), 0.0, 1e-2))) throw std::runtime_error("test failed: radial corner");
	// Midpoint between center and corner should be ~0.5
	double mid_val = static_cast<double>(img(5, 0));
	if (!(mid_val > 0.2 && mid_val < 0.8))
		throw std::runtime_error("test failed: radial mid-distance");

	std::cout << "  gradient_radial: passed\n";
}

// ========== Shapes ==========

void test_gaussian_blob() {
	auto img = gaussian_blob<float>(21, 21, 3.0);
	// Center should be maximum (amplitude = 1)
	if (!(near(img(10, 10), 1.0, 1e-3))) throw std::runtime_error("test failed: blob center");
	// Far from center should be near zero
	if (!(static_cast<double>(img(0, 0)) < 0.01))
		throw std::runtime_error("test failed: blob corner should be near zero");
	// Should be symmetric
	if (!(near(img(10, 7), img(10, 13), 1e-3)))
		throw std::runtime_error("test failed: blob symmetry");

	std::cout << "  gaussian_blob: passed\n";
}

void test_circle() {
	auto img = circle<float>(21, 21, 5);
	// Center should be foreground
	if (!(near(img(10, 10), 1.0f))) throw std::runtime_error("test failed: circle center");
	// Corner should be background
	if (!(near(img(0, 0), 0.0f))) throw std::runtime_error("test failed: circle corner");
	// Just inside radius
	if (!(near(img(10, 14), 1.0f))) throw std::runtime_error("test failed: circle edge inside");
	// Just outside radius
	if (!(near(img(10, 16), 0.0f))) throw std::runtime_error("test failed: circle edge outside");

	std::cout << "  circle: passed\n";
}

void test_rectangle() {
	auto img = rectangle<float>(10, 10, 2, 3, 4, 5);
	// Inside rectangle
	if (!(near(img(3, 5), 1.0f))) throw std::runtime_error("test failed: rect inside");
	// Outside rectangle
	if (!(near(img(0, 0), 0.0f))) throw std::runtime_error("test failed: rect outside");
	// Boundary
	if (!(near(img(2, 3), 1.0f))) throw std::runtime_error("test failed: rect top-left corner");
	if (!(near(img(5, 7), 1.0f))) throw std::runtime_error("test failed: rect bottom-right inside");
	if (!(near(img(6, 8), 0.0f))) throw std::runtime_error("test failed: rect past bottom-right");

	std::cout << "  rectangle: passed\n";
}

// ========== Noise ==========

void test_uniform_noise() {
	auto img = uniform_noise_image<float>(50, 50, 0.0f, 1.0f, 42);
	if (!(img.num_rows() == 50 && img.num_cols() == 50))
		throw std::runtime_error("test failed: uniform noise dimensions");

	// All values should be in [0, 1]
	double sum = 0;
	for (std::size_t r = 0; r < 50; ++r) {
		for (std::size_t c = 0; c < 50; ++c) {
			double v = static_cast<double>(img(r, c));
			if (!(v >= 0.0 && v <= 1.0))
				throw std::runtime_error("test failed: uniform noise out of range");
			sum += v;
		}
	}
	// Mean should be near 0.5
	double mean = sum / 2500.0;
	if (!(near(mean, 0.5, 0.05)))
		throw std::runtime_error("test failed: uniform noise mean");

	std::cout << "  uniform_noise: passed (mean=" << mean << ")\n";
}

void test_gaussian_noise() {
	auto img = gaussian_noise_image<float>(50, 50, 0.0f, 1.0f, 42);

	double sum = 0, sum_sq = 0;
	for (std::size_t r = 0; r < 50; ++r) {
		for (std::size_t c = 0; c < 50; ++c) {
			double v = static_cast<double>(img(r, c));
			sum += v;
			sum_sq += v * v;
		}
	}
	double mean = sum / 2500.0;
	double variance = sum_sq / 2500.0 - mean * mean;

	// Mean should be near 0, variance near 1
	if (!(near(mean, 0.0, 0.1)))
		throw std::runtime_error("test failed: gaussian noise mean");
	if (!(near(variance, 1.0, 0.2)))
		throw std::runtime_error("test failed: gaussian noise variance");

	std::cout << "  gaussian_noise: passed (mean=" << mean << ", var=" << variance << ")\n";
}

void test_salt_and_pepper() {
	auto img = salt_and_pepper<float>(100, 100, 0.1, 0.0f, 1.0f, 42);

	int salt_count = 0, pepper_count = 0, mid_count = 0;
	for (std::size_t r = 0; r < 100; ++r) {
		for (std::size_t c = 0; c < 100; ++c) {
			float v = img(r, c);
			if (near(v, 0.0f, 1e-6)) ++pepper_count;
			else if (near(v, 1.0f, 1e-6)) ++salt_count;
			else ++mid_count;
		}
	}

	// density=0.1 means ~5% salt + ~5% pepper = ~10% total noise
	int noise_total = salt_count + pepper_count;
	if (!(noise_total > 500 && noise_total < 1500))
		throw std::runtime_error("test failed: salt_and_pepper noise density");
	if (!(mid_count > 8500))
		throw std::runtime_error("test failed: salt_and_pepper mid pixels");

	std::cout << "  salt_and_pepper: passed (salt=" << salt_count
	          << ", pepper=" << pepper_count << ", mid=" << mid_count << ")\n";
}

// ========== Determinism ==========

void test_deterministic_seed() {
	auto img1 = uniform_noise_image<float>(10, 10, 0.0f, 1.0f, 123);
	auto img2 = uniform_noise_image<float>(10, 10, 0.0f, 1.0f, 123);

	for (std::size_t r = 0; r < 10; ++r)
		for (std::size_t c = 0; c < 10; ++c)
			if (!(img1(r, c) == img2(r, c)))
				throw std::runtime_error("test failed: same seed should produce identical images");

	std::cout << "  deterministic_seed: passed\n";
}

int main() {
	try {
		std::cout << "Image Generator Tests\n";

		// Geometric patterns
		test_checkerboard();
		test_stripes_horizontal();
		test_stripes_vertical();
		test_grid();

		// Gradients
		test_gradient_horizontal();
		test_gradient_vertical();
		test_gradient_radial();

		// Shapes
		test_gaussian_blob();
		test_circle();
		test_rectangle();

		// Noise
		test_uniform_noise();
		test_gaussian_noise();
		test_salt_and_pepper();

		// Determinism
		test_deterministic_seed();

		std::cout << "All image generator tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << '\n';
		return 1;
	}
}
