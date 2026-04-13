#pragma once
// generators.hpp: synthetic image generator functions
//
// Free functions that produce standard test images as mtl::mat::dense2D<T>.
// All generators are templated on DspField T for mixed-precision support.
//
// Categories:
//   Geometric patterns: checkerboard, stripes, grid
//   Gradients: horizontal, vertical, radial
//   Shapes: gaussian_blob, circle, rectangle
//   Noise: uniform, gaussian, salt-and-pepper
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <random>
#include <mtl/mat/dense2D.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

// ============================================================================
// Geometric patterns
// ============================================================================

// Alternating blocks of low/high values.
// Ideal for testing edge detectors and morphological operations.
template <DspField T>
mtl::mat::dense2D<T> checkerboard(std::size_t rows, std::size_t cols,
                                   std::size_t block_size,
                                   T low = T{}, T high = static_cast<T>(1)) {
	mtl::mat::dense2D<T> img(rows, cols);
	for (std::size_t r = 0; r < rows; ++r)
		for (std::size_t c = 0; c < cols; ++c)
			img(r, c) = ((r / block_size + c / block_size) % 2 == 0) ? high : low;
	return img;
}

// Horizontal stripes of alternating low/high values.
template <DspField T>
mtl::mat::dense2D<T> stripes_horizontal(std::size_t rows, std::size_t cols,
                                         std::size_t stripe_width,
                                         T low = T{}, T high = static_cast<T>(1)) {
	mtl::mat::dense2D<T> img(rows, cols);
	for (std::size_t r = 0; r < rows; ++r)
		for (std::size_t c = 0; c < cols; ++c)
			img(r, c) = ((r / stripe_width) % 2 == 0) ? high : low;
	return img;
}

// Vertical stripes of alternating low/high values.
template <DspField T>
mtl::mat::dense2D<T> stripes_vertical(std::size_t rows, std::size_t cols,
                                       std::size_t stripe_width,
                                       T low = T{}, T high = static_cast<T>(1)) {
	mtl::mat::dense2D<T> img(rows, cols);
	for (std::size_t r = 0; r < rows; ++r)
		for (std::size_t c = 0; c < cols; ++c)
			img(r, c) = ((c / stripe_width) % 2 == 0) ? high : low;
	return img;
}

// Grid pattern: thin lines at regular spacing.
template <DspField T>
mtl::mat::dense2D<T> grid(std::size_t rows, std::size_t cols,
                           std::size_t spacing,
                           T background = T{}, T line = static_cast<T>(1)) {
	mtl::mat::dense2D<T> img(rows, cols);
	for (std::size_t r = 0; r < rows; ++r)
		for (std::size_t c = 0; c < cols; ++c)
			img(r, c) = (r % spacing == 0 || c % spacing == 0) ? line : background;
	return img;
}

// ============================================================================
// Gradients
// ============================================================================

// Linear ramp from left (start) to right (end).
template <DspField T>
	requires ConvertibleToDouble<T>
mtl::mat::dense2D<T> gradient_horizontal(std::size_t rows, std::size_t cols,
                                          T start = T{}, T end = static_cast<T>(1)) {
	mtl::mat::dense2D<T> img(rows, cols);
	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			double t = (cols > 1) ? static_cast<double>(c) / static_cast<double>(cols - 1) : 0.0;
			img(r, c) = static_cast<T>(static_cast<double>(start) * (1.0 - t)
			                           + static_cast<double>(end) * t);
		}
	}
	return img;
}

// Linear ramp from top (start) to bottom (end).
template <DspField T>
	requires ConvertibleToDouble<T>
mtl::mat::dense2D<T> gradient_vertical(std::size_t rows, std::size_t cols,
                                        T start = T{}, T end = static_cast<T>(1)) {
	mtl::mat::dense2D<T> img(rows, cols);
	for (std::size_t r = 0; r < rows; ++r) {
		double t = (rows > 1) ? static_cast<double>(r) / static_cast<double>(rows - 1) : 0.0;
		T val = static_cast<T>(static_cast<double>(start) * (1.0 - t)
		                       + static_cast<double>(end) * t);
		for (std::size_t c = 0; c < cols; ++c)
			img(r, c) = val;
	}
	return img;
}

// Radial gradient: center_val at image center, edge_val at corners.
template <DspField T>
	requires ConvertibleToDouble<T>
mtl::mat::dense2D<T> gradient_radial(std::size_t rows, std::size_t cols,
                                      T center_val = static_cast<T>(1),
                                      T edge_val = T{}) {
	mtl::mat::dense2D<T> img(rows, cols);
	double cy = static_cast<double>(rows - 1) * 0.5;
	double cx = static_cast<double>(cols - 1) * 0.5;
	double max_dist = std::sqrt(cy * cy + cx * cx);
	if (max_dist < 1e-20) max_dist = 1.0;

	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			double dy = static_cast<double>(r) - cy;
			double dx = static_cast<double>(c) - cx;
			double t = std::sqrt(dy * dy + dx * dx) / max_dist;
			if (t > 1.0) t = 1.0;
			img(r, c) = static_cast<T>(static_cast<double>(center_val) * (1.0 - t)
			                           + static_cast<double>(edge_val) * t);
		}
	}
	return img;
}

// ============================================================================
// Shapes
// ============================================================================

// 2D Gaussian blob centered in the image.
// Useful for testing separable filters and blur operations.
template <DspField T>
	requires ConvertibleToDouble<T>
mtl::mat::dense2D<T> gaussian_blob(std::size_t rows, std::size_t cols,
                                    double sigma,
                                    T amplitude = static_cast<T>(1)) {
	mtl::mat::dense2D<T> img(rows, cols);
	double cy = static_cast<double>(rows - 1) * 0.5;
	double cx = static_cast<double>(cols - 1) * 0.5;
	double inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);

	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			double dy = static_cast<double>(r) - cy;
			double dx = static_cast<double>(c) - cx;
			double val = std::exp(-(dy * dy + dx * dx) * inv_2sigma2);
			img(r, c) = static_cast<T>(static_cast<double>(amplitude) * val);
		}
	}
	return img;
}

// Filled circle centered in the image.
template <DspField T>
	requires ConvertibleToDouble<T>
mtl::mat::dense2D<T> circle(std::size_t rows, std::size_t cols,
                             std::size_t radius,
                             T foreground = static_cast<T>(1), T background = T{}) {
	mtl::mat::dense2D<T> img(rows, cols);
	double cy = static_cast<double>(rows - 1) * 0.5;
	double cx = static_cast<double>(cols - 1) * 0.5;
	double r2 = static_cast<double>(radius) * static_cast<double>(radius);

	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			double dy = static_cast<double>(r) - cy;
			double dx = static_cast<double>(c) - cx;
			img(r, c) = (dy * dy + dx * dx <= r2) ? foreground : background;
		}
	}
	return img;
}

// Filled rectangle at specified position (y, x) with dimensions (h, w).
template <DspField T>
mtl::mat::dense2D<T> rectangle(std::size_t rows, std::size_t cols,
                                std::size_t y, std::size_t x,
                                std::size_t h, std::size_t w,
                                T foreground = static_cast<T>(1), T background = T{}) {
	mtl::mat::dense2D<T> img(rows, cols);
	for (std::size_t r = 0; r < rows; ++r)
		for (std::size_t c = 0; c < cols; ++c)
			img(r, c) = (r >= y && r < y + h && c >= x && c < x + w)
				? foreground : background;
	return img;
}

// ============================================================================
// Noise
// ============================================================================

// Uniform random noise in [low, high].
template <DspField T>
	requires ConvertibleToDouble<T>
mtl::mat::dense2D<T> uniform_noise_image(std::size_t rows, std::size_t cols,
                                          T low = T{}, T high = static_cast<T>(1),
                                          unsigned seed = 0) {
	mtl::mat::dense2D<T> img(rows, cols);
	std::mt19937 gen(seed);
	std::uniform_real_distribution<double> dist(static_cast<double>(low),
	                                            static_cast<double>(high));
	for (std::size_t r = 0; r < rows; ++r)
		for (std::size_t c = 0; c < cols; ++c)
			img(r, c) = static_cast<T>(dist(gen));
	return img;
}

// Gaussian random noise with specified mean and standard deviation.
template <DspField T>
	requires ConvertibleToDouble<T>
mtl::mat::dense2D<T> gaussian_noise_image(std::size_t rows, std::size_t cols,
                                           T mean = T{}, T stddev = static_cast<T>(1),
                                           unsigned seed = 0) {
	mtl::mat::dense2D<T> img(rows, cols);
	std::mt19937 gen(seed);
	std::normal_distribution<double> dist(static_cast<double>(mean),
	                                      static_cast<double>(stddev));
	for (std::size_t r = 0; r < rows; ++r)
		for (std::size_t c = 0; c < cols; ++c)
			img(r, c) = static_cast<T>(dist(gen));
	return img;
}

// Salt-and-pepper (impulse) noise.
// Randomly sets a fraction (density) of pixels to low or high values.
template <DspField T>
	requires ConvertibleToDouble<T>
mtl::mat::dense2D<T> salt_and_pepper(std::size_t rows, std::size_t cols,
                                      double density = 0.05,
                                      T low = T{}, T high = static_cast<T>(1),
                                      unsigned seed = 0) {
	// Start with a mid-gray image
	mtl::mat::dense2D<T> img(rows, cols);
	T mid = static_cast<T>((static_cast<double>(low) + static_cast<double>(high)) * 0.5);
	for (std::size_t r = 0; r < rows; ++r)
		for (std::size_t c = 0; c < cols; ++c)
			img(r, c) = mid;

	std::mt19937 gen(seed);
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	double half_density = density * 0.5;

	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			double p = dist(gen);
			if (p < half_density)
				img(r, c) = low;   // pepper
			else if (p < density)
				img(r, c) = high;  // salt
		}
	}
	return img;
}

} // namespace sw::dsp
