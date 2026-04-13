#pragma once
// edge.hpp: edge detection operators
//
// Sobel, Prewitt gradient operators; gradient magnitude;
// Canny edge detector with non-maximum suppression and hysteresis.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <mtl/mat/dense2D.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/image/image.hpp>
#include <sw/dsp/image/convolve2d.hpp>
#include <sw/dsp/image/separable.hpp>

namespace sw::dsp {

// ---------- Gradient operators ----------

// Sobel gradient in the X direction (horizontal edges).
// ksize: 3 (default). Uses separable decomposition.
template <DspField T>
mtl::mat::dense2D<T> sobel_x(const mtl::mat::dense2D<T>& image,
                              BorderMode border = BorderMode::reflect_101) {
	// Sobel X kernel:  [-1 0 1; -2 0 2; -1 0 1]
	// Separable: row=[-1,0,1] (horizontal derivative), col=[1,2,1] (vertical smooth)
	mtl::vec::dense_vector<T> row_kernel(3);
	row_kernel[0] = static_cast<T>(-1);
	row_kernel[1] = static_cast<T>(0);
	row_kernel[2] = static_cast<T>(1);

	mtl::vec::dense_vector<T> col_kernel(3);
	col_kernel[0] = static_cast<T>(1);
	col_kernel[1] = static_cast<T>(2);
	col_kernel[2] = static_cast<T>(1);

	return separable_filter(image, row_kernel, col_kernel, border);
}

// Sobel gradient in the Y direction (vertical edges).
template <DspField T>
mtl::mat::dense2D<T> sobel_y(const mtl::mat::dense2D<T>& image,
                              BorderMode border = BorderMode::reflect_101) {
	// Sobel Y kernel: [-1 -2 -1; 0 0 0; 1 2 1]
	// Separable: row=[1,2,1] (horizontal smooth), col=[-1,0,1] (vertical derivative)
	mtl::vec::dense_vector<T> row_kernel(3);
	row_kernel[0] = static_cast<T>(1);
	row_kernel[1] = static_cast<T>(2);
	row_kernel[2] = static_cast<T>(1);

	mtl::vec::dense_vector<T> col_kernel(3);
	col_kernel[0] = static_cast<T>(-1);
	col_kernel[1] = static_cast<T>(0);
	col_kernel[2] = static_cast<T>(1);

	return separable_filter(image, row_kernel, col_kernel, border);
}

// Prewitt gradient in the X direction.
template <DspField T>
mtl::mat::dense2D<T> prewitt_x(const mtl::mat::dense2D<T>& image,
                                BorderMode border = BorderMode::reflect_101) {
	// Prewitt X kernel: [-1 0 1; -1 0 1; -1 0 1]
	// Separable: row=[-1,0,1] (horizontal derivative), col=[1,1,1] (vertical smooth)
	mtl::vec::dense_vector<T> row_kernel(3);
	row_kernel[0] = static_cast<T>(-1);
	row_kernel[1] = static_cast<T>(0);
	row_kernel[2] = static_cast<T>(1);

	mtl::vec::dense_vector<T> col_kernel(3);
	col_kernel[0] = static_cast<T>(1);
	col_kernel[1] = static_cast<T>(1);
	col_kernel[2] = static_cast<T>(1);

	return separable_filter(image, row_kernel, col_kernel, border);
}

// Prewitt gradient in the Y direction.
template <DspField T>
mtl::mat::dense2D<T> prewitt_y(const mtl::mat::dense2D<T>& image,
                                BorderMode border = BorderMode::reflect_101) {
	// Prewitt Y kernel: [-1 -1 -1; 0 0 0; 1 1 1]
	// Separable: row=[1,1,1] (horizontal smooth), col=[-1,0,1] (vertical derivative)
	mtl::vec::dense_vector<T> row_kernel(3);
	row_kernel[0] = static_cast<T>(1);
	row_kernel[1] = static_cast<T>(1);
	row_kernel[2] = static_cast<T>(1);

	mtl::vec::dense_vector<T> col_kernel(3);
	col_kernel[0] = static_cast<T>(-1);
	col_kernel[1] = static_cast<T>(0);
	col_kernel[2] = static_cast<T>(1);

	return separable_filter(image, row_kernel, col_kernel, border);
}

// Gradient magnitude: sqrt(gx^2 + gy^2)
template <DspField T>
	requires ConvertibleToDouble<T>
mtl::mat::dense2D<T> gradient_magnitude(const mtl::mat::dense2D<T>& gx,
                                        const mtl::mat::dense2D<T>& gy) {
	std::size_t rows = gx.num_rows();
	std::size_t cols = gx.num_cols();
	if (rows != gy.num_rows() || cols != gy.num_cols())
		throw std::invalid_argument("gradient_magnitude: gx and gy must have same dimensions");
	mtl::mat::dense2D<T> mag(rows, cols);
	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			using std::sqrt;
			double dx = static_cast<double>(gx(r, c));
			double dy = static_cast<double>(gy(r, c));
			mag(r, c) = static_cast<T>(sqrt(dx * dx + dy * dy));
		}
	}
	return mag;
}

// ---------- Canny edge detector ----------

// Canny edge detection with non-maximum suppression and hysteresis.
//
// Returns a binary edge map where edge pixels are T{1} and non-edges are T{0}.
// Operates on single-channel (grayscale) input only.
//
// Parameters:
//   low_threshold:  lower hysteresis threshold
//   high_threshold: upper hysteresis threshold
//   sigma:          Gaussian blur sigma (0 to skip blurring)
template <DspField T>
	requires ConvertibleToDouble<T>
mtl::mat::dense2D<T> canny(const mtl::mat::dense2D<T>& image,
                           double low_threshold,
                           double high_threshold,
                           double sigma = 1.0) {
	std::size_t rows = image.num_rows();
	std::size_t cols = image.num_cols();

	// Step 1: Gaussian blur (noise reduction)
	mtl::mat::dense2D<T> blurred = (sigma > 0.0)
		? gaussian_blur(image, sigma)
		: image;

	// Step 2: Compute gradients
	auto gx = sobel_x(blurred);
	auto gy = sobel_y(blurred);
	auto mag = gradient_magnitude(gx, gy);

	// Step 3: Non-maximum suppression
	// Quantize gradient direction to 4 angles (0, 45, 90, 135 degrees)
	// and suppress pixels that are not local maxima along the gradient direction.
	mtl::mat::dense2D<T> nms(rows, cols);
	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			nms(r, c) = T{};
		}
	}

	for (std::size_t r = 1; r + 1 < rows; ++r) {
		for (std::size_t c = 1; c + 1 < cols; ++c) {
			double dx = static_cast<double>(gx(r, c));
			double dy = static_cast<double>(gy(r, c));
			using std::atan2;
			double angle = atan2(dy, dx);
			// Normalize to [0, pi)
			if (angle < 0.0) angle += 3.14159265358979323846;

			double m = static_cast<double>(mag(r, c));
			double m1, m2;

			// 0 degrees (horizontal)
			if ((angle < 0.3927) || (angle >= 2.7489)) {
				m1 = static_cast<double>(mag(r, c - 1));
				m2 = static_cast<double>(mag(r, c + 1));
			}
			// 45 degrees
			else if (angle < 1.1781) {
				m1 = static_cast<double>(mag(r - 1, c + 1));
				m2 = static_cast<double>(mag(r + 1, c - 1));
			}
			// 90 degrees (vertical)
			else if (angle < 1.9635) {
				m1 = static_cast<double>(mag(r - 1, c));
				m2 = static_cast<double>(mag(r + 1, c));
			}
			// 135 degrees
			else {
				m1 = static_cast<double>(mag(r - 1, c - 1));
				m2 = static_cast<double>(mag(r + 1, c + 1));
			}

			if (m >= m1 && m >= m2) {
				nms(r, c) = mag(r, c);
			}
		}
	}

	// Step 4: Double threshold and hysteresis
	// Mark strong edges, weak edges, and then connect weak edges
	// that are adjacent to strong edges.
	mtl::mat::dense2D<T> edges(rows, cols);
	for (std::size_t r = 0; r < rows; ++r)
		for (std::size_t c = 0; c < cols; ++c)
			edges(r, c) = T{};

	T strong = static_cast<T>(1);
	T weak = static_cast<T>(0.5);

	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			double val = static_cast<double>(nms(r, c));
			if (val >= high_threshold) {
				edges(r, c) = strong;
			} else if (val >= low_threshold) {
				edges(r, c) = weak;
			}
		}
	}

	// Hysteresis: promote weak edges adjacent to strong edges.
	// Simple iterative approach.
	bool changed = true;
	while (changed) {
		changed = false;
		for (std::size_t r = 1; r + 1 < rows; ++r) {
			for (std::size_t c = 1; c + 1 < cols; ++c) {
				if (static_cast<double>(edges(r, c)) < 0.4 ||
				    static_cast<double>(edges(r, c)) > 0.9) continue;
				// Check 8-connected neighbors for strong edge
				bool has_strong = false;
				for (int dr = -1; dr <= 1 && !has_strong; ++dr) {
					for (int dc = -1; dc <= 1 && !has_strong; ++dc) {
						if (dr == 0 && dc == 0) continue;
						std::size_t nr = static_cast<std::size_t>(static_cast<int>(r) + dr);
						std::size_t nc = static_cast<std::size_t>(static_cast<int>(c) + dc);
						if (static_cast<double>(edges(nr, nc)) > 0.9) {
							has_strong = true;
						}
					}
				}
				if (has_strong) {
					edges(r, c) = strong;
					changed = true;
				}
			}
		}
	}

	// Suppress remaining weak edges
	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			if (static_cast<double>(edges(r, c)) < 0.9) {
				edges(r, c) = T{};
			}
		}
	}

	return edges;
}

} // namespace sw::dsp
