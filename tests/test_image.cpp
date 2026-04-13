// test_image.cpp: test image processing — convolution, separable, morphology, edge
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

// ---------- Helper: create synthetic test images ----------

// Create a constant image
mtl::mat::dense2D<double> make_constant(std::size_t rows, std::size_t cols, double val) {
	mtl::mat::dense2D<double> img(rows, cols);
	for (std::size_t r = 0; r < rows; ++r)
		for (std::size_t c = 0; c < cols; ++c)
			img(r, c) = val;
	return img;
}

// Create a horizontal ramp: pixel value = column index
mtl::mat::dense2D<double> make_horizontal_ramp(std::size_t rows, std::size_t cols) {
	mtl::mat::dense2D<double> img(rows, cols);
	for (std::size_t r = 0; r < rows; ++r)
		for (std::size_t c = 0; c < cols; ++c)
			img(r, c) = static_cast<double>(c);
	return img;
}

// Create a vertical ramp: pixel value = row index
mtl::mat::dense2D<double> make_vertical_ramp(std::size_t rows, std::size_t cols) {
	mtl::mat::dense2D<double> img(rows, cols);
	for (std::size_t r = 0; r < rows; ++r)
		for (std::size_t c = 0; c < cols; ++c)
			img(r, c) = static_cast<double>(r);
	return img;
}

// Create a step edge: left half = 0, right half = 1
mtl::mat::dense2D<double> make_step_edge(std::size_t rows, std::size_t cols) {
	mtl::mat::dense2D<double> img(rows, cols);
	std::size_t mid = cols / 2;
	for (std::size_t r = 0; r < rows; ++r)
		for (std::size_t c = 0; c < cols; ++c)
			img(r, c) = (c >= mid) ? 1.0 : 0.0;
	return img;
}

// Create an identity kernel (single 1 at center)
mtl::mat::dense2D<double> make_identity_kernel() {
	mtl::mat::dense2D<double> k(3, 3);
	for (std::size_t r = 0; r < 3; ++r)
		for (std::size_t c = 0; c < 3; ++c)
			k(r, c) = 0.0;
	k(1, 1) = 1.0;
	return k;
}

// ========== Image Container Tests ==========

void test_image_construction() {
	Image<double, 3> rgb(10, 20);
	if (!(rgb.rows() == 10)) throw std::runtime_error("test failed: Image rows");
	if (!(rgb.cols() == 20)) throw std::runtime_error("test failed: Image cols");
	if (!(rgb.channels() == 3)) throw std::runtime_error("test failed: Image channels");

	// Default construction
	GrayImage<double> gray;
	RGBImage<float> rgb_f(5, 5);
	RGBAImage<double> rgba(8, 8);
	if (!(rgba.channels() == 4)) throw std::runtime_error("test failed: RGBA channels");

	std::cout << "  image_construction: passed\n";
}

void test_apply_per_channel() {
	RGBImage<double> rgb(4, 4);
	// Fill each channel with a different constant
	for (std::size_t r = 0; r < 4; ++r) {
		for (std::size_t c = 0; c < 4; ++c) {
			rgb[0](r, c) = 1.0;
			rgb[1](r, c) = 2.0;
			rgb[2](r, c) = 3.0;
		}
	}

	// Double each channel
	auto result = apply_per_channel(rgb, [](const mtl::mat::dense2D<double>& plane) {
		mtl::mat::dense2D<double> out(plane.num_rows(), plane.num_cols());
		for (std::size_t r = 0; r < plane.num_rows(); ++r)
			for (std::size_t c = 0; c < plane.num_cols(); ++c)
				out(r, c) = plane(r, c) * 2.0;
		return out;
	});

	if (!(near(result[0](0, 0), 2.0))) throw std::runtime_error("test failed: apply_per_channel R");
	if (!(near(result[1](0, 0), 4.0))) throw std::runtime_error("test failed: apply_per_channel G");
	if (!(near(result[2](0, 0), 6.0))) throw std::runtime_error("test failed: apply_per_channel B");

	std::cout << "  apply_per_channel: passed\n";
}

void test_rgb_to_gray() {
	RGBImage<double> rgb(2, 2);
	// White pixel: R=G=B=1.0
	for (std::size_t r = 0; r < 2; ++r) {
		for (std::size_t c = 0; c < 2; ++c) {
			rgb[0](r, c) = 1.0;
			rgb[1](r, c) = 1.0;
			rgb[2](r, c) = 1.0;
		}
	}
	auto gray = rgb_to_gray(rgb);
	// 0.299 + 0.587 + 0.114 = 1.0
	if (!(near(gray(0, 0), 1.0, 1e-3)))
		throw std::runtime_error("test failed: rgb_to_gray white");

	// Pure red
	for (std::size_t r = 0; r < 2; ++r) {
		for (std::size_t c = 0; c < 2; ++c) {
			rgb[0](r, c) = 1.0;
			rgb[1](r, c) = 0.0;
			rgb[2](r, c) = 0.0;
		}
	}
	gray = rgb_to_gray(rgb);
	if (!(near(gray(0, 0), 0.299, 1e-3)))
		throw std::runtime_error("test failed: rgb_to_gray red");

	std::cout << "  rgb_to_gray: passed\n";
}

// ========== Border Handling Tests ==========

void test_border_modes() {
	// 4-pixel row image: [10, 20, 30, 40]
	mtl::mat::dense2D<double> img(1, 4);
	img(0, 0) = 10; img(0, 1) = 20; img(0, 2) = 30; img(0, 3) = 40;

	// Zero padding
	double val = fetch_pixel(img, 0, -1, BorderMode::zero);
	if (!(near(val, 0.0))) throw std::runtime_error("test failed: border zero");

	// Replicate
	val = fetch_pixel(img, 0, -1, BorderMode::replicate);
	if (!(near(val, 10.0))) throw std::runtime_error("test failed: border replicate left");
	val = fetch_pixel(img, 0, 5, BorderMode::replicate);
	if (!(near(val, 40.0))) throw std::runtime_error("test failed: border replicate right");

	// Reflect (with edge): dcba|abcd|dcba -> at -1 we get b=20
	val = fetch_pixel(img, 0, -1, BorderMode::reflect);
	if (!(near(val, 20.0))) throw std::runtime_error("test failed: border reflect");

	// Reflect_101 (without edge): dcb|abcd|cba -> at -1 we get b=20
	val = fetch_pixel(img, 0, -1, BorderMode::reflect_101);
	if (!(near(val, 20.0))) throw std::runtime_error("test failed: border reflect_101");

	// Wrap
	val = fetch_pixel(img, 0, -1, BorderMode::wrap);
	if (!(near(val, 40.0))) throw std::runtime_error("test failed: border wrap left");
	val = fetch_pixel(img, 0, 4, BorderMode::wrap);
	if (!(near(val, 10.0))) throw std::runtime_error("test failed: border wrap right");

	// Constant with pad value
	val = fetch_pixel(img, 0, -1, BorderMode::constant, 99.0);
	if (!(near(val, 99.0))) throw std::runtime_error("test failed: border constant");

	std::cout << "  border_modes: passed\n";
}

// ========== Convolution Tests ==========

void test_convolve2d_identity() {
	auto img = make_horizontal_ramp(5, 5);
	auto kernel = make_identity_kernel();
	auto result = convolve2d(img, kernel);

	// Identity kernel should reproduce the input
	for (std::size_t r = 0; r < 5; ++r)
		for (std::size_t c = 0; c < 5; ++c)
			if (!(near(result(r, c), img(r, c))))
				throw std::runtime_error("test failed: convolve2d identity");

	std::cout << "  convolve2d_identity: passed\n";
}

void test_convolve2d_box_blur() {
	// 5x5 constant image with value 7 — box blur should leave it unchanged
	auto img = make_constant(5, 5, 7.0);
	mtl::mat::dense2D<double> box(3, 3);
	for (std::size_t r = 0; r < 3; ++r)
		for (std::size_t c = 0; c < 3; ++c)
			box(r, c) = 1.0 / 9.0;

	auto result = convolve2d(img, box);
	for (std::size_t r = 0; r < 5; ++r)
		for (std::size_t c = 0; c < 5; ++c)
			if (!(near(result(r, c), 7.0, 1e-6)))
				throw std::runtime_error("test failed: box blur on constant");

	std::cout << "  convolve2d_box_blur: passed\n";
}

void test_convolve2d_multichannel() {
	RGBImage<double> rgb(5, 5);
	for (std::size_t r = 0; r < 5; ++r)
		for (std::size_t c = 0; c < 5; ++c) {
			rgb[0](r, c) = 1.0;
			rgb[1](r, c) = 2.0;
			rgb[2](r, c) = 3.0;
		}

	auto kernel = make_identity_kernel();
	auto result = convolve2d(rgb, kernel);

	if (!(near(result[0](2, 2), 1.0))) throw std::runtime_error("test failed: multichannel conv R");
	if (!(near(result[1](2, 2), 2.0))) throw std::runtime_error("test failed: multichannel conv G");
	if (!(near(result[2](2, 2), 3.0))) throw std::runtime_error("test failed: multichannel conv B");

	std::cout << "  convolve2d_multichannel: passed\n";
}

// ========== Separable Filter Tests ==========

void test_separable_vs_2d() {
	// Separable box blur should match 2D box blur
	auto img = make_horizontal_ramp(8, 8);

	// 2D box kernel
	mtl::mat::dense2D<double> box2d(3, 3);
	for (std::size_t r = 0; r < 3; ++r)
		for (std::size_t c = 0; c < 3; ++c)
			box2d(r, c) = 1.0 / 9.0;

	// Separable box kernel: [1/3, 1/3, 1/3] x [1/3, 1/3, 1/3]
	auto box1d = box_kernel_1d<double>(3);

	auto result_2d = convolve2d(img, box2d);
	auto result_sep = separable_filter(img, box1d, box1d);

	for (std::size_t r = 0; r < 8; ++r)
		for (std::size_t c = 0; c < 8; ++c)
			if (!(near(result_2d(r, c), result_sep(r, c), 1e-10)))
				throw std::runtime_error("test failed: separable vs 2D mismatch");

	std::cout << "  separable_vs_2d: passed\n";
}

void test_gaussian_blur() {
	// Gaussian blur of a constant image should be the same constant
	auto img = make_constant(10, 10, 5.0);
	auto blurred = gaussian_blur(img, 1.0);

	for (std::size_t r = 1; r + 1 < 10; ++r)
		for (std::size_t c = 1; c + 1 < 10; ++c)
			if (!(near(blurred(r, c), 5.0, 1e-3)))
				throw std::runtime_error("test failed: gaussian blur constant");

	std::cout << "  gaussian_blur: passed\n";
}

void test_box_blur() {
	auto img = make_constant(8, 8, 3.0);
	auto blurred = box_blur(img, 3);
	// Should remain 3.0
	for (std::size_t r = 0; r < 8; ++r)
		for (std::size_t c = 0; c < 8; ++c)
			if (!(near(blurred(r, c), 3.0, 1e-6)))
				throw std::runtime_error("test failed: box blur constant");

	std::cout << "  box_blur: passed\n";
}

// ========== Morphology Tests ==========

void test_structuring_elements() {
	auto rect = make_rect_element(3, 3);
	for (std::size_t r = 0; r < 3; ++r)
		for (std::size_t c = 0; c < 3; ++c)
			if (!rect(r, c))
				throw std::runtime_error("test failed: rect element all true");

	auto cross = make_cross_element(3);
	if (!cross(1, 0)) throw std::runtime_error("test failed: cross center row");
	if (!cross(0, 1)) throw std::runtime_error("test failed: cross center col");
	if (cross(0, 0)) throw std::runtime_error("test failed: cross corner should be false");

	auto ellipse = make_ellipse_element(5);
	// Center should be true
	if (!ellipse(2, 2)) throw std::runtime_error("test failed: ellipse center");
	// Corners should be false for a 5x5 ellipse
	if (ellipse(0, 0)) throw std::runtime_error("test failed: ellipse corner");

	std::cout << "  structuring_elements: passed\n";
}

void test_dilate_erode() {
	// Single bright pixel in a dark field
	auto img = make_constant(7, 7, 0.0);
	img(3, 3) = 1.0;

	auto elem = make_rect_element(3, 3);

	// Dilation should spread the bright pixel to its 3x3 neighborhood
	auto dilated = dilate(img, elem);
	if (!(near(dilated(3, 3), 1.0))) throw std::runtime_error("test failed: dilate center");
	if (!(near(dilated(2, 2), 1.0))) throw std::runtime_error("test failed: dilate neighbor");
	if (!(near(dilated(1, 1), 0.0))) throw std::runtime_error("test failed: dilate far");

	// Erosion of the dilated result should shrink it back
	auto eroded = erode(dilated, elem);
	if (!(near(eroded(3, 3), 1.0))) throw std::runtime_error("test failed: erode center");
	// One pixel away from center should be 0 after erode(dilate)
	if (!(near(eroded(1, 1), 0.0))) throw std::runtime_error("test failed: erode far");

	std::cout << "  dilate_erode: passed\n";
}

void test_morphological_open_idempotent() {
	// Opening is idempotent: open(open(x)) == open(x)
	auto img = make_step_edge(8, 8);
	auto elem = make_rect_element(3, 3);

	auto opened = morphological_open(img, elem);
	auto opened2 = morphological_open(opened, elem);

	for (std::size_t r = 0; r < 8; ++r)
		for (std::size_t c = 0; c < 8; ++c)
			if (!(near(opened(r, c), opened2(r, c), 1e-10)))
				throw std::runtime_error("test failed: open idempotency");

	std::cout << "  morphological_open_idempotent: passed\n";
}

void test_morphological_close_idempotent() {
	auto img = make_step_edge(8, 8);
	auto elem = make_rect_element(3, 3);

	auto closed = morphological_close(img, elem);
	auto closed2 = morphological_close(closed, elem);

	for (std::size_t r = 0; r < 8; ++r)
		for (std::size_t c = 0; c < 8; ++c)
			if (!(near(closed(r, c), closed2(r, c), 1e-10)))
				throw std::runtime_error("test failed: close idempotency");

	std::cout << "  morphological_close_idempotent: passed\n";
}

void test_morphological_gradient() {
	// Gradient of a constant image should be zero
	auto img = make_constant(8, 8, 5.0);
	auto elem = make_rect_element(3, 3);
	auto grad = morphological_gradient(img, elem);

	for (std::size_t r = 0; r < 8; ++r)
		for (std::size_t c = 0; c < 8; ++c)
			if (!(near(grad(r, c), 0.0, 1e-10)))
				throw std::runtime_error("test failed: gradient of constant");

	std::cout << "  morphological_gradient: passed\n";
}

void test_tophat_blackhat() {
	auto img = make_constant(8, 8, 5.0);
	auto elem = make_rect_element(3, 3);

	// Top-hat and black-hat of a constant image should both be zero
	auto th = tophat(img, elem);
	auto bh = blackhat(img, elem);

	for (std::size_t r = 0; r < 8; ++r) {
		for (std::size_t c = 0; c < 8; ++c) {
			if (!(near(th(r, c), 0.0, 1e-10)))
				throw std::runtime_error("test failed: tophat constant");
			if (!(near(bh(r, c), 0.0, 1e-10)))
				throw std::runtime_error("test failed: blackhat constant");
		}
	}

	std::cout << "  tophat_blackhat: passed\n";
}

// ========== Edge Detection Tests ==========

void test_sobel_constant() {
	// Sobel of a constant image should be zero everywhere
	auto img = make_constant(8, 8, 5.0);
	auto gx = sobel_x(img);
	auto gy = sobel_y(img);

	for (std::size_t r = 1; r + 1 < 8; ++r) {
		for (std::size_t c = 1; c + 1 < 8; ++c) {
			if (!(near(gx(r, c), 0.0, 1e-6)))
				throw std::runtime_error("test failed: sobel_x constant");
			if (!(near(gy(r, c), 0.0, 1e-6)))
				throw std::runtime_error("test failed: sobel_y constant");
		}
	}

	std::cout << "  sobel_constant: passed\n";
}

void test_sobel_horizontal_ramp() {
	// Horizontal ramp: pixel = column index
	// Sobel X should detect the constant horizontal gradient
	// Sobel Y should be zero (no vertical variation)
	auto img = make_horizontal_ramp(10, 10);
	auto gx = sobel_x(img);
	auto gy = sobel_y(img);

	// Interior pixels should have constant gx (non-zero) and zero gy
	// The Sobel X kernel on a unit-slope ramp gives:
	// For f(r,c) = c: Sobel_X = [1,2,1]^T * [-1,0,1] applied to c
	// = (c+1 - (c-1)) * (1+2+1) = 2*4 = 8
	for (std::size_t r = 2; r + 2 < 10; ++r) {
		for (std::size_t c = 2; c + 2 < 10; ++c) {
			if (!(std::abs(gx(r, c)) > 0.1))
				throw std::runtime_error("test failed: sobel_x ramp should be non-zero");
			if (!(near(gy(r, c), 0.0, 1e-6)))
				throw std::runtime_error("test failed: sobel_y of h-ramp should be zero");
		}
	}

	// All interior gx values should be the same (constant gradient)
	double ref = gx(4, 4);
	for (std::size_t r = 2; r + 2 < 10; ++r) {
		for (std::size_t c = 2; c + 2 < 10; ++c) {
			if (!(near(gx(r, c), ref, 1e-6)))
				throw std::runtime_error("test failed: sobel_x ramp not constant");
		}
	}

	std::cout << "  sobel_horizontal_ramp: passed (gx=" << ref << ")\n";
}

void test_prewitt() {
	auto img = make_horizontal_ramp(8, 8);
	auto gx = prewitt_x(img);
	auto gy = prewitt_y(img);

	// Same logic as Sobel: constant horizontal gradient, zero vertical
	for (std::size_t r = 2; r + 2 < 8; ++r) {
		for (std::size_t c = 2; c + 2 < 8; ++c) {
			if (!(std::abs(gx(r, c)) > 0.1))
				throw std::runtime_error("test failed: prewitt_x ramp non-zero");
			if (!(near(gy(r, c), 0.0, 1e-6)))
				throw std::runtime_error("test failed: prewitt_y of h-ramp zero");
		}
	}

	std::cout << "  prewitt: passed\n";
}

void test_gradient_magnitude() {
	auto img = make_horizontal_ramp(8, 8);
	auto gx = sobel_x(img);
	auto gy = sobel_y(img);
	auto mag = gradient_magnitude(gx, gy);

	// Since gy ≈ 0 for horizontal ramp, magnitude ≈ |gx|
	for (std::size_t r = 2; r + 2 < 8; ++r) {
		for (std::size_t c = 2; c + 2 < 8; ++c) {
			double expected = std::abs(gx(r, c));
			if (!(near(mag(r, c), expected, 1e-6)))
				throw std::runtime_error("test failed: gradient magnitude");
		}
	}

	std::cout << "  gradient_magnitude: passed\n";
}

void test_canny_step_edge() {
	// Canny on a step edge should produce edges near the transition
	auto img = make_step_edge(20, 20);
	auto edges = canny(img, 0.1, 0.3, 1.0);

	// Check that edge pixels exist near column 10 (the step)
	bool found_edge = false;
	for (std::size_t r = 3; r + 3 < 20; ++r) {
		for (std::size_t c = 8; c <= 12; ++c) {
			if (static_cast<double>(edges(r, c)) > 0.9) {
				found_edge = true;
				break;
			}
		}
		if (found_edge) break;
	}
	if (!found_edge)
		throw std::runtime_error("test failed: canny should detect step edge");

	// Check that far-from-edge pixels are not marked
	bool clean_left = true;
	for (std::size_t r = 3; r + 3 < 20; ++r) {
		for (std::size_t c = 0; c < 5; ++c) {
			if (static_cast<double>(edges(r, c)) > 0.9) {
				clean_left = false;
				break;
			}
		}
	}
	if (!clean_left)
		throw std::runtime_error("test failed: canny should not mark flat regions");

	std::cout << "  canny_step_edge: passed\n";
}

// ========== Mixed Precision Test ==========

void test_mixed_precision_convolution() {
	// float image convolved with double kernel
	mtl::mat::dense2D<float> img(5, 5);
	for (std::size_t r = 0; r < 5; ++r)
		for (std::size_t c = 0; c < 5; ++c)
			img(r, c) = static_cast<float>(c);

	mtl::mat::dense2D<double> kernel(3, 3);
	for (std::size_t r = 0; r < 3; ++r)
		for (std::size_t c = 0; c < 3; ++c)
			kernel(r, c) = 0.0;
	kernel(1, 1) = 1.0;

	auto result = convolve2d(img, kernel);
	// Identity kernel should reproduce input
	if (!(near(static_cast<double>(result(2, 3)), 3.0, 1e-5)))
		throw std::runtime_error("test failed: mixed precision convolution");

	std::cout << "  mixed_precision_convolution: passed\n";
}

int main() {
	try {
		std::cout << "Image Processing Tests\n";

		// Container and utilities
		test_image_construction();
		test_apply_per_channel();
		test_rgb_to_gray();

		// Border handling
		test_border_modes();

		// Convolution
		test_convolve2d_identity();
		test_convolve2d_box_blur();
		test_convolve2d_multichannel();

		// Separable filters
		test_separable_vs_2d();
		test_gaussian_blur();
		test_box_blur();

		// Morphology
		test_structuring_elements();
		test_dilate_erode();
		test_morphological_open_idempotent();
		test_morphological_close_idempotent();
		test_morphological_gradient();
		test_tophat_blackhat();

		// Edge detection
		test_sobel_constant();
		test_sobel_horizontal_ramp();
		test_prewitt();
		test_gradient_magnitude();
		test_canny_step_edge();

		// Mixed precision
		test_mixed_precision_convolution();

		std::cout << "All image processing tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << '\n';
		return 1;
	}
}
