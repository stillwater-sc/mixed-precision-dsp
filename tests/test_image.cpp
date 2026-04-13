// test_image.cpp: test image processing — convolution, separable, morphology, edge
//
// Tests use float as the primary pixel type (realistic for image data).
// Algorithmic precision tests (Canny, Gaussian normalization) use double.
// Mixed-precision tests validate float pixels with double kernels.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/image/image.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

using namespace sw::dsp;

bool near(double a, double b, double eps = 1e-4) {
	return std::abs(a - b) < eps;
}

// ---------- Templated helpers for synthetic test images ----------

template <typename T>
mtl::mat::dense2D<T> make_constant(std::size_t rows, std::size_t cols, T val) {
	mtl::mat::dense2D<T> img(rows, cols);
	for (std::size_t r = 0; r < rows; ++r)
		for (std::size_t c = 0; c < cols; ++c)
			img(r, c) = val;
	return img;
}

template <typename T>
mtl::mat::dense2D<T> make_horizontal_ramp(std::size_t rows, std::size_t cols) {
	mtl::mat::dense2D<T> img(rows, cols);
	for (std::size_t r = 0; r < rows; ++r)
		for (std::size_t c = 0; c < cols; ++c)
			img(r, c) = static_cast<T>(c);
	return img;
}

template <typename T>
mtl::mat::dense2D<T> make_step_edge(std::size_t rows, std::size_t cols) {
	mtl::mat::dense2D<T> img(rows, cols);
	std::size_t mid = cols / 2;
	for (std::size_t r = 0; r < rows; ++r)
		for (std::size_t c = 0; c < cols; ++c)
			img(r, c) = (c >= mid) ? static_cast<T>(1) : static_cast<T>(0);
	return img;
}

// ========== Image Container Tests ==========

void test_image_construction() {
	// float — the typical image pixel type
	Image<float, 3> rgb(10, 20);
	if (!(rgb.rows() == 10)) throw std::runtime_error("test failed: Image rows");
	if (!(rgb.cols() == 20)) throw std::runtime_error("test failed: Image cols");
	if (!(rgb.channels() == 3)) throw std::runtime_error("test failed: Image channels");

	GrayImage<float> gray;
	RGBImage<float> rgb_f(5, 5);
	RGBAImage<float> rgba(8, 8);
	if (!(rgba.channels() == 4)) throw std::runtime_error("test failed: RGBA channels");

	// Verify container compiles with double (for precision-sensitive algorithms)
	Image<double, 1> gray_d(4, 4);
	(void)gray_d;

	std::cout << "  image_construction: passed\n";
}

void test_apply_per_channel() {
	RGBImage<float> rgb(4, 4);
	for (std::size_t r = 0; r < 4; ++r) {
		for (std::size_t c = 0; c < 4; ++c) {
			rgb[0](r, c) = 1.0f;
			rgb[1](r, c) = 2.0f;
			rgb[2](r, c) = 3.0f;
		}
	}

	auto result = apply_per_channel(rgb, [](const mtl::mat::dense2D<float>& plane) {
		mtl::mat::dense2D<float> out(plane.num_rows(), plane.num_cols());
		for (std::size_t r = 0; r < plane.num_rows(); ++r)
			for (std::size_t c = 0; c < plane.num_cols(); ++c)
				out(r, c) = plane(r, c) * 2.0f;
		return out;
	});

	if (!(near(result[0](0, 0), 2.0))) throw std::runtime_error("test failed: apply_per_channel R");
	if (!(near(result[1](0, 0), 4.0))) throw std::runtime_error("test failed: apply_per_channel G");
	if (!(near(result[2](0, 0), 6.0))) throw std::runtime_error("test failed: apply_per_channel B");

	std::cout << "  apply_per_channel: passed\n";
}

void test_rgb_to_gray() {
	RGBImage<float> rgb(2, 2);
	// White pixel: R=G=B=1.0
	for (std::size_t r = 0; r < 2; ++r) {
		for (std::size_t c = 0; c < 2; ++c) {
			rgb[0](r, c) = 1.0f;
			rgb[1](r, c) = 1.0f;
			rgb[2](r, c) = 1.0f;
		}
	}
	auto gray = rgb_to_gray(rgb);
	// 0.299 + 0.587 + 0.114 = 1.0
	if (!(near(gray(0, 0), 1.0, 1e-3)))
		throw std::runtime_error("test failed: rgb_to_gray white");

	// Pure red
	for (std::size_t r = 0; r < 2; ++r) {
		for (std::size_t c = 0; c < 2; ++c) {
			rgb[0](r, c) = 1.0f;
			rgb[1](r, c) = 0.0f;
			rgb[2](r, c) = 0.0f;
		}
	}
	gray = rgb_to_gray(rgb);
	if (!(near(gray(0, 0), 0.299, 1e-3)))
		throw std::runtime_error("test failed: rgb_to_gray red");

	std::cout << "  rgb_to_gray: passed\n";
}

// ========== Border Handling Tests ==========

void test_border_modes() {
	// Test with float — the type users will actually use
	mtl::mat::dense2D<float> img(1, 4);
	img(0, 0) = 10.0f; img(0, 1) = 20.0f; img(0, 2) = 30.0f; img(0, 3) = 40.0f;

	// Zero padding
	float val = fetch_pixel(img, 0, -1, BorderMode::zero);
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
	val = fetch_pixel(img, 0, -1, BorderMode::constant, 99.0f);
	if (!(near(val, 99.0))) throw std::runtime_error("test failed: border constant");

	std::cout << "  border_modes: passed\n";
}

// ========== Convolution Tests ==========

void test_convolve2d_identity() {
	// float image, double kernel — the typical mixed-precision path
	auto img = make_horizontal_ramp<float>(5, 5);
	mtl::mat::dense2D<double> kernel(3, 3);
	for (std::size_t r = 0; r < 3; ++r)
		for (std::size_t c = 0; c < 3; ++c)
			kernel(r, c) = 0.0;
	kernel(1, 1) = 1.0;

	auto result = convolve2d(img, kernel);

	for (std::size_t r = 0; r < 5; ++r)
		for (std::size_t c = 0; c < 5; ++c)
			if (!(near(result(r, c), img(r, c))))
				throw std::runtime_error("test failed: convolve2d identity");

	std::cout << "  convolve2d_identity: passed\n";
}

void test_convolve2d_box_blur() {
	// float constant image, double box kernel
	auto img = make_constant<float>(5, 5, 7.0f);
	mtl::mat::dense2D<double> box(3, 3);
	for (std::size_t r = 0; r < 3; ++r)
		for (std::size_t c = 0; c < 3; ++c)
			box(r, c) = 1.0 / 9.0;

	auto result = convolve2d(img, box);
	for (std::size_t r = 0; r < 5; ++r)
		for (std::size_t c = 0; c < 5; ++c)
			if (!(near(result(r, c), 7.0, 1e-5)))
				throw std::runtime_error("test failed: box blur on constant");

	std::cout << "  convolve2d_box_blur: passed\n";
}

void test_convolve2d_multichannel() {
	RGBImage<float> rgb(5, 5);
	for (std::size_t r = 0; r < 5; ++r)
		for (std::size_t c = 0; c < 5; ++c) {
			rgb[0](r, c) = 1.0f;
			rgb[1](r, c) = 2.0f;
			rgb[2](r, c) = 3.0f;
		}

	mtl::mat::dense2D<double> kernel(3, 3);
	for (std::size_t r = 0; r < 3; ++r)
		for (std::size_t c = 0; c < 3; ++c)
			kernel(r, c) = 0.0;
	kernel(1, 1) = 1.0;

	auto result = convolve2d(rgb, kernel);

	if (!(near(result[0](2, 2), 1.0))) throw std::runtime_error("test failed: multichannel conv R");
	if (!(near(result[1](2, 2), 2.0))) throw std::runtime_error("test failed: multichannel conv G");
	if (!(near(result[2](2, 2), 3.0))) throw std::runtime_error("test failed: multichannel conv B");

	std::cout << "  convolve2d_multichannel: passed\n";
}

// ========== Separable Filter Tests ==========

void test_separable_vs_2d() {
	// Both float image — verify separable matches non-separable
	auto img = make_horizontal_ramp<float>(8, 8);

	// 2D box kernel (double precision coefficients)
	mtl::mat::dense2D<double> box2d(3, 3);
	for (std::size_t r = 0; r < 3; ++r)
		for (std::size_t c = 0; c < 3; ++c)
			box2d(r, c) = 1.0 / 9.0;

	// Separable box kernel (double precision)
	auto box1d = box_kernel_1d<double>(3);

	auto result_2d = convolve2d(img, box2d);
	auto result_sep = separable_filter(img, box1d, box1d);

	for (std::size_t r = 0; r < 8; ++r)
		for (std::size_t c = 0; c < 8; ++c)
			if (!(near(result_2d(r, c), result_sep(r, c), 1e-5)))
				throw std::runtime_error("test failed: separable vs 2D mismatch");

	std::cout << "  separable_vs_2d: passed\n";
}

void test_gaussian_blur() {
	// Gaussian blur of a constant float image should preserve the value
	auto img = make_constant<float>(10, 10, 5.0f);
	auto blurred = gaussian_blur(img, 1.0);

	for (std::size_t r = 1; r + 1 < 10; ++r)
		for (std::size_t c = 1; c + 1 < 10; ++c)
			if (!(near(blurred(r, c), 5.0, 1e-3)))
				throw std::runtime_error("test failed: gaussian blur constant");

	std::cout << "  gaussian_blur: passed\n";
}

void test_box_blur() {
	auto img = make_constant<float>(8, 8, 3.0f);
	auto blurred = box_blur(img, 3);
	for (std::size_t r = 0; r < 8; ++r)
		for (std::size_t c = 0; c < 8; ++c)
			if (!(near(blurred(r, c), 3.0, 1e-5)))
				throw std::runtime_error("test failed: box blur constant");

	std::cout << "  box_blur: passed\n";
}

void test_blur_validation() {
	auto img = make_constant<float>(4, 4, 1.0f);

	bool caught = false;
	try { gaussian_blur(img, 0.0); } catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: gaussian_blur should reject sigma=0");

	caught = false;
	try { box_blur(img, 0); } catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: box_blur should reject size=0");

	std::cout << "  blur_validation: passed\n";
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
	if (!ellipse(2, 2)) throw std::runtime_error("test failed: ellipse center");
	if (ellipse(0, 0)) throw std::runtime_error("test failed: ellipse corner");

	std::cout << "  structuring_elements: passed\n";
}

void test_dilate_erode() {
	// Morphology on float — the typical use case
	auto img = make_constant<float>(7, 7, 0.0f);
	img(3, 3) = 1.0f;

	auto elem = make_rect_element(3, 3);

	auto dilated = dilate(img, elem);
	if (!(near(dilated(3, 3), 1.0))) throw std::runtime_error("test failed: dilate center");
	if (!(near(dilated(2, 2), 1.0))) throw std::runtime_error("test failed: dilate neighbor");
	if (!(near(dilated(1, 1), 0.0))) throw std::runtime_error("test failed: dilate far");

	auto eroded = erode(dilated, elem);
	if (!(near(eroded(3, 3), 1.0))) throw std::runtime_error("test failed: erode center");
	if (!(near(eroded(1, 1), 0.0))) throw std::runtime_error("test failed: erode far");

	std::cout << "  dilate_erode: passed\n";
}

void test_morphology_element_validation() {
	auto img = make_constant<float>(4, 4, 1.0f);

	// Empty element should throw
	mtl::mat::dense2D<bool> empty(0, 0);
	bool caught = false;
	try { dilate(img, empty); } catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: dilate should reject empty element");

	// Element with no active pixels should throw
	mtl::mat::dense2D<bool> all_false(3, 3);
	for (std::size_t r = 0; r < 3; ++r)
		for (std::size_t c = 0; c < 3; ++c)
			all_false(r, c) = false;
	caught = false;
	try { erode(img, all_false); } catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: erode should reject all-false element");

	std::cout << "  morphology_element_validation: passed\n";
}

void test_morphological_open_idempotent() {
	auto img = make_step_edge<float>(8, 8);
	auto elem = make_rect_element(3, 3);

	auto opened = morphological_open(img, elem);
	auto opened2 = morphological_open(opened, elem);

	for (std::size_t r = 0; r < 8; ++r)
		for (std::size_t c = 0; c < 8; ++c)
			if (!(near(opened(r, c), opened2(r, c), 1e-6)))
				throw std::runtime_error("test failed: open idempotency");

	std::cout << "  morphological_open_idempotent: passed\n";
}

void test_morphological_close_idempotent() {
	auto img = make_step_edge<float>(8, 8);
	auto elem = make_rect_element(3, 3);

	auto closed = morphological_close(img, elem);
	auto closed2 = morphological_close(closed, elem);

	for (std::size_t r = 0; r < 8; ++r)
		for (std::size_t c = 0; c < 8; ++c)
			if (!(near(closed(r, c), closed2(r, c), 1e-6)))
				throw std::runtime_error("test failed: close idempotency");

	std::cout << "  morphological_close_idempotent: passed\n";
}

void test_morphological_gradient() {
	auto img = make_constant<float>(8, 8, 5.0f);
	auto elem = make_rect_element(3, 3);
	auto grad = morphological_gradient(img, elem);

	for (std::size_t r = 0; r < 8; ++r)
		for (std::size_t c = 0; c < 8; ++c)
			if (!(near(grad(r, c), 0.0, 1e-6)))
				throw std::runtime_error("test failed: gradient of constant");

	std::cout << "  morphological_gradient: passed\n";
}

void test_tophat_blackhat() {
	auto img = make_constant<float>(8, 8, 5.0f);
	auto elem = make_rect_element(3, 3);

	auto th = tophat(img, elem);
	auto bh = blackhat(img, elem);

	for (std::size_t r = 0; r < 8; ++r) {
		for (std::size_t c = 0; c < 8; ++c) {
			if (!(near(th(r, c), 0.0, 1e-6)))
				throw std::runtime_error("test failed: tophat constant");
			if (!(near(bh(r, c), 0.0, 1e-6)))
				throw std::runtime_error("test failed: blackhat constant");
		}
	}

	std::cout << "  tophat_blackhat: passed\n";
}

// ========== Edge Detection Tests ==========

void test_sobel_constant() {
	auto img = make_constant<float>(8, 8, 5.0f);
	auto gx = sobel_x(img);
	auto gy = sobel_y(img);

	for (std::size_t r = 1; r + 1 < 8; ++r) {
		for (std::size_t c = 1; c + 1 < 8; ++c) {
			if (!(near(gx(r, c), 0.0, 1e-4)))
				throw std::runtime_error("test failed: sobel_x constant");
			if (!(near(gy(r, c), 0.0, 1e-4)))
				throw std::runtime_error("test failed: sobel_y constant");
		}
	}

	std::cout << "  sobel_constant: passed\n";
}

void test_sobel_horizontal_ramp() {
	auto img = make_horizontal_ramp<float>(10, 10);
	auto gx = sobel_x(img);
	auto gy = sobel_y(img);

	// Interior pixels: constant gx, zero gy
	for (std::size_t r = 2; r + 2 < 10; ++r) {
		for (std::size_t c = 2; c + 2 < 10; ++c) {
			if (!(std::abs(static_cast<double>(gx(r, c))) > 0.1))
				throw std::runtime_error("test failed: sobel_x ramp should be non-zero");
			if (!(near(gy(r, c), 0.0, 1e-4)))
				throw std::runtime_error("test failed: sobel_y of h-ramp should be zero");
		}
	}

	// All interior gx values should be the same (constant gradient)
	float ref = gx(4, 4);
	for (std::size_t r = 2; r + 2 < 10; ++r) {
		for (std::size_t c = 2; c + 2 < 10; ++c) {
			if (!(near(gx(r, c), ref, 1e-4)))
				throw std::runtime_error("test failed: sobel_x ramp not constant");
		}
	}

	std::cout << "  sobel_horizontal_ramp: passed (gx=" << ref << ")\n";
}

void test_prewitt() {
	auto img = make_horizontal_ramp<float>(8, 8);
	auto gx = prewitt_x(img);
	auto gy = prewitt_y(img);

	for (std::size_t r = 2; r + 2 < 8; ++r) {
		for (std::size_t c = 2; c + 2 < 8; ++c) {
			if (!(std::abs(static_cast<double>(gx(r, c))) > 0.1))
				throw std::runtime_error("test failed: prewitt_x ramp non-zero");
			if (!(near(gy(r, c), 0.0, 1e-4)))
				throw std::runtime_error("test failed: prewitt_y of h-ramp zero");
		}
	}

	std::cout << "  prewitt: passed\n";
}

void test_gradient_magnitude() {
	auto img = make_horizontal_ramp<float>(8, 8);
	auto gx = sobel_x(img);
	auto gy = sobel_y(img);
	auto mag = gradient_magnitude(gx, gy);

	for (std::size_t r = 2; r + 2 < 8; ++r) {
		for (std::size_t c = 2; c + 2 < 8; ++c) {
			double expected = std::abs(static_cast<double>(gx(r, c)));
			if (!(near(mag(r, c), expected, 1e-4)))
				throw std::runtime_error("test failed: gradient magnitude");
		}
	}

	std::cout << "  gradient_magnitude: passed\n";
}

void test_gradient_magnitude_validation() {
	mtl::mat::dense2D<float> a(4, 4), b(3, 3);
	bool caught = false;
	try { gradient_magnitude(a, b); } catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: gradient_magnitude should reject dim mismatch");

	std::cout << "  gradient_magnitude_validation: passed\n";
}

void test_canny_step_edge() {
	// Canny requires double for threshold math — test with double pixels
	auto img = make_step_edge<double>(20, 20);
	auto edges = canny(img, 0.1, 0.3, 1.0);

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

// ========== Mixed Precision Tests ==========

void test_mixed_precision_convolution() {
	// float image + double kernel: the primary mixed-precision use case.
	// The accumulator should use double precision, avoiding float rounding
	// in the inner product.
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
	if (!(near(static_cast<double>(result(2, 3)), 3.0, 1e-5)))
		throw std::runtime_error("test failed: float+double convolution");

	std::cout << "  mixed_precision_convolution: passed\n";
}

void test_mixed_precision_separable() {
	// float image + double kernel via separable filter
	auto img = make_constant<float>(6, 6, 4.0f);
	auto kernel = box_kernel_1d<double>(3);  // double kernel coefficients

	auto result = separable_filter(img, kernel, kernel);
	// Constant image should be unchanged
	for (std::size_t r = 0; r < 6; ++r)
		for (std::size_t c = 0; c < 6; ++c)
			if (!(near(result(r, c), 4.0, 1e-5)))
				throw std::runtime_error("test failed: float+double separable");

	std::cout << "  mixed_precision_separable: passed\n";
}

void test_mixed_precision_gaussian() {
	// float image with double-precision Gaussian kernel
	// (gaussian_blur already uses double kernels internally)
	auto img = make_constant<float>(8, 8, 2.5f);
	auto blurred = gaussian_blur(img, 1.0);

	for (std::size_t r = 1; r + 1 < 8; ++r)
		for (std::size_t c = 1; c + 1 < 8; ++c)
			if (!(near(blurred(r, c), 2.5, 1e-3)))
				throw std::runtime_error("test failed: float gaussian blur");

	std::cout << "  mixed_precision_gaussian: passed\n";
}

int main() {
	try {
		std::cout << "Image Processing Tests\n";

		// Container and utilities (float)
		test_image_construction();
		test_apply_per_channel();
		test_rgb_to_gray();

		// Border handling (float)
		test_border_modes();

		// Convolution (float image + double kernel)
		test_convolve2d_identity();
		test_convolve2d_box_blur();
		test_convolve2d_multichannel();

		// Separable filters (float image + double kernel)
		test_separable_vs_2d();
		test_gaussian_blur();
		test_box_blur();
		test_blur_validation();

		// Morphology (float)
		test_structuring_elements();
		test_dilate_erode();
		test_morphology_element_validation();
		test_morphological_open_idempotent();
		test_morphological_close_idempotent();
		test_morphological_gradient();
		test_tophat_blackhat();

		// Edge detection (float + double for Canny)
		test_sobel_constant();
		test_sobel_horizontal_ramp();
		test_prewitt();
		test_gradient_magnitude();
		test_gradient_magnitude_validation();
		test_canny_step_edge();

		// Mixed precision (float pixel + double kernel)
		test_mixed_precision_convolution();
		test_mixed_precision_separable();
		test_mixed_precision_gaussian();

		std::cout << "All image processing tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << '\n';
		return 1;
	}
}
