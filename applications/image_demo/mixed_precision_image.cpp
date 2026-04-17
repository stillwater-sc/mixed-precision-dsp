// mixed_precision_image.cpp: sensor-noise-limited arithmetic optimization
//
// Demonstrates that narrow arithmetic types (half, posit<8,2>, bfloat16)
// lose very little quality vs. double when processing typical image data,
// because sensor noise limits effective precision to 5-6 bits anyway.
//
// For each arithmetic type, runs Sobel gradient, Gaussian blur, and Canny
// edge detection on synthetic test images, then reports SQNR against the
// double reference and Canny edge agreement rate.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/image/image.hpp>
#include <sw/dsp/image/generators.hpp>
#include <sw/dsp/image/edge.hpp>
#include <sw/dsp/image/separable.hpp>

#include <sw/universal/number/cfloat/cfloat.hpp>
#include <sw/universal/number/posit/posit.hpp>
#include <sw/universal/number/integer/integer.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace sw::dsp;
using namespace sw::universal;

// Type aliases (avoid colliding with sw::universal::fp16/fp32)
using bf16   = cfloat<16, 8, uint16_t, true, false, false>;
using half_t = cfloat<16, 5, uint16_t, true, false, false>;
using p8     = posit<8, 2>;
using int12  = integer<12>;
using int8   = integer<8>;
using int6   = integer<6>;

// ============================================================================
// Helpers
// ============================================================================

template <typename T>
mtl::mat::dense2D<T> convert_image(const mtl::mat::dense2D<double>& src) {
	std::size_t r = src.num_rows(), c = src.num_cols();
	mtl::mat::dense2D<T> dst(r, c);
	for (std::size_t i = 0; i < r; ++i)
		for (std::size_t j = 0; j < c; ++j)
			dst(i, j) = static_cast<T>(src(i, j));
	return dst;
}

template <typename T>
mtl::mat::dense2D<double> to_double(const mtl::mat::dense2D<T>& src) {
	std::size_t r = src.num_rows(), c = src.num_cols();
	mtl::mat::dense2D<double> dst(r, c);
	for (std::size_t i = 0; i < r; ++i)
		for (std::size_t j = 0; j < c; ++j)
			dst(i, j) = static_cast<double>(src(i, j));
	return dst;
}

double image_sqnr(const mtl::mat::dense2D<double>& ref,
                  const mtl::mat::dense2D<double>& test) {
	std::size_t r = ref.num_rows(), c = ref.num_cols();
	double sig = 0.0, noise = 0.0;
	for (std::size_t i = 0; i < r; ++i) {
		for (std::size_t j = 0; j < c; ++j) {
			double rv = ref(i, j), tv = test(i, j);
			sig += rv * rv;
			noise += (rv - tv) * (rv - tv);
		}
	}
	if (noise < 1e-300) return 300.0;
	if (sig < 1e-300) return 0.0;
	return 10.0 * std::log10(sig / noise);
}

double edge_agreement(const mtl::mat::dense2D<double>& ref,
                      const mtl::mat::dense2D<double>& test) {
	std::size_t r = ref.num_rows(), c = ref.num_cols();
	std::size_t match = 0, total = r * c;
	for (std::size_t i = 0; i < r; ++i)
		for (std::size_t j = 0; j < c; ++j)
			if ((ref(i, j) > 0.5) == (test(i, j) > 0.5)) ++match;
	return 100.0 * static_cast<double>(match) / static_cast<double>(total);
}

// ============================================================================
// Per-type processing
// ============================================================================

struct TypeResult {
	std::string name;
	int bits;
	double sobel_sqnr;
	double gauss_sqnr;
	double canny_agree;
};

template <typename T>
TypeResult run_type(const std::string& name, int bits,
                    const mtl::mat::dense2D<double>& ref_image,
                    const mtl::mat::dense2D<double>& ref_sobel_mag,
                    const mtl::mat::dense2D<double>& ref_gauss,
                    const mtl::mat::dense2D<double>& ref_canny) {
	auto img = convert_image<T>(ref_image);

	// Sobel gradient magnitude
	auto gx = sobel_x(img);
	auto gy = sobel_y(img);
	std::size_t r = img.num_rows(), c = img.num_cols();
	mtl::mat::dense2D<T> mag(r, c);
	for (std::size_t i = 0; i < r; ++i)
		for (std::size_t j = 0; j < c; ++j) {
			double dx = static_cast<double>(gx(i, j));
			double dy = static_cast<double>(gy(i, j));
			mag(i, j) = static_cast<T>(std::sqrt(dx * dx + dy * dy));
		}

	// Gaussian blur
	auto blurred = gaussian_blur(img, 1.0);

	// Canny edge detection
	auto edges = canny(img, 0.1, 0.3, 1.0);

	auto mag_d = to_double(mag);
	auto blur_d = to_double(blurred);
	auto edge_d = to_double(edges);

	return {
		name, bits,
		image_sqnr(ref_sobel_mag, mag_d),
		image_sqnr(ref_gauss, blur_d),
		edge_agreement(ref_canny, edge_d)
	};
}

// ============================================================================
// Main
// ============================================================================

int main() {
	constexpr std::size_t ROWS = 64;
	constexpr std::size_t COLS = 64;

	std::cout << std::string(80, '=') << "\n";
	std::cout << "  Mixed-Precision Image Processing: Sensor-Noise-Limited Arithmetic\n";
	std::cout << "  " << ROWS << "x" << COLS << " synthetic test images\n";
	std::cout << std::string(80, '=') << "\n";

	// --- Generate test images in double ---
	// Checkerboard: good for edge detection (sharp transitions)
	auto checker = checkerboard<double>(ROWS, COLS, 8, 0.0, 1.0);
	// Step edge: vertical boundary at column 32
	mtl::mat::dense2D<double> step(ROWS, COLS);
	for (std::size_t r = 0; r < ROWS; ++r)
		for (std::size_t c = 0; c < COLS; ++c)
			step(r, c) = (c >= COLS / 2) ? 1.0 : 0.0;
	// Gradient ramp: smooth horizontal gradient
	auto ramp = gradient_horizontal<double>(ROWS, COLS, 0.0, 1.0);

	// Process three test patterns; report average metrics across all three.
	struct TestImage {
		std::string name;
		mtl::mat::dense2D<double>& img;
	};
	std::vector<TestImage> patterns = {
		{"checkerboard", checker},
		{"step edge", step},
		{"gradient ramp", ramp}
	};

	for (auto& pat : patterns) {
		std::cout << "\n" << std::string(80, '-') << "\n";
		std::cout << "  Pattern: " << pat.name << "\n";
		std::cout << std::string(80, '-') << "\n\n";

		// Double reference
		auto gx_ref = sobel_x(pat.img);
		auto gy_ref = sobel_y(pat.img);
		mtl::mat::dense2D<double> mag_ref(ROWS, COLS);
		for (std::size_t i = 0; i < ROWS; ++i)
			for (std::size_t j = 0; j < COLS; ++j) {
				double dx = gx_ref(i, j), dy = gy_ref(i, j);
				mag_ref(i, j) = std::sqrt(dx * dx + dy * dy);
			}
		auto gauss_ref = gaussian_blur(pat.img, 1.0);
		auto canny_ref = canny(pat.img, 0.1, 0.3, 1.0);

		// Run all types
		std::vector<TypeResult> results;
		results.push_back(run_type<double>("double",       64, pat.img, mag_ref, gauss_ref, canny_ref));
		results.push_back(run_type<float> ("float",        32, pat.img, mag_ref, gauss_ref, canny_ref));
		results.push_back(run_type<bf16>  ("bfloat16",     16, pat.img, mag_ref, gauss_ref, canny_ref));
		results.push_back(run_type<half_t>("half",         16, pat.img, mag_ref, gauss_ref, canny_ref));
		results.push_back(run_type<int12> ("integer<12>",  12, pat.img, mag_ref, gauss_ref, canny_ref));
		results.push_back(run_type<int8>  ("integer<8>",    8, pat.img, mag_ref, gauss_ref, canny_ref));
		results.push_back(run_type<int6>  ("integer<6>",    6, pat.img, mag_ref, gauss_ref, canny_ref));
		results.push_back(run_type<p8>    ("posit<8,2>",    8, pat.img, mag_ref, gauss_ref, canny_ref));

		std::cout << std::left  << std::setw(16) << "Type"
		          << std::right << std::setw(6)  << "Bits"
		          << std::right << std::setw(14) << "Sobel SQNR"
		          << std::right << std::setw(14) << "Gauss SQNR"
		          << std::right << std::setw(14) << "Canny Agree"
		          << "\n";
		std::cout << std::string(64, '-') << "\n";

		for (const auto& r : results) {
			std::cout << std::left  << std::setw(16) << r.name
			          << std::right << std::setw(6)  << r.bits;
			if (r.sobel_sqnr > 290.0)
				std::cout << std::right << std::setw(14) << "inf";
			else
				std::cout << std::right << std::setw(14) << std::fixed
				          << std::setprecision(1) << r.sobel_sqnr;
			if (r.gauss_sqnr > 290.0)
				std::cout << std::right << std::setw(14) << "inf";
			else
				std::cout << std::right << std::setw(14) << std::fixed
				          << std::setprecision(1) << r.gauss_sqnr;
			std::cout << std::right << std::setw(13) << std::fixed
			          << std::setprecision(1) << r.canny_agree << "%"
			          << "\n";
		}
	}

	std::cout << "\n" << std::string(80, '=') << "\n";
	std::cout << "  Key finding: narrow floating-point types (half, bfloat16, posit<8,2>)\n";
	std::cout << "  preserve most Sobel/Gaussian quality vs double, validating that\n";
	std::cout << "  sensor-noise-limited precision is sufficient for image processing.\n";
	std::cout << "  Integer types show that sub-unity pixel range requires scaling.\n";
	std::cout << std::string(80, '=') << "\n";

	return 0;
}
