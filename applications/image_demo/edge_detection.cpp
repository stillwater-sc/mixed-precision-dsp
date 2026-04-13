// edge_detection.cpp: edge detection demonstration
//
// Demonstrates Sobel and Canny edge detection on synthetic test images,
// showing the planar Image<T,C> container, border handling, and
// morphological operations.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/image/image.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

using namespace sw::dsp;

void print_separator(const std::string& title) {
	std::cout << "\n" << std::string(60, '=') << "\n";
	std::cout << "  " << title << "\n";
	std::cout << std::string(60, '=') << "\n\n";
}

// Print a dense2D matrix as a text-art visualization
template <typename T>
void print_image(const mtl::mat::dense2D<T>& img, const std::string& label,
                 int width = 6, int precision = 2) {
	std::cout << label << " (" << img.num_rows() << "x" << img.num_cols() << "):\n";
	for (std::size_t r = 0; r < img.num_rows(); ++r) {
		std::cout << "  ";
		for (std::size_t c = 0; c < img.num_cols(); ++c) {
			std::cout << std::setw(width) << std::fixed
			          << std::setprecision(precision)
			          << static_cast<double>(img(r, c));
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

// Print a binary edge map as ASCII art
template <typename T>
void print_edge_map(const mtl::mat::dense2D<T>& edges, const std::string& label) {
	std::cout << label << " (" << edges.num_rows() << "x" << edges.num_cols() << "):\n";
	for (std::size_t r = 0; r < edges.num_rows(); ++r) {
		std::cout << "  ";
		for (std::size_t c = 0; c < edges.num_cols(); ++c) {
			std::cout << (static_cast<double>(edges(r, c)) > 0.5 ? "#" : ".");
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

int main() {
	print_separator("Edge Detection Demo");
	std::cout << "Planar image processing with sw::dsp\n\n";

	// --- Step Edge ---
	print_separator("1. Step Edge: Sobel Gradient");

	std::size_t rows = 12, cols = 16;
	mtl::mat::dense2D<double> step(rows, cols);
	for (std::size_t r = 0; r < rows; ++r)
		for (std::size_t c = 0; c < cols; ++c)
			step(r, c) = (c >= cols / 2) ? 1.0 : 0.0;

	print_image(step, "Input: step edge");

	auto gx = sobel_x(step);
	auto gy = sobel_y(step);
	auto mag = gradient_magnitude(gx, gy);

	print_image(gx, "Sobel X (horizontal gradient)");
	print_image(mag, "Gradient magnitude");

	// --- Canny on Step Edge ---
	print_separator("2. Canny Edge Detection");

	auto edges = canny(step, 0.5, 1.5, 0.8);
	print_edge_map(edges, "Canny edges");

	// --- Checkerboard Pattern ---
	print_separator("3. Checkerboard: Morphological Gradient");

	std::size_t brows = 8, bcols = 8;
	mtl::mat::dense2D<double> checker(brows, bcols);
	for (std::size_t r = 0; r < brows; ++r)
		for (std::size_t c = 0; c < bcols; ++c)
			checker(r, c) = ((r / 2 + c / 2) % 2 == 0) ? 1.0 : 0.0;

	print_image(checker, "Input: checkerboard");

	auto elem = make_rect_element(3, 3);
	auto morph_grad = morphological_gradient(checker, elem);
	print_image(morph_grad, "Morphological gradient");

	// --- Gaussian Blur ---
	print_separator("4. Gaussian Blur");

	mtl::mat::dense2D<double> impulse(7, 7);
	for (std::size_t r = 0; r < 7; ++r)
		for (std::size_t c = 0; c < 7; ++c)
			impulse(r, c) = 0.0;
	impulse(3, 3) = 1.0;

	print_image(impulse, "Input: impulse");
	auto blurred = gaussian_blur(impulse, 1.0);
	print_image(blurred, "Gaussian blur (sigma=1.0)", 8, 4);

	// --- RGB Multi-channel Processing ---
	print_separator("5. Multi-Channel Processing");

	RGBImage<double> rgb(6, 6);
	for (std::size_t r = 0; r < 6; ++r) {
		for (std::size_t c = 0; c < 6; ++c) {
			rgb[0](r, c) = (c >= 3) ? 1.0 : 0.0;  // R: step at col 3
			rgb[1](r, c) = (r >= 3) ? 1.0 : 0.0;  // G: step at row 3
			rgb[2](r, c) = 0.5;                     // B: constant
		}
	}

	std::cout << "RGB image (6x6): R=horizontal step, G=vertical step, B=constant\n\n";

	auto gray = rgb_to_gray(rgb);
	print_image(gray, "Grayscale (BT.601 luminance)");

	auto gray_edges = canny(gray, 0.1, 0.3, 0.8);
	print_edge_map(gray_edges, "Canny edges on grayscale");

	// Per-channel Sobel
	auto rgb_gx = apply_per_channel(rgb, [](const mtl::mat::dense2D<double>& plane) {
		return sobel_x(plane);
	});

	std::cout << "Per-channel Sobel X:\n";
	print_image(rgb_gx[0], "  R channel Sobel X");
	print_image(rgb_gx[1], "  G channel Sobel X");
	print_image(rgb_gx[2], "  B channel Sobel X");

	// --- Border Mode Comparison ---
	print_separator("6. Border Mode Comparison");

	mtl::mat::dense2D<double> small(1, 5);
	small(0, 0) = 10; small(0, 1) = 20; small(0, 2) = 30;
	small(0, 3) = 40; small(0, 4) = 50;

	std::cout << "Input: [10, 20, 30, 40, 50]\n\n";
	std::cout << "Pixel at index -1 by border mode:\n";
	std::cout << "  Zero:        " << fetch_pixel(small, 0, -1, BorderMode::zero) << "\n";
	std::cout << "  Replicate:   " << fetch_pixel(small, 0, -1, BorderMode::replicate) << "\n";
	std::cout << "  Reflect:     " << fetch_pixel(small, 0, -1, BorderMode::reflect) << "\n";
	std::cout << "  Reflect_101: " << fetch_pixel(small, 0, -1, BorderMode::reflect_101) << "\n";
	std::cout << "  Wrap:        " << fetch_pixel(small, 0, -1, BorderMode::wrap) << "\n";

	std::cout << "\nPixel at index 5 by border mode:\n";
	std::cout << "  Zero:        " << fetch_pixel(small, 0, 5, BorderMode::zero) << "\n";
	std::cout << "  Replicate:   " << fetch_pixel(small, 0, 5, BorderMode::replicate) << "\n";
	std::cout << "  Reflect:     " << fetch_pixel(small, 0, 5, BorderMode::reflect) << "\n";
	std::cout << "  Reflect_101: " << fetch_pixel(small, 0, 5, BorderMode::reflect_101) << "\n";
	std::cout << "  Wrap:        " << fetch_pixel(small, 0, 5, BorderMode::wrap) << "\n";

	print_separator("Done");

	return 0;
}
