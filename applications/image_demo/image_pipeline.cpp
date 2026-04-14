// image_pipeline.cpp: canonical computer vision image processing pipeline
//
// Demonstrates the full sw::dsp image processing chain:
//   1. Generate synthetic test scene (composite of shapes)
//   2. Add Gaussian noise (simulating sensor noise)
//   3. Denoise with Gaussian blur
//   4. Compute gradient magnitude (Sobel)
//   5. Detect edges (Canny)
//   6. Clean up edges with morphological closing
//   7. Write each stage as a PGM file for visual inspection
//
// Also generates canonical test patterns (zone plate, checkerboard)
// and writes them as PGM/PPM for frequency response evaluation.
//
// Output files are written to the current working directory.
// View with: display *.pgm (ImageMagick), or open in GIMP/Python PIL.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/image/image.hpp>
#include <sw/dsp/image/generators.hpp>
#include <sw/dsp/io/pgm.hpp>
#include <sw/dsp/io/ppm.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

using namespace sw::dsp;

void print_separator(const std::string& title) {
	std::cout << "\n" << std::string(70, '=') << "\n";
	std::cout << "  " << title << "\n";
	std::cout << std::string(70, '=') << "\n\n";
}

// Print image statistics
template <typename T>
void print_stats(const mtl::mat::dense2D<T>& img, const std::string& label) {
	double min_v = 1e30, max_v = -1e30, sum = 0;
	std::size_t n = img.num_rows() * img.num_cols();
	for (std::size_t r = 0; r < img.num_rows(); ++r) {
		for (std::size_t c = 0; c < img.num_cols(); ++c) {
			double v = static_cast<double>(img(r, c));
			if (v < min_v) min_v = v;
			if (v > max_v) max_v = v;
			sum += v;
		}
	}
	std::cout << "  " << std::setw(28) << std::left << label
	          << " min=" << std::setw(8) << std::fixed << std::setprecision(4) << min_v
	          << " max=" << std::setw(8) << max_v
	          << " mean=" << std::setw(8) << (sum / static_cast<double>(n))
	          << " [" << img.num_rows() << "x" << img.num_cols() << "]\n";
}

int main() {
	constexpr std::size_t W = 256;
	constexpr std::size_t H = 256;

	print_separator("Image Processing Pipeline Demo");
	std::cout << "Image size: " << H << "x" << W << "\n";
	std::cout << "Output: PGM files in current directory\n";

	// ================================================================
	// Stage 1: Generate synthetic test scene
	// ================================================================
	print_separator("Stage 1: Scene Generation");

	// Composite scene: circle on gradient background with checkerboard inset
	auto background = gradient_radial<float>(H, W, 0.3f, 0.0f);
	auto disk = circle<float>(H, W, 60, 0.8f, 0.0f);
	auto small_checker = checkerboard<float>(H, W, 16, 0.0f, 0.5f);
	auto small_rect = rectangle<float>(H, W, 160, 160, 80, 80, 1.0f, 0.0f);

	// Compose: background + disk + checkerboard in bottom-right corner
	mtl::mat::dense2D<float> scene(H, W);
	for (std::size_t r = 0; r < H; ++r) {
		for (std::size_t c = 0; c < W; ++c) {
			float v = background(r, c);
			if (disk(r, c) > 0.5f) v = disk(r, c);
			if (small_rect(r, c) > 0.5f) v = small_checker(r, c);
			scene(r, c) = v;
		}
	}

	print_stats(scene, "scene (composite)");
	io::write_pgm("01_scene.pgm", scene);
	std::cout << "  -> 01_scene.pgm\n";

	// ================================================================
	// Stage 2: Add noise
	// ================================================================
	print_separator("Stage 2: Add Gaussian Noise (sigma=0.08)");

	auto noisy = add_noise(scene, 0.08, 42);
	print_stats(noisy, "noisy scene");
	io::write_pgm("02_noisy.pgm", noisy);
	std::cout << "  -> 02_noisy.pgm\n";

	// ================================================================
	// Stage 3: Denoise with Gaussian blur
	// ================================================================
	print_separator("Stage 3: Gaussian Blur (sigma=1.5)");

	auto denoised = gaussian_blur(noisy, 1.5);
	print_stats(denoised, "denoised (Gaussian blur)");
	io::write_pgm("03_denoised.pgm", denoised);
	std::cout << "  -> 03_denoised.pgm\n";

	// ================================================================
	// Stage 4: Sobel gradient magnitude
	// ================================================================
	print_separator("Stage 4: Sobel Gradient Magnitude");

	auto gx = sobel_x(denoised);
	auto gy = sobel_y(denoised);
	auto mag = gradient_magnitude(gx, gy);

	// Normalize gradient to [0, 1] for display
	float gmax = 0;
	for (std::size_t r = 0; r < H; ++r)
		for (std::size_t c = 0; c < W; ++c)
			if (mag(r, c) > gmax) gmax = mag(r, c);
	if (gmax > 0) {
		for (std::size_t r = 0; r < H; ++r)
			for (std::size_t c = 0; c < W; ++c)
				mag(r, c) = mag(r, c) / gmax;
	}

	print_stats(mag, "gradient magnitude (norm)");
	io::write_pgm("04_gradient.pgm", mag);
	std::cout << "  -> 04_gradient.pgm\n";

	// ================================================================
	// Stage 5: Canny edge detection
	// ================================================================
	print_separator("Stage 5: Canny Edge Detection");

	auto edges = canny(denoised, 0.3, 0.8, 1.0);
	print_stats(edges, "Canny edges");
	io::write_pgm("05_canny.pgm", edges);
	std::cout << "  -> 05_canny.pgm\n";

	// ================================================================
	// Stage 6: Morphological closing (fill gaps in edges)
	// ================================================================
	print_separator("Stage 6: Morphological Closing");

	auto elem = make_rect_element(3, 3);
	auto closed = morphological_close(edges, elem);
	print_stats(closed, "edges after closing");
	io::write_pgm("06_closed.pgm", closed);
	std::cout << "  -> 06_closed.pgm\n";

	// ================================================================
	// Stage 7: Threshold the gradient for a binary edge mask
	// ================================================================
	print_separator("Stage 7: Threshold (gradient > 0.15)");

	// Re-compute gradient on denoised (unnormalized) for thresholding
	auto gx2 = sobel_x(denoised);
	auto gy2 = sobel_y(denoised);
	auto mag2 = gradient_magnitude(gx2, gy2);
	auto binary = threshold(mag2, 0.15f * gmax);
	print_stats(binary, "binary edge mask");
	io::write_pgm("07_threshold.pgm", binary);
	std::cout << "  -> 07_threshold.pgm\n";

	// ================================================================
	// Canonical test patterns
	// ================================================================
	print_separator("Canonical Test Patterns");

	// Zone plate — spatial frequency sweep
	auto zp = zone_plate<float>(H, W);
	print_stats(zp, "zone plate");
	io::write_pgm("pattern_zone_plate.pgm", zp);
	std::cout << "  -> pattern_zone_plate.pgm\n";

	// Zone plate after Gaussian blur — shows frequency cutoff
	auto zp_blurred = gaussian_blur(zp, 2.0);
	print_stats(zp_blurred, "zone plate (blurred)");
	io::write_pgm("pattern_zone_plate_blur.pgm", zp_blurred);
	std::cout << "  -> pattern_zone_plate_blur.pgm\n";

	// Checkerboard
	auto cb = checkerboard<float>(H, W, 8);
	io::write_pgm("pattern_checkerboard.pgm", cb);
	std::cout << "  -> pattern_checkerboard.pgm\n";

	// Gaussian blob
	auto blob = gaussian_blob<float>(H, W, 30.0);
	io::write_pgm("pattern_gaussian_blob.pgm", blob);
	std::cout << "  -> pattern_gaussian_blob.pgm\n";

	// Color PPM: RGB gradient
	auto red_ch   = gradient_horizontal<float>(H, W);
	auto green_ch = gradient_vertical<float>(H, W);
	auto blue_ch  = gradient_radial<float>(H, W);
	io::write_ppm("pattern_rgb_gradient.ppm", red_ch, green_ch, blue_ch);
	std::cout << "  -> pattern_rgb_gradient.ppm\n";

	// ================================================================
	// Summary
	// ================================================================
	print_separator("Output Summary");

	std::cout << "  Pipeline stages:\n";
	std::cout << "    01_scene.pgm           - Synthetic composite scene\n";
	std::cout << "    02_noisy.pgm           - Scene + Gaussian noise\n";
	std::cout << "    03_denoised.pgm        - After Gaussian blur\n";
	std::cout << "    04_gradient.pgm        - Sobel gradient magnitude\n";
	std::cout << "    05_canny.pgm           - Canny edge detection\n";
	std::cout << "    06_closed.pgm          - Morphological closing\n";
	std::cout << "    07_threshold.pgm       - Binary edge mask\n";
	std::cout << "\n  Test patterns:\n";
	std::cout << "    pattern_zone_plate.pgm      - Frequency sweep\n";
	std::cout << "    pattern_zone_plate_blur.pgm - Frequency sweep after blur\n";
	std::cout << "    pattern_checkerboard.pgm    - 8x8 blocks\n";
	std::cout << "    pattern_gaussian_blob.pgm   - 2D Gaussian\n";
	std::cout << "    pattern_rgb_gradient.ppm    - RGB color gradient\n";
	std::cout << "\n  View with: display *.pgm  (ImageMagick)\n";
	std::cout << "             python3 -c \"from PIL import Image; Image.open('05_canny.pgm').show()\"\n";
	std::cout << "             gimp 04_gradient.pgm\n";

	return 0;
}
