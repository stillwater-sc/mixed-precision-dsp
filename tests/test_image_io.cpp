// test_image_io.cpp: test PGM, PPM, and BMP image file I/O
//
// Tests round-trip: generate → write → read → compare.
// Uses temporary files cleaned up after each test.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/io/pgm.hpp>
#include <sw/dsp/io/ppm.hpp>
#include <sw/dsp/io/bmp.hpp>
#include <sw/dsp/image/generators.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace sw::dsp;

bool near(double a, double b, double eps = 1e-4) {
	return std::abs(a - b) < eps;
}

// Platform-portable temp directory
inline std::string temp_dir() {
	auto p = std::filesystem::temp_directory_path();
	return p.string();
}

// RAII temp file that deletes on destruction
struct TempFile {
	std::string path;
	TempFile(const std::string& name) : path(temp_dir() + "/dsp_test_" + name) {}
	~TempFile() { std::remove(path.c_str()); }
};

// ========== PGM Tests ==========

void test_pgm_roundtrip_checkerboard() {
	// Generate a checkerboard, write PGM, read back, compare.
	// 8-bit quantization: values are exact for 0 and 1.
	auto img = checkerboard<float>(8, 8, 2, 0.0f, 1.0f);

	TempFile tmp("checker.pgm");
	io::write_pgm(tmp.path, img);
	auto loaded = io::read_pgm<float>(tmp.path);

	if (!(loaded.num_rows() == 8 && loaded.num_cols() == 8))
		throw std::runtime_error("test failed: PGM roundtrip dimensions");

	for (std::size_t r = 0; r < 8; ++r)
		for (std::size_t c = 0; c < 8; ++c)
			if (!(near(loaded(r, c), img(r, c), 1.0 / 255.0 + 1e-6)))
				throw std::runtime_error("test failed: PGM roundtrip pixel mismatch");

	std::cout << "  pgm_roundtrip_checkerboard: passed\n";
}

void test_pgm_roundtrip_gradient() {
	auto img = gradient_horizontal<float>(4, 256);

	TempFile tmp("gradient.pgm");
	io::write_pgm(tmp.path, img);
	auto loaded = io::read_pgm<float>(tmp.path);

	// 8-bit quantization: error should be <= 1/255 ≈ 0.004
	double max_err = 0;
	for (std::size_t r = 0; r < 4; ++r) {
		for (std::size_t c = 0; c < 256; ++c) {
			double err = std::abs(static_cast<double>(loaded(r, c))
			                    - static_cast<double>(img(r, c)));
			if (err > max_err) max_err = err;
		}
	}
	if (!(max_err < 1.0 / 255.0 + 1e-6))
		throw std::runtime_error("test failed: PGM gradient max error too large: "
			+ std::to_string(max_err));

	std::cout << "  pgm_roundtrip_gradient: passed (max_err=" << max_err << ")\n";
}

void test_pgm_16bit() {
	auto img = gradient_horizontal<double>(2, 100);

	TempFile tmp("gradient16.pgm");
	io::write_pgm(tmp.path, img, 65535);
	auto loaded = io::read_pgm<double>(tmp.path);

	// 16-bit: error should be <= 1/65535 ≈ 1.5e-5
	double max_err = 0;
	for (std::size_t r = 0; r < 2; ++r) {
		for (std::size_t c = 0; c < 100; ++c) {
			double err = std::abs(loaded(r, c) - img(r, c));
			if (err > max_err) max_err = err;
		}
	}
	if (!(max_err < 1.0 / 65535.0 + 1e-6))
		throw std::runtime_error("test failed: PGM 16-bit max error: "
			+ std::to_string(max_err));

	std::cout << "  pgm_16bit: passed (max_err=" << max_err << ")\n";
}

void test_pgm_validation() {
	auto img = checkerboard<float>(4, 4, 2);

	bool caught = false;
	try { io::write_pgm((temp_dir() + "/dsp_test_bad.pgm").c_str(), img, 0); }
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: write_pgm should reject max_val=0");

	caught = false;
	try { io::read_pgm<float>("/nonexistent/path.pgm"); }
	catch (const std::runtime_error&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: read_pgm should throw on missing file");

	std::cout << "  pgm_validation: passed\n";
}

// ========== PPM Tests ==========

void test_ppm_roundtrip() {
	std::size_t rows = 6, cols = 8;
	auto red = gradient_horizontal<float>(rows, cols);
	auto green = gradient_vertical<float>(rows, cols);
	auto blue = checkerboard<float>(rows, cols, 2, 0.2f, 0.8f);

	TempFile tmp("color.ppm");
	io::write_ppm(tmp.path, red, green, blue);
	auto loaded = io::read_ppm<float>(tmp.path);

	if (!(loaded.rows() == rows && loaded.cols() == cols))
		throw std::runtime_error("test failed: PPM roundtrip dimensions");

	double max_err = 0;
	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			double er = std::abs(static_cast<double>(loaded.r(r, c) - red(r, c)));
			double eg = std::abs(static_cast<double>(loaded.g(r, c) - green(r, c)));
			double eb = std::abs(static_cast<double>(loaded.b(r, c) - blue(r, c)));
			double e = std::max({er, eg, eb});
			if (e > max_err) max_err = e;
		}
	}
	if (!(max_err < 1.0 / 255.0 + 1e-6))
		throw std::runtime_error("test failed: PPM roundtrip max error: "
			+ std::to_string(max_err));

	std::cout << "  ppm_roundtrip: passed (max_err=" << max_err << ")\n";
}

void test_ppm_validation() {
	auto r = checkerboard<float>(4, 4, 2);
	auto g = checkerboard<float>(4, 4, 2);
	auto b = checkerboard<float>(3, 4, 2);  // wrong size

	bool caught = false;
	try { io::write_ppm((temp_dir() + "/dsp_test_bad.ppm").c_str(), r, g, b); }
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: write_ppm should reject mismatched channels");

	std::cout << "  ppm_validation: passed\n";
}

// ========== BMP Tests ==========

void test_bmp_grayscale_roundtrip() {
	auto img = checkerboard<float>(10, 12, 3, 0.0f, 1.0f);

	TempFile tmp("gray.bmp");
	io::write_bmp(tmp.path, img);
	auto loaded = io::read_bmp<float>(tmp.path);

	if (!(loaded.rows() == 10 && loaded.cols() == 12))
		throw std::runtime_error("test failed: BMP grayscale roundtrip dimensions");
	if (!(loaded.bits_per_pixel == 8))
		throw std::runtime_error("test failed: BMP should be 8-bit");

	double max_err = 0;
	for (std::size_t r = 0; r < 10; ++r) {
		for (std::size_t c = 0; c < 12; ++c) {
			double err = std::abs(static_cast<double>(loaded.r(r, c) - img(r, c)));
			if (err > max_err) max_err = err;
		}
	}
	if (!(max_err < 1.0 / 255.0 + 1e-6))
		throw std::runtime_error("test failed: BMP grayscale max error: "
			+ std::to_string(max_err));

	std::cout << "  bmp_grayscale_roundtrip: passed (max_err=" << max_err << ")\n";
}

void test_bmp_color_roundtrip() {
	std::size_t rows = 8, cols = 10;
	auto red = gradient_horizontal<float>(rows, cols);
	auto green = gradient_vertical<float>(rows, cols);
	auto blue = checkerboard<float>(rows, cols, 2, 0.1f, 0.9f);

	TempFile tmp("color.bmp");
	io::write_bmp(tmp.path, red, green, blue);
	auto loaded = io::read_bmp<float>(tmp.path);

	if (!(loaded.rows() == rows && loaded.cols() == cols))
		throw std::runtime_error("test failed: BMP color roundtrip dimensions");
	if (!(loaded.bits_per_pixel == 24))
		throw std::runtime_error("test failed: BMP should be 24-bit");

	double max_err = 0;
	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			double er = std::abs(static_cast<double>(loaded.r(r, c) - red(r, c)));
			double eg = std::abs(static_cast<double>(loaded.g(r, c) - green(r, c)));
			double eb = std::abs(static_cast<double>(loaded.b(r, c) - blue(r, c)));
			double e = std::max({er, eg, eb});
			if (e > max_err) max_err = e;
		}
	}
	if (!(max_err < 1.0 / 255.0 + 1e-6))
		throw std::runtime_error("test failed: BMP color max error: "
			+ std::to_string(max_err));

	std::cout << "  bmp_color_roundtrip: passed (max_err=" << max_err << ")\n";
}

// ========== Cross-format Test ==========

void test_pgm_bmp_consistency() {
	// Write the same image as PGM and BMP, read both, compare
	auto img = gaussian_blob<float>(16, 16, 3.0f);

	TempFile pgm_file("blob.pgm");
	TempFile bmp_file("blob.bmp");
	io::write_pgm(pgm_file.path, img);
	io::write_bmp(bmp_file.path, img);

	auto pgm = io::read_pgm<float>(pgm_file.path);
	auto bmp = io::read_bmp<float>(bmp_file.path);

	double max_diff = 0;
	for (std::size_t r = 0; r < 16; ++r)
		for (std::size_t c = 0; c < 16; ++c) {
			double d = std::abs(static_cast<double>(pgm(r, c) - bmp.r(r, c)));
			if (d > max_diff) max_diff = d;
		}
	// Both 8-bit, should be identical
	if (!(max_diff < 1e-6))
		throw std::runtime_error("test failed: PGM/BMP should produce same 8-bit values");

	std::cout << "  pgm_bmp_consistency: passed\n";
}

int main() {
	try {
		std::cout << "Image I/O Tests\n";

		// PGM
		test_pgm_roundtrip_checkerboard();
		test_pgm_roundtrip_gradient();
		test_pgm_16bit();
		test_pgm_validation();

		// PPM
		test_ppm_roundtrip();
		test_ppm_validation();

		// BMP
		test_bmp_grayscale_roundtrip();
		test_bmp_color_roundtrip();

		// Cross-format
		test_pgm_bmp_consistency();

		std::cout << "All image I/O tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << '\n';
		return 1;
	}
}
