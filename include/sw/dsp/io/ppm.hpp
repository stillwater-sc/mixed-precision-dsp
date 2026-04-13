#pragma once
// ppm.hpp: PPM (Portable Pixmap) reader/writer
//
// Supports P3 (ASCII) and P6 (binary) PPM formats.
// Color images stored as three separate dense2D<T> channel matrices
// (planar layout, consistent with Image<T, 3>).
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <mtl/mat/dense2D.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp::io {

// Result of reading a PPM file: three channel matrices.
template <typename T>
struct PpmData {
	mtl::mat::dense2D<T> r, g, b;
	std::size_t rows() const { return r.num_rows(); }
	std::size_t cols() const { return r.num_cols(); }
};

// Write a color image as binary PPM (P6).
// Each channel's T values are clamped to [0, 1] and mapped to [0, max_val].
template <DspField T>
	requires ConvertibleToDouble<T>
void write_ppm(const std::string& path,
               const mtl::mat::dense2D<T>& red,
               const mtl::mat::dense2D<T>& green,
               const mtl::mat::dense2D<T>& blue,
               int max_val = 255) {
	if (max_val < 1 || max_val > 255)
		throw std::invalid_argument("write_ppm: max_val must be in [1, 255]");

	std::size_t rows = red.num_rows();
	std::size_t cols = red.num_cols();
	if (green.num_rows() != rows || green.num_cols() != cols ||
	    blue.num_rows() != rows || blue.num_cols() != cols)
		throw std::invalid_argument("write_ppm: channel dimensions must match");

	std::ofstream ofs(path, std::ios::binary);
	if (!ofs)
		throw std::runtime_error("write_ppm: cannot open file: " + path);

	ofs << "P6\n" << cols << " " << rows << "\n" << max_val << "\n";

	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			auto clamp_pixel = [max_val](double v) -> uint8_t {
				if (v < 0.0) v = 0.0;
				if (v > 1.0) v = 1.0;
				int p = static_cast<int>(v * max_val + 0.5);
				if (p > max_val) p = max_val;
				return static_cast<uint8_t>(p);
			};
			uint8_t rgb[3] = {
				clamp_pixel(static_cast<double>(red(r, c))),
				clamp_pixel(static_cast<double>(green(r, c))),
				clamp_pixel(static_cast<double>(blue(r, c)))
			};
			ofs.write(reinterpret_cast<const char*>(rgb), 3);
		}
	}
}

// Read a PPM file (P3 ASCII or P6 binary), return PpmData<T> with channels normalized to [0, 1].
template <DspField T>
	requires ConvertibleToDouble<T>
PpmData<T> read_ppm(const std::string& path) {
	std::ifstream ifs(path, std::ios::binary);
	if (!ifs)
		throw std::runtime_error("read_ppm: cannot open file: " + path);

	std::string magic;
	ifs >> magic;
	if (magic != "P3" && magic != "P6")
		throw std::runtime_error("read_ppm: unsupported format: " + magic);

	bool binary = (magic == "P6");

	auto skip_comments = [&]() {
		while (ifs.peek() == '#' || ifs.peek() == '\n' || ifs.peek() == '\r' || ifs.peek() == ' ') {
			if (ifs.peek() == '#') {
				std::string line;
				std::getline(ifs, line);
			} else {
				ifs.get();
			}
		}
	};

	skip_comments();
	int width, height, max_val;
	ifs >> width >> height;
	skip_comments();
	ifs >> max_val;

	if (width <= 0 || height <= 0 || max_val <= 0)
		throw std::runtime_error("read_ppm: invalid dimensions or max_val");

	if (binary) ifs.get();

	std::size_t rows = static_cast<std::size_t>(height);
	std::size_t cols = static_cast<std::size_t>(width);
	double inv_max = 1.0 / static_cast<double>(max_val);

	PpmData<T> data;
	data.r = mtl::mat::dense2D<T>(rows, cols);
	data.g = mtl::mat::dense2D<T>(rows, cols);
	data.b = mtl::mat::dense2D<T>(rows, cols);

	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			int rv, gv, bv;
			if (binary) {
				uint8_t rgb[3];
				ifs.read(reinterpret_cast<char*>(rgb), 3);
				rv = rgb[0]; gv = rgb[1]; bv = rgb[2];
			} else {
				ifs >> rv >> gv >> bv;
			}
			data.r(r, c) = static_cast<T>(rv * inv_max);
			data.g(r, c) = static_cast<T>(gv * inv_max);
			data.b(r, c) = static_cast<T>(bv * inv_max);
		}
	}

	return data;
}

} // namespace sw::dsp::io
