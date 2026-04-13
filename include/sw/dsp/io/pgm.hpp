#pragma once
// pgm.hpp: PGM (Portable Graymap) reader/writer
//
// Supports P2 (ASCII) and P5 (binary) PGM formats.
// 8-bit and 16-bit max values.
// Header-only, no external dependencies.
//
// Write: T values in [0, 1] mapped to [0, max_val].
// Read: pixel values normalized to [0, 1] as T.
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

// Write a grayscale image as binary PGM (P5).
// T values are clamped to [0, 1] and mapped to [0, max_val].
template <DspField T>
	requires ConvertibleToDouble<T>
void write_pgm(const std::string& path, const mtl::mat::dense2D<T>& image,
               int max_val = 255) {
	if (max_val < 1 || max_val > 65535)
		throw std::invalid_argument("write_pgm: max_val must be in [1, 65535]");

	std::ofstream ofs(path, std::ios::binary);
	if (!ofs)
		throw std::runtime_error("write_pgm: cannot open file: " + path);

	std::size_t rows = image.num_rows();
	std::size_t cols = image.num_cols();

	// Header
	ofs << "P5\n" << cols << " " << rows << "\n" << max_val << "\n";

	// Pixel data
	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			double v = static_cast<double>(image(r, c));
			if (v < 0.0) v = 0.0;
			if (v > 1.0) v = 1.0;
			int pixel = static_cast<int>(v * max_val + 0.5);
			if (pixel > max_val) pixel = max_val;

			if (max_val <= 255) {
				uint8_t b = static_cast<uint8_t>(pixel);
				ofs.write(reinterpret_cast<const char*>(&b), 1);
			} else {
				// 16-bit PGM: big-endian per PNM spec
				uint8_t hi = static_cast<uint8_t>((pixel >> 8) & 0xFF);
				uint8_t lo = static_cast<uint8_t>(pixel & 0xFF);
				ofs.write(reinterpret_cast<const char*>(&hi), 1);
				ofs.write(reinterpret_cast<const char*>(&lo), 1);
			}
		}
	}
}

// Read a PGM file (P2 ASCII or P5 binary), return dense2D<T> normalized to [0, 1].
template <DspField T>
	requires ConvertibleToDouble<T>
mtl::mat::dense2D<T> read_pgm(const std::string& path) {
	std::ifstream ifs(path, std::ios::binary);
	if (!ifs)
		throw std::runtime_error("read_pgm: cannot open file: " + path);

	// Read magic number
	std::string magic;
	ifs >> magic;
	if (magic != "P2" && magic != "P5")
		throw std::runtime_error("read_pgm: unsupported format: " + magic);

	bool binary = (magic == "P5");

	// Skip comments
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
		throw std::runtime_error("read_pgm: invalid dimensions or max_val");

	// Skip single whitespace after max_val (before binary data)
	if (binary) ifs.get();

	std::size_t rows = static_cast<std::size_t>(height);
	std::size_t cols = static_cast<std::size_t>(width);
	mtl::mat::dense2D<T> image(rows, cols);
	double inv_max = 1.0 / static_cast<double>(max_val);

	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			int pixel;
			if (binary) {
				if (max_val <= 255) {
					uint8_t b;
					ifs.read(reinterpret_cast<char*>(&b), 1);
					pixel = b;
				} else {
					uint8_t b[2];
					ifs.read(reinterpret_cast<char*>(b), 2);
					pixel = (static_cast<int>(b[0]) << 8) | b[1];
				}
			} else {
				ifs >> pixel;
			}
			image(r, c) = static_cast<T>(static_cast<double>(pixel) * inv_max);
		}
	}

	return image;
}

} // namespace sw::dsp::io
