#pragma once
// bmp.hpp: BMP (Windows Bitmap) reader/writer
//
// Supports 8-bit grayscale (with palette) and 24-bit RGB.
// Little-endian per the BMP specification.
// Header-only, no external dependencies.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <mtl/mat/dense2D.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp::io {

namespace detail {

inline void bmp_write_le16(std::ostream& os, uint16_t v) {
	char b[2] = { static_cast<char>(v & 0xFF), static_cast<char>((v >> 8) & 0xFF) };
	os.write(b, 2);
}

inline void bmp_write_le32(std::ostream& os, uint32_t v) {
	char b[4] = { static_cast<char>(v & 0xFF), static_cast<char>((v >> 8) & 0xFF),
	              static_cast<char>((v >> 16) & 0xFF), static_cast<char>((v >> 24) & 0xFF) };
	os.write(b, 4);
}

inline uint16_t bmp_read_le16(std::istream& is) {
	uint8_t b[2];
	is.read(reinterpret_cast<char*>(b), 2);
	return static_cast<uint16_t>(b[0]) | (static_cast<uint16_t>(b[1]) << 8);
}

inline uint32_t bmp_read_le32(std::istream& is) {
	uint8_t b[4];
	is.read(reinterpret_cast<char*>(b), 4);
	return static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
	       (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
}

} // namespace detail

// Write an 8-bit grayscale BMP with a 256-entry grayscale palette.
template <DspField T>
	requires ConvertibleToDouble<T>
void write_bmp(const std::string& path, const mtl::mat::dense2D<T>& image) {
	std::ofstream ofs(path, std::ios::binary);
	if (!ofs)
		throw std::runtime_error("write_bmp: cannot open file: " + path);

	uint32_t rows = static_cast<uint32_t>(image.num_rows());
	uint32_t cols = static_cast<uint32_t>(image.num_cols());

	// BMP rows must be padded to 4-byte boundaries
	uint32_t row_stride = (cols + 3) & ~3u;
	uint32_t palette_size = 256 * 4;  // 256 RGBQUAD entries
	uint32_t header_size = 14 + 40;   // BITMAPFILEHEADER + BITMAPINFOHEADER
	uint32_t pixel_offset = header_size + palette_size;
	uint32_t image_size = row_stride * rows;
	uint32_t file_size = pixel_offset + image_size;

	// BITMAPFILEHEADER (14 bytes)
	ofs.write("BM", 2);
	detail::bmp_write_le32(ofs, file_size);
	detail::bmp_write_le16(ofs, 0);  // reserved
	detail::bmp_write_le16(ofs, 0);  // reserved
	detail::bmp_write_le32(ofs, pixel_offset);

	// BITMAPINFOHEADER (40 bytes)
	detail::bmp_write_le32(ofs, 40);         // header size
	detail::bmp_write_le32(ofs, cols);        // width
	detail::bmp_write_le32(ofs, rows);        // height (positive = bottom-up)
	detail::bmp_write_le16(ofs, 1);           // planes
	detail::bmp_write_le16(ofs, 8);           // bits per pixel
	detail::bmp_write_le32(ofs, 0);           // compression (BI_RGB)
	detail::bmp_write_le32(ofs, image_size);  // image size
	detail::bmp_write_le32(ofs, 2835);        // x pixels per meter (~72 DPI)
	detail::bmp_write_le32(ofs, 2835);        // y pixels per meter
	detail::bmp_write_le32(ofs, 256);         // colors used
	detail::bmp_write_le32(ofs, 0);           // important colors

	// Grayscale palette: 256 entries of (B, G, R, 0)
	for (int i = 0; i < 256; ++i) {
		uint8_t entry[4] = { static_cast<uint8_t>(i), static_cast<uint8_t>(i),
		                     static_cast<uint8_t>(i), 0 };
		ofs.write(reinterpret_cast<const char*>(entry), 4);
	}

	// Pixel data (bottom-up row order)
	std::vector<uint8_t> row_buf(row_stride, 0);
	for (uint32_t r = 0; r < rows; ++r) {
		uint32_t src_row = rows - 1 - r;  // BMP is bottom-up
		for (uint32_t c = 0; c < cols; ++c) {
			double v = static_cast<double>(image(src_row, c));
			if (v < 0.0) v = 0.0;
			if (v > 1.0) v = 1.0;
			row_buf[c] = static_cast<uint8_t>(v * 255.0 + 0.5);
		}
		ofs.write(reinterpret_cast<const char*>(row_buf.data()),
		          static_cast<std::streamsize>(row_stride));
	}
}

// Write a 24-bit color BMP from three channel matrices.
template <DspField T>
	requires ConvertibleToDouble<T>
void write_bmp(const std::string& path,
               const mtl::mat::dense2D<T>& red,
               const mtl::mat::dense2D<T>& green,
               const mtl::mat::dense2D<T>& blue) {
	std::size_t rows = red.num_rows();
	std::size_t cols = red.num_cols();
	if (green.num_rows() != rows || green.num_cols() != cols ||
	    blue.num_rows() != rows || blue.num_cols() != cols)
		throw std::invalid_argument("write_bmp: channel dimensions must match");

	std::ofstream ofs(path, std::ios::binary);
	if (!ofs)
		throw std::runtime_error("write_bmp: cannot open file: " + path);

	uint32_t h = static_cast<uint32_t>(rows);
	uint32_t w = static_cast<uint32_t>(cols);
	uint32_t row_stride = (w * 3 + 3) & ~3u;
	uint32_t header_size = 14 + 40;
	uint32_t image_size = row_stride * h;
	uint32_t file_size = header_size + image_size;

	// BITMAPFILEHEADER
	ofs.write("BM", 2);
	detail::bmp_write_le32(ofs, file_size);
	detail::bmp_write_le16(ofs, 0);
	detail::bmp_write_le16(ofs, 0);
	detail::bmp_write_le32(ofs, header_size);

	// BITMAPINFOHEADER
	detail::bmp_write_le32(ofs, 40);
	detail::bmp_write_le32(ofs, w);
	detail::bmp_write_le32(ofs, h);
	detail::bmp_write_le16(ofs, 1);
	detail::bmp_write_le16(ofs, 24);
	detail::bmp_write_le32(ofs, 0);
	detail::bmp_write_le32(ofs, image_size);
	detail::bmp_write_le32(ofs, 2835);
	detail::bmp_write_le32(ofs, 2835);
	detail::bmp_write_le32(ofs, 0);
	detail::bmp_write_le32(ofs, 0);

	auto clamp_byte = [](double v) -> uint8_t {
		if (v < 0.0) v = 0.0;
		if (v > 1.0) v = 1.0;
		return static_cast<uint8_t>(v * 255.0 + 0.5);
	};

	// Pixel data (bottom-up, BGR order)
	std::vector<uint8_t> row_buf(row_stride, 0);
	for (uint32_t r = 0; r < h; ++r) {
		uint32_t src_row = h - 1 - r;
		for (uint32_t c = 0; c < w; ++c) {
			row_buf[c * 3 + 0] = clamp_byte(static_cast<double>(blue(src_row, c)));
			row_buf[c * 3 + 1] = clamp_byte(static_cast<double>(green(src_row, c)));
			row_buf[c * 3 + 2] = clamp_byte(static_cast<double>(red(src_row, c)));
		}
		ofs.write(reinterpret_cast<const char*>(row_buf.data()),
		          static_cast<std::streamsize>(row_stride));
	}
}

// BMP read result.
template <typename T>
struct BmpData {
	mtl::mat::dense2D<T> r, g, b;
	int bits_per_pixel{0};
	std::size_t rows() const { return r.num_rows(); }
	std::size_t cols() const { return r.num_cols(); }
	bool is_grayscale() const { return bits_per_pixel == 8; }
};

// Read a BMP file. Supports 8-bit (grayscale palette) and 24-bit RGB.
// Returns BmpData with channels normalized to [0, 1].
// For 8-bit, r == g == b (all contain the grayscale values).
template <DspField T>
	requires ConvertibleToDouble<T>
BmpData<T> read_bmp(const std::string& path) {
	std::ifstream ifs(path, std::ios::binary);
	if (!ifs)
		throw std::runtime_error("read_bmp: cannot open file: " + path);

	// BITMAPFILEHEADER
	char sig[2];
	ifs.read(sig, 2);
	if (sig[0] != 'B' || sig[1] != 'M')
		throw std::runtime_error("read_bmp: not a BMP file");

	detail::bmp_read_le32(ifs);  // file size
	detail::bmp_read_le16(ifs);  // reserved
	detail::bmp_read_le16(ifs);  // reserved
	uint32_t pixel_offset = detail::bmp_read_le32(ifs);

	// BITMAPINFOHEADER
	uint32_t info_size = detail::bmp_read_le32(ifs);
	if (info_size < 40)
		throw std::runtime_error("read_bmp: unsupported header size");

	int32_t width = static_cast<int32_t>(detail::bmp_read_le32(ifs));
	int32_t height = static_cast<int32_t>(detail::bmp_read_le32(ifs));
	detail::bmp_read_le16(ifs);  // planes
	uint16_t bpp = detail::bmp_read_le16(ifs);
	uint32_t compression = detail::bmp_read_le32(ifs);

	if (compression != 0)
		throw std::runtime_error("read_bmp: only uncompressed BMP supported");
	if (bpp != 8 && bpp != 24)
		throw std::runtime_error("read_bmp: only 8-bit and 24-bit BMP supported");

	bool bottom_up = (height > 0);
	uint32_t h = static_cast<uint32_t>(bottom_up ? height : -height);
	uint32_t w = static_cast<uint32_t>(width);

	// Seek to pixel data
	ifs.seekg(pixel_offset, std::ios::beg);

	BmpData<T> data;
	data.bits_per_pixel = bpp;
	data.r = mtl::mat::dense2D<T>(h, w);
	data.g = mtl::mat::dense2D<T>(h, w);
	data.b = mtl::mat::dense2D<T>(h, w);

	if (bpp == 8) {
		// Read palette (we assume grayscale: palette[i] = (i, i, i, 0))
		// Skip to pixel data (already seeked)
		uint32_t row_stride = (w + 3) & ~3u;
		std::vector<uint8_t> row_buf(row_stride);

		for (uint32_t r = 0; r < h; ++r) {
			uint32_t dst_row = bottom_up ? (h - 1 - r) : r;
			ifs.read(reinterpret_cast<char*>(row_buf.data()),
			         static_cast<std::streamsize>(row_stride));
			for (uint32_t c = 0; c < w; ++c) {
				T val = static_cast<T>(static_cast<double>(row_buf[c]) / 255.0);
				data.r(dst_row, c) = val;
				data.g(dst_row, c) = val;
				data.b(dst_row, c) = val;
			}
		}
	} else {  // 24-bit
		uint32_t row_stride = (w * 3 + 3) & ~3u;
		std::vector<uint8_t> row_buf(row_stride);

		for (uint32_t r = 0; r < h; ++r) {
			uint32_t dst_row = bottom_up ? (h - 1 - r) : r;
			ifs.read(reinterpret_cast<char*>(row_buf.data()),
			         static_cast<std::streamsize>(row_stride));
			for (uint32_t c = 0; c < w; ++c) {
				data.b(dst_row, c) = static_cast<T>(row_buf[c * 3 + 0] / 255.0);
				data.g(dst_row, c) = static_cast<T>(row_buf[c * 3 + 1] / 255.0);
				data.r(dst_row, c) = static_cast<T>(row_buf[c * 3 + 2] / 255.0);
			}
		}
	}

	return data;
}

} // namespace sw::dsp::io
