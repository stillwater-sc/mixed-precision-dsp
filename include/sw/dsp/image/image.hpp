#pragma once
// image.hpp: planar image container and utilities for 2D signal processing
//
// Multi-channel images are represented as an array of single-channel
// dense2D<T> planes. All core algorithms operate on individual planes;
// multi-channel processing is handled by apply_per_channel().
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <mtl/mat/dense2D.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

// Border extrapolation mode for spatial filtering operations.
enum class BorderMode : std::uint8_t {
	zero,         // pad with T{0}
	constant,     // pad with user-specified value
	replicate,    // clamp to nearest edge pixel
	reflect,      // mirror including the edge pixel: dcba|abcd|dcba
	reflect_101,  // mirror excluding the edge pixel: dcb|abcd|cba (default)
	wrap          // periodic tiling: abcd|abcd|abcd
};

constexpr std::string_view to_string(BorderMode mode) {
	switch (mode) {
	case BorderMode::zero:        return "Zero";
	case BorderMode::constant:    return "Constant";
	case BorderMode::replicate:   return "Replicate";
	case BorderMode::reflect:     return "Reflect";
	case BorderMode::reflect_101: return "Reflect101";
	case BorderMode::wrap:        return "Wrap";
	}
	return "Unknown";
}

// Map an out-of-bounds coordinate to a valid index using the given border mode.
// rows_or_cols is the size of the dimension being indexed.
inline std::size_t border_index(int idx, std::size_t dim, BorderMode mode) {
	int n = static_cast<int>(dim);
	if (idx >= 0 && idx < n) return static_cast<std::size_t>(idx);

	switch (mode) {
	case BorderMode::zero:
	case BorderMode::constant:
		// Caller must handle these by returning a constant value
		return 0;  // sentinel — caller checks bounds first
	case BorderMode::replicate:
		if (idx < 0) return 0;
		return static_cast<std::size_t>(n - 1);
	case BorderMode::reflect: {
		// dcba|abcd|dcba  (edge pixel repeated)
		if (n == 1) return 0;
		int period = 2 * n - 2;
		int p = ((idx % period) + period) % period;
		return static_cast<std::size_t>(p < n ? p : period - p);
	}
	case BorderMode::reflect_101: {
		// dcb|abcd|cba  (edge pixel not repeated)
		if (n == 1) return 0;
		int period = 2 * (n - 1);
		int p = ((idx % period) + period) % period;
		return static_cast<std::size_t>(p < n ? p : period - p);
	}
	case BorderMode::wrap: {
		int p = ((idx % n) + n) % n;
		return static_cast<std::size_t>(p);
	}
	}
	return 0;
}

// Fetch a pixel with border handling.
// For zero/constant modes, returns the pad value when out of bounds.
template <DspScalar T>
T fetch_pixel(const mtl::mat::dense2D<T>& img, int r, int c,
              BorderMode mode, T pad = T{}) {
	int rows = static_cast<int>(img.num_rows());
	int cols = static_cast<int>(img.num_cols());

	if (r >= 0 && r < rows && c >= 0 && c < cols) {
		return img(static_cast<std::size_t>(r), static_cast<std::size_t>(c));
	}

	if (mode == BorderMode::zero) return T{};
	if (mode == BorderMode::constant) return pad;

	std::size_t ri = border_index(r, static_cast<std::size_t>(rows), mode);
	std::size_t ci = border_index(c, static_cast<std::size_t>(cols), mode);
	return img(ri, ci);
}

// Planar multi-channel image container.
//
// Stores Channels separate dense2D<T> matrices (one per channel).
// All core 2D algorithms operate on individual planes; use
// apply_per_channel() for multi-channel convenience.
template <DspScalar T, std::size_t Channels = 1>
struct Image {
	std::array<mtl::mat::dense2D<T>, Channels> planes;

	Image() = default;

	Image(std::size_t rows, std::size_t cols)
	{
		for (auto& p : planes) p.change_dim(rows, cols);
	}

	std::size_t rows() const { return planes[0].num_rows(); }
	std::size_t cols() const { return planes[0].num_cols(); }
	static constexpr std::size_t channels() { return Channels; }

	mtl::mat::dense2D<T>&       operator[](std::size_t c)       { return planes[c]; }
	const mtl::mat::dense2D<T>& operator[](std::size_t c) const { return planes[c]; }
};

// Convenience aliases
template <typename T> using GrayImage = Image<T, 1>;
template <typename T> using RGBImage  = Image<T, 3>;
template <typename T> using RGBAImage = Image<T, 4>;

// Apply a function to each plane of a multi-channel image.
// The function signature should be: dense2D<T> func(const dense2D<T>&)
template <DspScalar T, std::size_t C, typename Func>
Image<T, C> apply_per_channel(const Image<T, C>& img, Func&& func) {
	Image<T, C> result;
	for (std::size_t i = 0; i < C; ++i) {
		result[i] = func(img[i]);
	}
	return result;
}

// Convert an RGB image to grayscale using luminance weights.
// Y = 0.299*R + 0.587*G + 0.114*B (ITU-R BT.601)
template <DspField T>
	requires ConvertibleToDouble<T>
mtl::mat::dense2D<T> rgb_to_gray(const Image<T, 3>& rgb) {
	std::size_t rows = rgb.rows();
	std::size_t cols = rgb.cols();
	mtl::mat::dense2D<T> gray(rows, cols);
	T wr = static_cast<T>(0.299);
	T wg = static_cast<T>(0.587);
	T wb = static_cast<T>(0.114);
	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			gray(r, c) = wr * rgb[0](r, c) + wg * rgb[1](r, c) + wb * rgb[2](r, c);
		}
	}
	return gray;
}

} // namespace sw::dsp

// Umbrella: include all image processing sub-headers.
// These depend on the types and utilities defined above.
#include <sw/dsp/image/convolve2d.hpp>
#include <sw/dsp/image/separable.hpp>
#include <sw/dsp/image/morphology.hpp>
#include <sw/dsp/image/edge.hpp>
#include <sw/dsp/image/generators.hpp>
