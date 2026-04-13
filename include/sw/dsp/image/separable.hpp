#pragma once
// separable.hpp: separable 2D filtering and kernel factories
//
// A separable kernel can be decomposed into a row filter and a column
// filter, reducing complexity from O(k^2) to O(k) per pixel.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/image/image.hpp>

namespace sw::dsp {

// Apply a separable filter (row kernel then column kernel).
//
// This is equivalent to convolve2d(image, outer_product(col_kernel, row_kernel))
// but runs in O(krow + kcol) per pixel instead of O(krow * kcol).
template <DspField T, DspField K>
mtl::mat::dense2D<T> separable_filter(const mtl::mat::dense2D<T>& image,
                                      const mtl::vec::dense_vector<K>& row_kernel,
                                      const mtl::vec::dense_vector<K>& col_kernel,
                                      BorderMode border = BorderMode::reflect_101,
                                      T pad = T{}) {
	std::size_t rows = image.num_rows();
	std::size_t cols = image.num_cols();
	int rk = static_cast<int>(row_kernel.size() / 2);
	int ck = static_cast<int>(col_kernel.size() / 2);

	// Pass 1: filter along rows
	mtl::mat::dense2D<T> temp(rows, cols);
	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			T sum{};
			for (std::size_t k = 0; k < row_kernel.size(); ++k) {
				int ic = static_cast<int>(c) + static_cast<int>(k) - rk;
				T pixel = fetch_pixel(image, static_cast<int>(r), ic, border, pad);
				sum = sum + static_cast<T>(row_kernel[k]) * pixel;
			}
			temp(r, c) = sum;
		}
	}

	// Pass 2: filter along columns
	mtl::mat::dense2D<T> result(rows, cols);
	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			T sum{};
			for (std::size_t k = 0; k < col_kernel.size(); ++k) {
				int ir = static_cast<int>(r) + static_cast<int>(k) - ck;
				T pixel = fetch_pixel(temp, ir, static_cast<int>(c), border, pad);
				sum = sum + static_cast<T>(col_kernel[k]) * pixel;
			}
			result(r, c) = sum;
		}
	}
	return result;
}

// Create a 1D Gaussian kernel with the given sigma and radius.
// The kernel size is 2*radius+1.
template <DspField T>
mtl::vec::dense_vector<T> gaussian_kernel_1d(double sigma, std::size_t radius) {
	std::size_t size = 2 * radius + 1;
	mtl::vec::dense_vector<T> kernel(size);
	double sum = 0.0;
	for (std::size_t i = 0; i < size; ++i) {
		double x = static_cast<double>(i) - static_cast<double>(radius);
		double val = std::exp(-0.5 * x * x / (sigma * sigma));
		kernel[i] = static_cast<T>(val);
		sum += val;
	}
	// Normalize
	for (std::size_t i = 0; i < size; ++i) {
		kernel[i] = static_cast<T>(static_cast<double>(kernel[i]) / sum);
	}
	return kernel;
}

// Create a 1D box (uniform) kernel of the given size.
template <DspField T>
mtl::vec::dense_vector<T> box_kernel_1d(std::size_t size) {
	mtl::vec::dense_vector<T> kernel(size);
	T val = static_cast<T>(1.0 / static_cast<double>(size));
	for (std::size_t i = 0; i < size; ++i) {
		kernel[i] = val;
	}
	return kernel;
}

// Gaussian blur using separable decomposition.
template <DspField T>
mtl::mat::dense2D<T> gaussian_blur(const mtl::mat::dense2D<T>& image,
                                   double sigma,
                                   std::size_t radius = 0,
                                   BorderMode border = BorderMode::reflect_101) {
	if (radius == 0) {
		radius = static_cast<std::size_t>(std::ceil(3.0 * sigma));
		if (radius < 1) radius = 1;
	}
	auto kernel = gaussian_kernel_1d<T>(sigma, radius);
	return separable_filter(image, kernel, kernel, border);
}

// Box blur using separable decomposition.
template <DspField T>
mtl::mat::dense2D<T> box_blur(const mtl::mat::dense2D<T>& image,
                               std::size_t size,
                               BorderMode border = BorderMode::reflect_101) {
	auto kernel = box_kernel_1d<T>(size);
	return separable_filter(image, kernel, kernel, border);
}

} // namespace sw::dsp
