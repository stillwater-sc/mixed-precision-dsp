#pragma once
// convolve2d.hpp: 2D spatial convolution
//
// Applies a 2D kernel to a single-channel image with configurable
// border handling. The kernel type K can differ from the image type T
// to support mixed-precision convolution.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <mtl/mat/dense2D.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/image/image.hpp>

namespace sw::dsp {

// 2D spatial convolution.
//
// Convolves image with kernel using the specified border mode.
// The kernel is applied in correlation order (no flip); pass a
// pre-flipped kernel if true convolution is needed.
//
// Template parameters:
//   T: image pixel type
//   K: kernel coefficient type (may differ for mixed precision)
template <DspField T, DspField K>
mtl::mat::dense2D<T> convolve2d(const mtl::mat::dense2D<T>& image,
                                const mtl::mat::dense2D<K>& kernel,
                                BorderMode border = BorderMode::reflect_101,
                                T pad = T{}) {
	std::size_t rows = image.num_rows();
	std::size_t cols = image.num_cols();
	std::size_t krows = kernel.num_rows();
	std::size_t kcols = kernel.num_cols();
	int kr = static_cast<int>(krows / 2);
	int kc = static_cast<int>(kcols / 2);

	mtl::mat::dense2D<T> result(rows, cols);

	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			T sum{};
			for (std::size_t ki = 0; ki < krows; ++ki) {
				for (std::size_t kj = 0; kj < kcols; ++kj) {
					int ir = static_cast<int>(r) + static_cast<int>(ki) - kr;
					int ic = static_cast<int>(c) + static_cast<int>(kj) - kc;
					T pixel = fetch_pixel(image, ir, ic, border, pad);
					sum = sum + static_cast<T>(kernel(ki, kj)) * pixel;
				}
			}
			result(r, c) = sum;
		}
	}
	return result;
}

// Convenience: convolve all channels of a multi-channel image.
template <DspField T, std::size_t C, DspField K>
Image<T, C> convolve2d(const Image<T, C>& img,
                       const mtl::mat::dense2D<K>& kernel,
                       BorderMode border = BorderMode::reflect_101,
                       T pad = T{}) {
	return apply_per_channel(img, [&](const mtl::mat::dense2D<T>& plane) {
		return convolve2d(plane, kernel, border, pad);
	});
}

} // namespace sw::dsp
