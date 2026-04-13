#pragma once
// morphology.hpp: mathematical morphology operations
//
// Dilation, erosion, and compound operations (open, close, gradient,
// tophat, blackhat) using structuring elements.
//
// Requires DspOrderedField (min/max comparisons).
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <mtl/mat/dense2D.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/image/image.hpp>

namespace sw::dsp {

// ---------- Structuring element factories ----------

// Rectangular structuring element (all true).
inline mtl::mat::dense2D<bool> make_rect_element(std::size_t rows, std::size_t cols) {
	mtl::mat::dense2D<bool> elem(rows, cols);
	for (std::size_t r = 0; r < rows; ++r)
		for (std::size_t c = 0; c < cols; ++c)
			elem(r, c) = true;
	return elem;
}

// Square structuring element.
inline mtl::mat::dense2D<bool> make_rect_element(std::size_t size) {
	return make_rect_element(size, size);
}

// Cross-shaped structuring element.
inline mtl::mat::dense2D<bool> make_cross_element(std::size_t size) {
	mtl::mat::dense2D<bool> elem(size, size);
	std::size_t center = size / 2;
	for (std::size_t r = 0; r < size; ++r)
		for (std::size_t c = 0; c < size; ++c)
			elem(r, c) = (r == center || c == center);
	return elem;
}

// Elliptical (disk-like) structuring element.
inline mtl::mat::dense2D<bool> make_ellipse_element(std::size_t size) {
	mtl::mat::dense2D<bool> elem(size, size);
	double center = static_cast<double>(size - 1) * 0.5;
	double rx = center + 0.5;
	double ry = center + 0.5;
	for (std::size_t r = 0; r < size; ++r) {
		for (std::size_t c = 0; c < size; ++c) {
			double dr = (static_cast<double>(r) - center) / ry;
			double dc = (static_cast<double>(c) - center) / rx;
			elem(r, c) = (dr * dr + dc * dc <= 1.0);
		}
	}
	return elem;
}

// ---------- Basic morphological operations ----------

// Dilation: each output pixel is the maximum over the structuring element neighborhood.
template <DspOrderedField T>
mtl::mat::dense2D<T> dilate(const mtl::mat::dense2D<T>& image,
                            const mtl::mat::dense2D<bool>& element) {
	std::size_t rows = image.num_rows();
	std::size_t cols = image.num_cols();
	std::size_t erows = element.num_rows();
	std::size_t ecols = element.num_cols();
	int er = static_cast<int>(erows / 2);
	int ec = static_cast<int>(ecols / 2);

	mtl::mat::dense2D<T> result(rows, cols);

	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			T max_val = std::numeric_limits<T>::lowest();
			for (std::size_t ei = 0; ei < erows; ++ei) {
				for (std::size_t ej = 0; ej < ecols; ++ej) {
					if (!element(ei, ej)) continue;
					int ir = static_cast<int>(r) + static_cast<int>(ei) - er;
					int ic = static_cast<int>(c) + static_cast<int>(ej) - ec;
					T pixel = fetch_pixel(image, ir, ic,
					                      BorderMode::replicate);
					if (pixel > max_val) max_val = pixel;
				}
			}
			result(r, c) = max_val;
		}
	}
	return result;
}

// Erosion: each output pixel is the minimum over the structuring element neighborhood.
template <DspOrderedField T>
mtl::mat::dense2D<T> erode(const mtl::mat::dense2D<T>& image,
                           const mtl::mat::dense2D<bool>& element) {
	std::size_t rows = image.num_rows();
	std::size_t cols = image.num_cols();
	std::size_t erows = element.num_rows();
	std::size_t ecols = element.num_cols();
	int er = static_cast<int>(erows / 2);
	int ec = static_cast<int>(ecols / 2);

	mtl::mat::dense2D<T> result(rows, cols);

	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			T min_val = std::numeric_limits<T>::max();
			for (std::size_t ei = 0; ei < erows; ++ei) {
				for (std::size_t ej = 0; ej < ecols; ++ej) {
					if (!element(ei, ej)) continue;
					int ir = static_cast<int>(r) + static_cast<int>(ei) - er;
					int ic = static_cast<int>(c) + static_cast<int>(ej) - ec;
					T pixel = fetch_pixel(image, ir, ic,
					                      BorderMode::replicate);
					if (pixel < min_val) min_val = pixel;
				}
			}
			result(r, c) = min_val;
		}
	}
	return result;
}

// ---------- Compound morphological operations ----------

// Opening: erosion followed by dilation.
// Removes small bright regions while preserving shape.
template <DspOrderedField T>
mtl::mat::dense2D<T> morphological_open(const mtl::mat::dense2D<T>& image,
                                        const mtl::mat::dense2D<bool>& element) {
	return dilate(erode(image, element), element);
}

// Closing: dilation followed by erosion.
// Fills small dark regions while preserving shape.
template <DspOrderedField T>
mtl::mat::dense2D<T> morphological_close(const mtl::mat::dense2D<T>& image,
                                         const mtl::mat::dense2D<bool>& element) {
	return erode(dilate(image, element), element);
}

// Morphological gradient: dilation minus erosion.
// Highlights edges/boundaries.
template <DspOrderedField T>
mtl::mat::dense2D<T> morphological_gradient(const mtl::mat::dense2D<T>& image,
                                            const mtl::mat::dense2D<bool>& element) {
	auto d = dilate(image, element);
	auto e = erode(image, element);
	std::size_t rows = image.num_rows();
	std::size_t cols = image.num_cols();
	mtl::mat::dense2D<T> result(rows, cols);
	for (std::size_t r = 0; r < rows; ++r)
		for (std::size_t c = 0; c < cols; ++c)
			result(r, c) = d(r, c) - e(r, c);
	return result;
}

// Top-hat: image minus opening.
// Extracts small bright features.
template <DspOrderedField T>
mtl::mat::dense2D<T> tophat(const mtl::mat::dense2D<T>& image,
                            const mtl::mat::dense2D<bool>& element) {
	auto opened = morphological_open(image, element);
	std::size_t rows = image.num_rows();
	std::size_t cols = image.num_cols();
	mtl::mat::dense2D<T> result(rows, cols);
	for (std::size_t r = 0; r < rows; ++r)
		for (std::size_t c = 0; c < cols; ++c)
			result(r, c) = image(r, c) - opened(r, c);
	return result;
}

// Black-hat: closing minus image.
// Extracts small dark features.
template <DspOrderedField T>
mtl::mat::dense2D<T> blackhat(const mtl::mat::dense2D<T>& image,
                              const mtl::mat::dense2D<bool>& element) {
	auto closed = morphological_close(image, element);
	std::size_t rows = image.num_rows();
	std::size_t cols = image.num_cols();
	mtl::mat::dense2D<T> result(rows, cols);
	for (std::size_t r = 0; r < rows; ++r)
		for (std::size_t c = 0; c < cols; ++c)
			result(r, c) = closed(r, c) - image(r, c);
	return result;
}

} // namespace sw::dsp
