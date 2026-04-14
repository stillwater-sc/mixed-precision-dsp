#pragma once
// projection.hpp: type projection and embedding for mixed-precision workflows
//
// project_onto<Target>(source): convert from a wider type to a narrower type.
//   This is a lossy operation — precision is reduced. The mathematical
//   analogy is projecting a vector from a high-dimensional space onto
//   a lower-dimensional subspace.
//
// embed_into<Target>(source): convert from a narrower type to a wider type.
//   This is a lossless operation — the value is exactly representable
//   in the target type. The mathematical analogy is embedding a
//   lower-dimensional object into a higher-dimensional space.
//
// These operations are fundamental to mixed-precision workflows:
//   1. Design filter coefficients in double (high precision)
//   2. project_onto<fixpnt<16,14>>() for FPGA deployment
//   3. Verify quality loss with pole_displacement()
//   4. embed_into<double>() for analysis comparison
//
// Supported types:
//   - BiquadCoefficients<T>
//   - Cascade<T, MaxStages>
//   - mtl::vec::dense_vector<T>
//   - mtl::mat::dense2D<T>
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <limits>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/types/biquad_coefficients.hpp>
#include <sw/dsp/filter/biquad/cascade.hpp>

namespace sw::dsp {

// Compile-time directional constraints using std::numeric_limits::digits.
// ProjectableOnto: Target has fewer (or equal) significant digits than Source.
// EmbeddableInto:  Target has more (or equal) significant digits than Source.
template <typename Target, typename Source>
concept ProjectableOnto =
	std::numeric_limits<Target>::digits <= std::numeric_limits<Source>::digits;

template <typename Target, typename Source>
concept EmbeddableInto =
	std::numeric_limits<Target>::digits >= std::numeric_limits<Source>::digits;

// ============================================================================
// project_onto<Target>: wider → narrower (lossy)
// ============================================================================

// Project biquad coefficients onto a narrower type.
template <DspField Target, DspField Source>
	requires ProjectableOnto<Target, Source>
BiquadCoefficients<Target> project_onto(const BiquadCoefficients<Source>& src) {
	return BiquadCoefficients<Target>(
		static_cast<Target>(src.b0),
		static_cast<Target>(src.b1),
		static_cast<Target>(src.b2),
		static_cast<Target>(src.a1),
		static_cast<Target>(src.a2)
	);
}

// Project a cascade onto a narrower coefficient type.
template <DspField Target, DspField Source, int MaxStages>
	requires ProjectableOnto<Target, Source>
Cascade<Target, MaxStages> project_onto(const Cascade<Source, MaxStages>& src) {
	Cascade<Target, MaxStages> dst;
	dst.set_num_stages(src.num_stages());
	for (int i = 0; i < src.num_stages(); ++i) {
		dst.stage(i) = project_onto<Target>(src.stage(i));
	}
	return dst;
}

// Project a dense vector onto a narrower element type.
template <DspField Target, DspField Source>
	requires ProjectableOnto<Target, Source>
mtl::vec::dense_vector<Target> project_onto(const mtl::vec::dense_vector<Source>& src) {
	mtl::vec::dense_vector<Target> dst(src.size());
	for (std::size_t i = 0; i < src.size(); ++i) {
		dst[i] = static_cast<Target>(src[i]);
	}
	return dst;
}

// Project a dense 2D matrix onto a narrower element type.
template <DspField Target, DspField Source>
	requires ProjectableOnto<Target, Source>
mtl::mat::dense2D<Target> project_onto(const mtl::mat::dense2D<Source>& src) {
	mtl::mat::dense2D<Target> dst(src.num_rows(), src.num_cols());
	for (std::size_t r = 0; r < src.num_rows(); ++r) {
		for (std::size_t c = 0; c < src.num_cols(); ++c) {
			dst(r, c) = static_cast<Target>(src(r, c));
		}
	}
	return dst;
}

// ============================================================================
// embed_into<Target>: narrower → wider (lossless)
// ============================================================================

// Embed biquad coefficients into a wider type.
template <DspField Target, DspField Source>
	requires EmbeddableInto<Target, Source>
BiquadCoefficients<Target> embed_into(const BiquadCoefficients<Source>& src) {
	return BiquadCoefficients<Target>(
		static_cast<Target>(src.b0),
		static_cast<Target>(src.b1),
		static_cast<Target>(src.b2),
		static_cast<Target>(src.a1),
		static_cast<Target>(src.a2)
	);
}

// Embed a cascade into a wider coefficient type.
template <DspField Target, DspField Source, int MaxStages>
	requires EmbeddableInto<Target, Source>
Cascade<Target, MaxStages> embed_into(const Cascade<Source, MaxStages>& src) {
	Cascade<Target, MaxStages> dst;
	dst.set_num_stages(src.num_stages());
	for (int i = 0; i < src.num_stages(); ++i) {
		dst.stage(i) = embed_into<Target>(src.stage(i));
	}
	return dst;
}

// Embed a dense vector into a wider element type.
template <DspField Target, DspField Source>
	requires EmbeddableInto<Target, Source>
mtl::vec::dense_vector<Target> embed_into(const mtl::vec::dense_vector<Source>& src) {
	mtl::vec::dense_vector<Target> dst(src.size());
	for (std::size_t i = 0; i < src.size(); ++i) {
		dst[i] = static_cast<Target>(src[i]);
	}
	return dst;
}

// Embed a dense 2D matrix into a wider element type.
template <DspField Target, DspField Source>
	requires EmbeddableInto<Target, Source>
mtl::mat::dense2D<Target> embed_into(const mtl::mat::dense2D<Source>& src) {
	mtl::mat::dense2D<Target> dst(src.num_rows(), src.num_cols());
	for (std::size_t r = 0; r < src.num_rows(); ++r) {
		for (std::size_t c = 0; c < src.num_cols(); ++c) {
			dst(r, c) = static_cast<Target>(src(r, c));
		}
	}
	return dst;
}

} // namespace sw::dsp
