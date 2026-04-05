#pragma once
// state.hpp: biquad state forms for sample processing
//
// Direct Form I, Direct Form II, and Transposed Direct Form II.
// Each is templated on StateScalar for mixed-precision accumulation.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/types/biquad_coefficients.hpp>

namespace sw::dsp {

// Direct Form I
//
// Difference equation:
//   y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
//
// Stores both input and output history. More robust to coefficient
// quantization than Direct Form II, at the cost of more state.
template <DspField StateScalar>
struct DirectFormI {
	StateScalar x1{}, x2{};  // input history:  x[n-1], x[n-2]
	StateScalar y1{}, y2{};  // output history: y[n-1], y[n-2]

	void reset() {
		x1 = StateScalar{};
		x2 = StateScalar{};
		y1 = StateScalar{};
		y2 = StateScalar{};
	}

	template <DspField CoeffScalar, DspScalar SampleScalar>
	SampleScalar process(SampleScalar in, const BiquadCoefficients<CoeffScalar>& c) {
		StateScalar x0 = static_cast<StateScalar>(in);
		StateScalar out = static_cast<StateScalar>(c.b0) * x0
		                + static_cast<StateScalar>(c.b1) * x1
		                + static_cast<StateScalar>(c.b2) * x2
		                - static_cast<StateScalar>(c.a1) * y1
		                - static_cast<StateScalar>(c.a2) * y2;
		x2 = x1;
		x1 = x0;
		y2 = y1;
		y1 = out;
		return static_cast<SampleScalar>(out);
	}
};

// Direct Form II
//
// Difference equation:
//   v[n] = x[n]        - a1*v[n-1] - a2*v[n-2]
//   y[n] = b0*v[n] + b1*v[n-1] + b2*v[n-2]
//
// Minimum state (2 variables). Default realization form.
template <DspField StateScalar>
struct DirectFormII {
	StateScalar v1{}, v2{};  // intermediate state: v[n-1], v[n-2]

	void reset() {
		v1 = StateScalar{};
		v2 = StateScalar{};
	}

	template <DspField CoeffScalar, DspScalar SampleScalar>
	SampleScalar process(SampleScalar in, const BiquadCoefficients<CoeffScalar>& c) {
		StateScalar w = static_cast<StateScalar>(in)
		             - static_cast<StateScalar>(c.a1) * v1
		             - static_cast<StateScalar>(c.a2) * v2;
		StateScalar out = static_cast<StateScalar>(c.b0) * w
		               + static_cast<StateScalar>(c.b1) * v1
		               + static_cast<StateScalar>(c.b2) * v2;
		v2 = v1;
		v1 = w;
		return static_cast<SampleScalar>(out);
	}
};

// Transposed Direct Form II
//
// Difference equation:
//   y[n] = b0*x[n]           + s1[n-1]
//   s1[n] = b1*x[n] - a1*y[n] + s2[n-1]
//   s2[n] = b2*x[n] - a2*y[n]
//
// Better numerical properties for floating-point: each state
// variable accumulates smaller quantities. Preferred for
// high-order or narrow-band filters.
template <DspField StateScalar>
struct TransposedDirectFormII {
	StateScalar s1{}, s2{};

	void reset() {
		s1 = StateScalar{};
		s2 = StateScalar{};
	}

	template <DspField CoeffScalar, DspScalar SampleScalar>
	SampleScalar process(SampleScalar in, const BiquadCoefficients<CoeffScalar>& c) {
		StateScalar x = static_cast<StateScalar>(in);
		StateScalar out = static_cast<StateScalar>(c.b0) * x + s1;
		s1 = static_cast<StateScalar>(c.b1) * x
		   - static_cast<StateScalar>(c.a1) * out + s2;
		s2 = static_cast<StateScalar>(c.b2) * x
		   - static_cast<StateScalar>(c.a2) * out;
		return static_cast<SampleScalar>(out);
	}
};

} // namespace sw::dsp
