#pragma once
// noise_shaping.hpp: error-feedback noise shaping
//
// Feeds the quantization error back through a filter to reshape
// the noise spectrum, pushing noise energy out of the band of
// interest.
//
// NTF designs:
//   1st order: (1 - z⁻¹)        → -e[n-1]
//   2nd order: (1 - z⁻¹)²       → -2e[n-1] + e[n-2]
//   3rd order: (1 - z⁻¹)³       → -3e[n-1] + 3e[n-2] - e[n-3]
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/denormal.hpp>

namespace sw::dsp {

// First-order error-feedback noise shaper.
// NTF = (1 - z⁻¹)
template <DspField HighPrecT, DspField LowPrecT>
class FirstOrderNoiseShaper {
public:
	LowPrecT process(HighPrecT input) {
		HighPrecT shaped = input + error_feedback_;
		LowPrecT quantized = static_cast<LowPrecT>(shaped);
		HighPrecT reconstructed = static_cast<HighPrecT>(quantized);
		HighPrecT error = shaped - reconstructed;
		error_feedback_ = error + denormal_.ac();
		return quantized;
	}

	mtl::vec::dense_vector<LowPrecT> process(const mtl::vec::dense_vector<HighPrecT>& input) {
		mtl::vec::dense_vector<LowPrecT> output(input.size());
		for (std::size_t i = 0; i < input.size(); ++i) {
			output[i] = process(input[i]);
		}
		return output;
	}

	void reset() { error_feedback_ = HighPrecT{}; }

private:
	HighPrecT error_feedback_{};
	DenormalPrevention<HighPrecT> denormal_;
};

// Second-order error-feedback noise shaper.
// NTF = (1 - z⁻¹)²
// Feedback: -2·e[n-1] + e[n-2]
template <DspField HighPrecT, DspField LowPrecT>
class SecondOrderNoiseShaper {
public:
	LowPrecT process(HighPrecT input) {
		HighPrecT feedback = HighPrecT(2) * e1_ - e2_;
		HighPrecT shaped = input + feedback;
		LowPrecT quantized = static_cast<LowPrecT>(shaped);
		HighPrecT reconstructed = static_cast<HighPrecT>(quantized);
		HighPrecT error = shaped - reconstructed;
		e2_ = e1_;
		e1_ = error + denormal_.ac();
		return quantized;
	}

	mtl::vec::dense_vector<LowPrecT> process(const mtl::vec::dense_vector<HighPrecT>& input) {
		mtl::vec::dense_vector<LowPrecT> output(input.size());
		for (std::size_t i = 0; i < input.size(); ++i) {
			output[i] = process(input[i]);
		}
		return output;
	}

	void reset() { e1_ = HighPrecT{}; e2_ = HighPrecT{}; }

private:
	HighPrecT e1_{};
	HighPrecT e2_{};
	DenormalPrevention<HighPrecT> denormal_;
};

// Third-order error-feedback noise shaper.
// NTF = (1 - z⁻¹)³
// Feedback: -3·e[n-1] + 3·e[n-2] - e[n-3]
template <DspField HighPrecT, DspField LowPrecT>
class ThirdOrderNoiseShaper {
public:
	LowPrecT process(HighPrecT input) {
		HighPrecT feedback = HighPrecT(3) * e1_
		                   - HighPrecT(3) * e2_ + e3_;
		HighPrecT shaped = input + feedback;
		LowPrecT quantized = static_cast<LowPrecT>(shaped);
		HighPrecT reconstructed = static_cast<HighPrecT>(quantized);
		HighPrecT error = shaped - reconstructed;
		e3_ = e2_;
		e2_ = e1_;
		e1_ = error + denormal_.ac();
		return quantized;
	}

	mtl::vec::dense_vector<LowPrecT> process(const mtl::vec::dense_vector<HighPrecT>& input) {
		mtl::vec::dense_vector<LowPrecT> output(input.size());
		for (std::size_t i = 0; i < input.size(); ++i) {
			output[i] = process(input[i]);
		}
		return output;
	}

	void reset() { e1_ = HighPrecT{}; e2_ = HighPrecT{}; e3_ = HighPrecT{}; }

private:
	HighPrecT e1_{};
	HighPrecT e2_{};
	HighPrecT e3_{};
	DenormalPrevention<HighPrecT> denormal_;
};

} // namespace sw::dsp
