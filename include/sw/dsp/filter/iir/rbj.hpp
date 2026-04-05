#pragma once
// rbj.hpp: Robert Bristow-Johnson Audio EQ Cookbook biquad filters
//
// Direct coefficient formulas — no analog prototype needed.
// Reference: http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <stdexcept>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/types/biquad_coefficients.hpp>
#include <sw/dsp/filter/biquad/cascade.hpp>
#include <sw/dsp/filter/biquad/state.hpp>

namespace sw::dsp::iir::rbj {

namespace detail {

// Normalize biquad by dividing all coefficients by a0
template <DspField T>
BiquadCoefficients<T> normalize(T b0, T b1, T b2, T a0, T a1, T a2) {
	return BiquadCoefficients<T>(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
}

} // namespace detail

// Each RBJ filter is a single biquad with three-scalar parameterization.
// CoeffScalar is used for coefficient computation; StateScalar and
// SampleScalar are forwarded to the cascade/state for processing.

template <DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class LowPass {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = 1;

	void setup(double sample_rate, double cutoff_freq, double q = 0.7071) {
		CoeffScalar w0 = two_pi_v<CoeffScalar> * static_cast<CoeffScalar>(cutoff_freq / sample_rate);
		CoeffScalar cs = static_cast<CoeffScalar>(std::cos(static_cast<double>(w0)));
		CoeffScalar sn = static_cast<CoeffScalar>(std::sin(static_cast<double>(w0)));
		CoeffScalar alpha = sn / (CoeffScalar{2} * static_cast<CoeffScalar>(q));

		CoeffScalar b0 = (CoeffScalar{1} - cs) / CoeffScalar{2};
		CoeffScalar b1 =  CoeffScalar{1} - cs;
		CoeffScalar b2 = (CoeffScalar{1} - cs) / CoeffScalar{2};
		CoeffScalar a0 =  CoeffScalar{1} + alpha;
		CoeffScalar a1 =  CoeffScalar{-2} * cs;
		CoeffScalar a2 =  CoeffScalar{1} - alpha;

		cascade_.set_num_stages(1);
		cascade_.stage(0) = detail::normalize(b0, b1, b2, a0, a1, a2);
	}

	const Cascade<CoeffScalar, 1>& cascade() const { return cascade_; }

private:
	Cascade<CoeffScalar, 1> cascade_{};
};

template <DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class HighPass {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = 1;

	void setup(double sample_rate, double cutoff_freq, double q = 0.7071) {
		CoeffScalar w0 = two_pi_v<CoeffScalar> * static_cast<CoeffScalar>(cutoff_freq / sample_rate);
		CoeffScalar cs = static_cast<CoeffScalar>(std::cos(static_cast<double>(w0)));
		CoeffScalar sn = static_cast<CoeffScalar>(std::sin(static_cast<double>(w0)));
		CoeffScalar alpha = sn / (CoeffScalar{2} * static_cast<CoeffScalar>(q));

		CoeffScalar b0 =  (CoeffScalar{1} + cs) / CoeffScalar{2};
		CoeffScalar b1 = -(CoeffScalar{1} + cs);
		CoeffScalar b2 =  (CoeffScalar{1} + cs) / CoeffScalar{2};
		CoeffScalar a0 =  CoeffScalar{1} + alpha;
		CoeffScalar a1 =  CoeffScalar{-2} * cs;
		CoeffScalar a2 =  CoeffScalar{1} - alpha;

		cascade_.set_num_stages(1);
		cascade_.stage(0) = detail::normalize(b0, b1, b2, a0, a1, a2);
	}

	const Cascade<CoeffScalar, 1>& cascade() const { return cascade_; }

private:
	Cascade<CoeffScalar, 1> cascade_{};
};

template <DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class BandPass {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = 1;

	void setup(double sample_rate, double center_freq, double bandwidth = 1.0) {
		CoeffScalar w0 = two_pi_v<CoeffScalar> * static_cast<CoeffScalar>(center_freq / sample_rate);
		CoeffScalar cs = static_cast<CoeffScalar>(std::cos(static_cast<double>(w0)));
		CoeffScalar sn = static_cast<CoeffScalar>(std::sin(static_cast<double>(w0)));
		CoeffScalar alpha = sn / (CoeffScalar{2} * static_cast<CoeffScalar>(bandwidth));

		CoeffScalar b0 =  alpha;
		CoeffScalar b1 =  CoeffScalar{};
		CoeffScalar b2 = -alpha;
		CoeffScalar a0 =  CoeffScalar{1} + alpha;
		CoeffScalar a1 =  CoeffScalar{-2} * cs;
		CoeffScalar a2 =  CoeffScalar{1} - alpha;

		cascade_.set_num_stages(1);
		cascade_.stage(0) = detail::normalize(b0, b1, b2, a0, a1, a2);
	}

	const Cascade<CoeffScalar, 1>& cascade() const { return cascade_; }

private:
	Cascade<CoeffScalar, 1> cascade_{};
};

template <DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class BandStop {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = 1;

	void setup(double sample_rate, double center_freq, double bandwidth = 1.0) {
		CoeffScalar w0 = two_pi_v<CoeffScalar> * static_cast<CoeffScalar>(center_freq / sample_rate);
		CoeffScalar cs = static_cast<CoeffScalar>(std::cos(static_cast<double>(w0)));
		CoeffScalar sn = static_cast<CoeffScalar>(std::sin(static_cast<double>(w0)));
		CoeffScalar alpha = sn / (CoeffScalar{2} * static_cast<CoeffScalar>(bandwidth));

		CoeffScalar b0 =  CoeffScalar{1};
		CoeffScalar b1 =  CoeffScalar{-2} * cs;
		CoeffScalar b2 =  CoeffScalar{1};
		CoeffScalar a0 =  CoeffScalar{1} + alpha;
		CoeffScalar a1 =  CoeffScalar{-2} * cs;
		CoeffScalar a2 =  CoeffScalar{1} - alpha;

		cascade_.set_num_stages(1);
		cascade_.stage(0) = detail::normalize(b0, b1, b2, a0, a1, a2);
	}

	const Cascade<CoeffScalar, 1>& cascade() const { return cascade_; }

private:
	Cascade<CoeffScalar, 1> cascade_{};
};

template <DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class AllPass {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = 1;

	void setup(double sample_rate, double center_freq, double q = 0.7071) {
		CoeffScalar w0 = two_pi_v<CoeffScalar> * static_cast<CoeffScalar>(center_freq / sample_rate);
		CoeffScalar cs = static_cast<CoeffScalar>(std::cos(static_cast<double>(w0)));
		CoeffScalar sn = static_cast<CoeffScalar>(std::sin(static_cast<double>(w0)));
		CoeffScalar alpha = sn / (CoeffScalar{2} * static_cast<CoeffScalar>(q));

		CoeffScalar b0 =  CoeffScalar{1} - alpha;
		CoeffScalar b1 =  CoeffScalar{-2} * cs;
		CoeffScalar b2 =  CoeffScalar{1} + alpha;
		CoeffScalar a0 =  CoeffScalar{1} + alpha;
		CoeffScalar a1 =  CoeffScalar{-2} * cs;
		CoeffScalar a2 =  CoeffScalar{1} - alpha;

		cascade_.set_num_stages(1);
		cascade_.stage(0) = detail::normalize(b0, b1, b2, a0, a1, a2);
	}

	const Cascade<CoeffScalar, 1>& cascade() const { return cascade_; }

private:
	Cascade<CoeffScalar, 1> cascade_{};
};

template <DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class LowShelf {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = 1;

	void setup(double sample_rate, double cutoff_freq, double gain_db, double slope = 1.0) {
		if (!(slope > 0.0)) throw std::invalid_argument("rbj::LowShelf: slope must be > 0");
		double A  = std::pow(10.0, gain_db / 40.0);
		CoeffScalar w0 = two_pi_v<CoeffScalar> * static_cast<CoeffScalar>(cutoff_freq / sample_rate);
		double cs = std::cos(static_cast<double>(w0));
		double sn = std::sin(static_cast<double>(w0));
		double alpha = sn / 2.0 * std::sqrt((A + 1.0/A) * (1.0/slope - 1.0) + 2.0);
		double sq = 2.0 * std::sqrt(A) * alpha;

		auto C = [](double v) { return static_cast<CoeffScalar>(v); };

		CoeffScalar b0 = C(A * ((A+1) - (A-1)*cs + sq));
		CoeffScalar b1 = C(2*A * ((A-1) - (A+1)*cs));
		CoeffScalar b2 = C(A * ((A+1) - (A-1)*cs - sq));
		CoeffScalar a0 = C((A+1) + (A-1)*cs + sq);
		CoeffScalar a1 = C(-2 * ((A-1) + (A+1)*cs));
		CoeffScalar a2 = C((A+1) + (A-1)*cs - sq);

		cascade_.set_num_stages(1);
		cascade_.stage(0) = detail::normalize(b0, b1, b2, a0, a1, a2);
	}

	const Cascade<CoeffScalar, 1>& cascade() const { return cascade_; }

private:
	Cascade<CoeffScalar, 1> cascade_{};
};

template <DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class HighShelf {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	static constexpr int max_stages = 1;

	void setup(double sample_rate, double cutoff_freq, double gain_db, double slope = 1.0) {
		if (!(slope > 0.0)) throw std::invalid_argument("rbj::HighShelf: slope must be > 0");
		double A  = std::pow(10.0, gain_db / 40.0);
		CoeffScalar w0 = two_pi_v<CoeffScalar> * static_cast<CoeffScalar>(cutoff_freq / sample_rate);
		double cs = std::cos(static_cast<double>(w0));
		double sn = std::sin(static_cast<double>(w0));
		double alpha = sn / 2.0 * std::sqrt((A + 1.0/A) * (1.0/slope - 1.0) + 2.0);
		double sq = 2.0 * std::sqrt(A) * alpha;

		auto C = [](double v) { return static_cast<CoeffScalar>(v); };

		CoeffScalar b0 = C(A * ((A+1) + (A-1)*cs + sq));
		CoeffScalar b1 = C(-2*A * ((A-1) + (A+1)*cs));
		CoeffScalar b2 = C(A * ((A+1) + (A-1)*cs - sq));
		CoeffScalar a0 = C((A+1) - (A-1)*cs + sq);
		CoeffScalar a1 = C(2 * ((A-1) - (A+1)*cs));
		CoeffScalar a2 = C((A+1) - (A-1)*cs - sq);

		cascade_.set_num_stages(1);
		cascade_.stage(0) = detail::normalize(b0, b1, b2, a0, a1, a2);
	}

	const Cascade<CoeffScalar, 1>& cascade() const { return cascade_; }

private:
	Cascade<CoeffScalar, 1> cascade_{};
};

} // namespace sw::dsp::iir::rbj
