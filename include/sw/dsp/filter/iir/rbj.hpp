#pragma once
// rbj.hpp: Robert Bristow-Johnson Audio EQ Cookbook biquad filters
//
// Direct coefficient formulas — no analog prototype needed.
// Reference: http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt
//
// The public setup() APIs take `double` parameters (sample_rate, freq, Q,
// gain_db, slope) for interface stability, but all intermediate
// coefficient math runs in CoeffScalar. A posit or fixed-point CoeffScalar
// therefore produces coefficients that were computed end-to-end at the
// caller's declared precision — required for embedded mixed-precision
// deployments where filter design executes on the target.
//
// ADL-friendly trig (using std::cos; cos(x);) picks up sw::universal::cos
// for Universal number types and std::cos for native floats.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
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

inline void validate_freq(double sample_rate, double freq, const char* name) {
	if (!(sample_rate > 0.0) || !std::isfinite(sample_rate))
		throw std::invalid_argument(std::string(name) + ": sample_rate must be > 0 and finite");
	if (!(freq > 0.0) || !std::isfinite(freq))
		throw std::invalid_argument(std::string(name) + ": frequency must be > 0 and finite");
	if (freq >= sample_rate * 0.5)
		throw std::invalid_argument(std::string(name) + ": frequency must be < Nyquist (sample_rate/2)");
}

inline void validate_q(double q, const char* name) {
	if (!(q > 0.0) || !std::isfinite(q))
		throw std::invalid_argument(std::string(name) + ": Q/bandwidth must be > 0 and finite");
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
		using std::cos; using std::sin;
		detail::validate_freq(sample_rate, cutoff_freq, "rbj::LowPass");
		detail::validate_q(q, "rbj::LowPass");

		constexpr CoeffScalar one = CoeffScalar(1);
		constexpr CoeffScalar two = CoeffScalar(2);
		const CoeffScalar w0 = CoeffScalar(two_pi)
		                     * CoeffScalar(cutoff_freq) / CoeffScalar(sample_rate);
		const CoeffScalar cs = cos(w0);
		const CoeffScalar sn = sin(w0);
		const CoeffScalar alpha = sn / (two * CoeffScalar(q));

		CoeffScalar b0 = (one - cs) / two;
		CoeffScalar b1 =  one - cs;
		CoeffScalar b2 = (one - cs) / two;
		CoeffScalar a0 =  one + alpha;
		CoeffScalar a1 = -(two * cs);
		CoeffScalar a2 =  one - alpha;

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
		using std::cos; using std::sin;
		detail::validate_freq(sample_rate, cutoff_freq, "rbj::HighPass");
		detail::validate_q(q, "rbj::HighPass");

		constexpr CoeffScalar one = CoeffScalar(1);
		constexpr CoeffScalar two = CoeffScalar(2);
		const CoeffScalar w0 = CoeffScalar(two_pi)
		                     * CoeffScalar(cutoff_freq) / CoeffScalar(sample_rate);
		const CoeffScalar cs = cos(w0);
		const CoeffScalar sn = sin(w0);
		const CoeffScalar alpha = sn / (two * CoeffScalar(q));

		CoeffScalar b0 =  (one + cs) / two;
		CoeffScalar b1 = -(one + cs);
		CoeffScalar b2 =  (one + cs) / two;
		CoeffScalar a0 =   one + alpha;
		CoeffScalar a1 = -(two * cs);
		CoeffScalar a2 =   one - alpha;

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
		using std::cos; using std::sin;
		detail::validate_freq(sample_rate, center_freq, "rbj::BandPass");
		detail::validate_q(bandwidth, "rbj::BandPass");

		constexpr CoeffScalar zero = CoeffScalar(0);
		constexpr CoeffScalar one  = CoeffScalar(1);
		constexpr CoeffScalar two  = CoeffScalar(2);
		const CoeffScalar w0 = CoeffScalar(two_pi)
		                     * CoeffScalar(center_freq) / CoeffScalar(sample_rate);
		const CoeffScalar cs = cos(w0);
		const CoeffScalar sn = sin(w0);
		const CoeffScalar alpha = sn / (two * CoeffScalar(bandwidth));

		CoeffScalar b0 =  alpha;
		CoeffScalar b1 =  zero;
		CoeffScalar b2 = -alpha;
		CoeffScalar a0 =  one + alpha;
		CoeffScalar a1 = -(two * cs);
		CoeffScalar a2 =  one - alpha;

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
		using std::cos; using std::sin;
		detail::validate_freq(sample_rate, center_freq, "rbj::BandStop");
		detail::validate_q(bandwidth, "rbj::BandStop");

		constexpr CoeffScalar one = CoeffScalar(1);
		constexpr CoeffScalar two = CoeffScalar(2);
		const CoeffScalar w0 = CoeffScalar(two_pi)
		                     * CoeffScalar(center_freq) / CoeffScalar(sample_rate);
		const CoeffScalar cs = cos(w0);
		const CoeffScalar sn = sin(w0);
		const CoeffScalar alpha = sn / (two * CoeffScalar(bandwidth));

		CoeffScalar b0 =  one;
		CoeffScalar b1 = -(two * cs);
		CoeffScalar b2 =  one;
		CoeffScalar a0 =  one + alpha;
		CoeffScalar a1 = -(two * cs);
		CoeffScalar a2 =  one - alpha;

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
		using std::cos; using std::sin;
		detail::validate_freq(sample_rate, center_freq, "rbj::AllPass");
		detail::validate_q(q, "rbj::AllPass");

		constexpr CoeffScalar one = CoeffScalar(1);
		constexpr CoeffScalar two = CoeffScalar(2);
		const CoeffScalar w0 = CoeffScalar(two_pi)
		                     * CoeffScalar(center_freq) / CoeffScalar(sample_rate);
		const CoeffScalar cs = cos(w0);
		const CoeffScalar sn = sin(w0);
		const CoeffScalar alpha = sn / (two * CoeffScalar(q));

		CoeffScalar b0 =  one - alpha;
		CoeffScalar b1 = -(two * cs);
		CoeffScalar b2 =  one + alpha;
		CoeffScalar a0 =  one + alpha;
		CoeffScalar a1 = -(two * cs);
		CoeffScalar a2 =  one - alpha;

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
		using std::cos; using std::sin; using std::pow; using std::sqrt;
		detail::validate_freq(sample_rate, cutoff_freq, "rbj::LowShelf");
		if (!(slope > 0.0)) throw std::invalid_argument("rbj::LowShelf: slope must be > 0");
		if (!std::isfinite(gain_db)) throw std::invalid_argument("rbj::LowShelf: gain_db must be finite");

		constexpr CoeffScalar zero = CoeffScalar(0);
		constexpr CoeffScalar one  = CoeffScalar(1);
		constexpr CoeffScalar two  = CoeffScalar(2);
		constexpr CoeffScalar ten  = CoeffScalar(10);
		constexpr CoeffScalar forty = CoeffScalar(40);

		const CoeffScalar A = pow(ten, CoeffScalar(gain_db) / forty);
		const CoeffScalar w0 = CoeffScalar(two_pi)
		                     * CoeffScalar(cutoff_freq) / CoeffScalar(sample_rate);
		const CoeffScalar cs = cos(w0);
		const CoeffScalar sn = sin(w0);
		const CoeffScalar inv_A = one / A;
		const CoeffScalar inv_slope = one / CoeffScalar(slope);
		CoeffScalar radicand = (A + inv_A) * (inv_slope - one) + two;
		if (radicand < zero) radicand = zero;
		const CoeffScalar alpha = sn / two * sqrt(radicand);
		const CoeffScalar sq = two * sqrt(A) * alpha;

		const CoeffScalar Ap1 = A + one;
		const CoeffScalar Am1 = A - one;

		CoeffScalar b0 =  A * (Ap1 - Am1*cs + sq);
		CoeffScalar b1 =  two * A * (Am1 - Ap1*cs);
		CoeffScalar b2 =  A * (Ap1 - Am1*cs - sq);
		CoeffScalar a0 =  Ap1 + Am1*cs + sq;
		CoeffScalar a1 = -(two * (Am1 + Ap1*cs));
		CoeffScalar a2 =  Ap1 + Am1*cs - sq;

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
		using std::cos; using std::sin; using std::pow; using std::sqrt;
		detail::validate_freq(sample_rate, cutoff_freq, "rbj::HighShelf");
		if (!(slope > 0.0)) throw std::invalid_argument("rbj::HighShelf: slope must be > 0");
		if (!std::isfinite(gain_db)) throw std::invalid_argument("rbj::HighShelf: gain_db must be finite");

		constexpr CoeffScalar zero = CoeffScalar(0);
		constexpr CoeffScalar one  = CoeffScalar(1);
		constexpr CoeffScalar two  = CoeffScalar(2);
		constexpr CoeffScalar ten  = CoeffScalar(10);
		constexpr CoeffScalar forty = CoeffScalar(40);

		const CoeffScalar A = pow(ten, CoeffScalar(gain_db) / forty);
		const CoeffScalar w0 = CoeffScalar(two_pi)
		                     * CoeffScalar(cutoff_freq) / CoeffScalar(sample_rate);
		const CoeffScalar cs = cos(w0);
		const CoeffScalar sn = sin(w0);
		const CoeffScalar inv_A = one / A;
		const CoeffScalar inv_slope = one / CoeffScalar(slope);
		CoeffScalar radicand = (A + inv_A) * (inv_slope - one) + two;
		if (radicand < zero) radicand = zero;
		const CoeffScalar alpha = sn / two * sqrt(radicand);
		const CoeffScalar sq = two * sqrt(A) * alpha;

		const CoeffScalar Ap1 = A + one;
		const CoeffScalar Am1 = A - one;

		CoeffScalar b0 =  A * (Ap1 + Am1*cs + sq);
		CoeffScalar b1 = -(two * A * (Am1 + Ap1*cs));
		CoeffScalar b2 =  A * (Ap1 + Am1*cs - sq);
		CoeffScalar a0 =  Ap1 - Am1*cs + sq;
		CoeffScalar a1 =  two * (Am1 - Ap1*cs);
		CoeffScalar a2 =  Ap1 - Am1*cs - sq;

		cascade_.set_num_stages(1);
		cascade_.stage(0) = detail::normalize(b0, b1, b2, a0, a1, a2);
	}

	const Cascade<CoeffScalar, 1>& cascade() const { return cascade_; }

private:
	Cascade<CoeffScalar, 1> cascade_{};
};

} // namespace sw::dsp::iir::rbj
