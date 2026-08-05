#pragma once
// loop_filter.hpp: the second-order proportional-integral loop filter shared
// by the synchronization loops.
//
// Both timing recovery and carrier recovery need the same thing: turn a
// normalized noise bandwidth and a damping factor into a pair of gains, then
// run a PI filter over a detector's error output. Writing that twice invites
// the two copies to drift, and the parts most worth stating once are the
// parts easiest to get subtly wrong — the gain mapping and the fact that the
// integrator must hold a DEVIATION.
//
// GAIN MAPPING (Rice, "Digital Communications: A Discrete-Time Approach",
// sec. 7.2), for normalized noise bandwidth Bn*T, damping zeta and detector
// slope Kp:
//
//   theta = (Bn*T) / (zeta + 1/(4*zeta))
//   k_p   = 4*zeta*theta / (1 + 2*zeta*theta + theta^2) / Kp
//   k_i   = 4*theta^2    / (1 + 2*zeta*theta + theta^2) / Kp
//
// so callers ask for a bandwidth and a damping rather than for two opaque
// numbers.
//
// THE INTEGRATOR HOLDS A DEVIATION FROM NOMINAL, always starting at zero,
// never an absolute quantity. Its per-step correction is k_i * error, which
// for a narrow loop runs around 1e-4; held against an absolute value of order
// 1 or 2 that increment sits below the ULP of a 16-bit type and the
// integrator simply stops moving. Held as a deviation it works near zero,
// where resolution is finest. Whatever a caller is tracking — samples per
// symbol, radians per sample — it adds the nominal itself.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <stdexcept>
#include <string>

#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp::sdr {

template <DspField T = double>
struct LoopFilterConfig {
	T bandwidth     = T(0.01);   // Bn*T, normalized to the symbol or sample rate
	T damping       = T(0.707);
	T detector_gain = T(1);      // Kp, the detector's slope at its zero crossing
};

// Second-order PI loop filter. Stateless apart from the integrator.
template <DspField T = double>
class PiLoopFilter {
public:
	explicit PiLoopFilter(const LoopFilterConfig<T>& cfg = LoopFilterConfig<T>{}) {
		configure(cfg);
		reset();
	}

	void configure(const LoopFilterConfig<T>& cfg) {
		if (!(cfg.bandwidth > T{}))
			throw std::invalid_argument("PiLoopFilter: bandwidth must be > 0");
		if (!(cfg.damping > T{}))
			throw std::invalid_argument("PiLoopFilter: damping must be > 0");
		if (!(cfg.detector_gain > T{}))
			throw std::invalid_argument("PiLoopFilter: detector_gain must be > 0");
		const T z = cfg.damping;
		const T theta = cfg.bandwidth / (z + T{1} / (T(4) * z));
		const T den = T{1} + T(2) * z * theta + theta * theta;
		k_p_ = (T(4) * z * theta / den) / cfg.detector_gain;
		k_i_ = (T(4) * theta * theta / den) / cfg.detector_gain;
	}

	void reset() { integral_ = T{}; }

	// Fold one error sample in and return the filter output: the integrator
	// state (a deviation from nominal) plus the proportional term.
	T advance(T error) {
		integral_ = integral_ + k_i_ * error;
		return integral_ + k_p_ * error;
	}

	// The integrator on its own — the tracked deviation from nominal.
	T integral() const { return integral_; }
	void set_integral(T v) { integral_ = v; }

	// Clamp the integrator into [-limit, +limit]. Callers apply whatever
	// bound their quantity has; the filter does not invent one.
	void clamp_integral(T limit) {
		if (integral_ < -limit) integral_ = -limit;
		if (integral_ >  limit) integral_ =  limit;
	}

	T proportional_gain() const { return k_p_; }
	T integral_gain() const { return k_i_; }

private:
	T k_p_{}, k_i_{}, integral_{};
};

} // namespace sw::dsp::sdr
