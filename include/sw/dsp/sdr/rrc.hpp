#pragma once
// rrc.hpp: root-raised-cosine and raised-cosine pulse-shaping filter design.
//
// The RRC pulse is the standard shape for a digital link because it splits
// the Nyquist filter evenly between transmitter and receiver: an RRC on each
// end convolves to a full raised cosine, which has zero crossings at every
// non-zero multiple of the symbol period. That gives zero intersymbol
// interference at the sampling instants while keeping the receive filter
// matched to the transmit pulse, which is what maximizes SNR.
//
// These are DESIGN FUNCTIONS — they return taps and hold no state. The
// shaping itself reuses the multirate primitives this library already has:
//
//   TX:  PolyphaseInterpolator<...> shaper(rrc_filter<T>(N, sps, alpha), sps);
//        auto waveform = shaper.process_block(symbols);
//
//   RX:  PolyphaseDecimator<...> matched(rrc_filter<T>(N, sps, alpha), sps);
//        auto [ready, y] = matched.process(sample);
//
// RRC is symmetric, so "convolve with the time-reversed RRC" — the matched
// filter — is the same tap set. No separate reversal step is needed, and
// none is provided; a reversal helper would be an inert copy.
//
// FORMULA. With x = t/T (T the symbol period) and rolloff alpha:
//
//   h_rrc(x) = [ sin(pi*x*(1-a)) + 4*a*x*cos(pi*x*(1+a)) ]
//              / [ pi*x*(1 - (4*a*x)^2) ]
//
// which has two removable singularities that must be handled by their
// limits rather than evaluated:
//
//   x = 0          h = 1 + a*(4/pi - 1)
//   |4*a*x| = 1    h = (a/sqrt(2)) * [ (1+2/pi)*sin(pi/(4a))
//                                    + (1-2/pi)*cos(pi/(4a)) ]
//
// The second exists only for a > 0, and lands on an actual tap whenever
// samples_per_symbol/(4*alpha) is an integer — alpha = 1 with sps = 4, say.
// Evaluating the general form there produces 0/0.
//
// The raised cosine, provided for the composite check and for links that
// shape only at the transmitter:
//
//   h_rc(x) = sinc(x) * cos(pi*a*x) / (1 - (2*a*x)^2)
//   x = 0          h = 1
//   |2*a*x| = 1    h = (pi/4) * sinc(1/(2a))
//
// NORMALIZATION defaults to unit energy (sum h^2 = 1), matching MATLAB's
// rcosdesign. That is the meaningful choice for a matched pair: the
// composite then peaks at exactly 1 on the symbol instant, so the zero-ISI
// property reads directly off the composite samples.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>

namespace sw::dsp::sdr {

// How the returned taps are scaled.
enum class PulseNormalization {
	// sum h^2 = 1. The matched-filter choice: an RRC pair then composites
	// to a peak of exactly 1 at the symbol instant. Matches MATLAB
	// rcosdesign(). This is the default.
	unit_energy,
	// sum h = 1. Passes a DC input through at unity gain, which is what you
	// want when the pulse is used as an interpolation filter rather than as
	// half of a matched pair.
	unit_dc_gain
};

namespace detail {

// How close a value must come to a singularity before the limit is used
// instead of the general form. The singularities sit at exact rational
// positions (x = 0 and |x| = 1/(4a)), so a tap either lands on one or is a
// full grid step away; the threshold only has to absorb the rounding in
// computing x, never to arbitrate a near miss.
inline constexpr double pulse_singularity_tol = 1e-12;

// Normalized sinc, sin(pi*x)/(pi*x), with the removable singularity at 0.
inline double pulse_sinc(double x) {
	if (std::abs(x) < pulse_singularity_tol) return 1.0;
	const double px = pi * x;
	return std::sin(px) / px;
}

// One RRC tap at x = t/T. See the formula block above.
inline double rrc_tap(double x, double alpha) {
	if (std::abs(x) < pulse_singularity_tol)
		return 1.0 + alpha * (4.0 / pi - 1.0);

	if (alpha > 0.0) {
		// The 1 - (4*a*x)^2 denominator vanishes at |x| = 1/(4a).
		const double quarter = 1.0 / (4.0 * alpha);
		if (std::abs(std::abs(x) - quarter) < pulse_singularity_tol) {
			const double s = std::sin(pi * quarter);
			const double c = std::cos(pi * quarter);
			return (alpha / std::sqrt(2.0)) *
			       ((1.0 + 2.0 / pi) * s + (1.0 - 2.0 / pi) * c);
		}
	}

	const double numer = std::sin(pi * x * (1.0 - alpha)) +
	                     4.0 * alpha * x * std::cos(pi * x * (1.0 + alpha));
	const double denom = pi * x * (1.0 - (4.0 * alpha * x) * (4.0 * alpha * x));
	return numer / denom;
}

// One raised-cosine tap at x = t/T.
inline double rc_tap(double x, double alpha) {
	if (alpha > 0.0) {
		const double half = 1.0 / (2.0 * alpha);
		if (std::abs(std::abs(x) - half) < pulse_singularity_tol)
			return (pi / 4.0) * pulse_sinc(half);
	}
	const double denom = 1.0 - (2.0 * alpha * x) * (2.0 * alpha * x);
	return pulse_sinc(x) * std::cos(pi * alpha * x) / denom;
}

// Shared argument checking for both designers.
inline void validate_pulse_args(std::size_t num_taps,
                                 std::size_t samples_per_symbol,
                                 double rolloff, const char* who) {
	if (num_taps < 3)
		throw std::invalid_argument(std::string(who) + ": num_taps must be >= 3");
	if ((num_taps & 1u) == 0u)
		throw std::invalid_argument(std::string(who) +
			": num_taps must be odd so a tap lands on t = 0 — the symbol "
			"instant the zero-ISI property is defined at. Use "
			"span * samples_per_symbol + 1.");
	if (samples_per_symbol < 1)
		throw std::invalid_argument(std::string(who) +
			": samples_per_symbol must be >= 1");
	if (!(rolloff >= 0.0) || !(rolloff <= 1.0))
		throw std::invalid_argument(std::string(who) +
			": rolloff must be in [0, 1], got " + std::to_string(rolloff));
}

// Scale a tap set in place according to `norm`.
template <DspField T>
void normalize_pulse(mtl::vec::dense_vector<T>& h, const std::vector<double>& raw,
                      PulseNormalization norm, const char* who) {
	double scale = 0.0;
	if (norm == PulseNormalization::unit_energy) {
		double energy = 0.0;
		for (double v : raw) energy += v * v;
		if (!(energy > 0.0))
			throw std::runtime_error(std::string(who) + ": degenerate design, zero energy");
		scale = 1.0 / std::sqrt(energy);
	} else {
		double sum = 0.0;
		for (double v : raw) sum += v;
		if (!(std::abs(sum) > 0.0))
			throw std::runtime_error(std::string(who) + ": degenerate design, zero DC gain");
		scale = 1.0 / sum;
	}
	for (std::size_t i = 0; i < raw.size(); ++i)
		h[i] = static_cast<T>(raw[i] * scale);
}

} // namespace detail

// ============================================================================
// Design functions
// ============================================================================

// Root-raised-cosine pulse.
//
// Pre:  num_taps >= 3 and odd; samples_per_symbol >= 1; rolloff in [0, 1].
// Post: returns num_taps taps, symmetric about the centre, scaled per `norm`.
//       An RRC pair with matching arguments composites to a raised cosine
//       whose samples at non-zero symbol offsets are zero to within the
//       truncation set by num_taps.
//
// The pulse is computed in double and projected to T at the end: the design
// is a one-off cost and the closed form is far better conditioned in double
// than in a narrow type, which keeps a posit16 tap set as close to the ideal
// pulse as its own precision allows rather than compounding design error on
// top of representation error.
template <DspField T>
mtl::vec::dense_vector<T> rrc_filter(
    std::size_t num_taps,
    std::size_t samples_per_symbol,
    double rolloff,
    PulseNormalization norm = PulseNormalization::unit_energy) {

	detail::validate_pulse_args(num_taps, samples_per_symbol, rolloff, "rrc_filter");

	const double centre = 0.5 * (static_cast<double>(num_taps) - 1.0);
	std::vector<double> raw(num_taps);
	for (std::size_t n = 0; n < num_taps; ++n) {
		const double x = (static_cast<double>(n) - centre) /
		                 static_cast<double>(samples_per_symbol);
		raw[n] = detail::rrc_tap(x, rolloff);
	}

	mtl::vec::dense_vector<T> h(num_taps);
	detail::normalize_pulse(h, raw, norm, "rrc_filter");
	return h;
}

// Raised-cosine pulse — the composite an RRC pair is designed to produce.
//
// Pre/Post as rrc_filter(). Sampled every samples_per_symbol taps from the
// centre, an un-truncated raised cosine is 1 at the centre and 0 everywhere
// else; that is the zero-ISI property.
template <DspField T>
mtl::vec::dense_vector<T> raised_cosine_filter(
    std::size_t num_taps,
    std::size_t samples_per_symbol,
    double rolloff,
    PulseNormalization norm = PulseNormalization::unit_energy) {

	detail::validate_pulse_args(num_taps, samples_per_symbol, rolloff,
	                            "raised_cosine_filter");

	const double centre = 0.5 * (static_cast<double>(num_taps) - 1.0);
	std::vector<double> raw(num_taps);
	for (std::size_t n = 0; n < num_taps; ++n) {
		const double x = (static_cast<double>(n) - centre) /
		                 static_cast<double>(samples_per_symbol);
		raw[n] = detail::rc_tap(x, rolloff);
	}

	mtl::vec::dense_vector<T> h(num_taps);
	detail::normalize_pulse(h, raw, norm, "raised_cosine_filter");
	return h;
}

// ============================================================================
// Intersymbol-interference measurement
// ============================================================================

// Worst residual ISI of a composite pulse, relative to its peak.
//
// `composite` is the full TX-RX response — the convolution of the transmit
// and receive pulses — and `samples_per_symbol` the sample rate it is
// expressed at. The peak tap is taken as the symbol instant; every other tap
// an exact multiple of samples_per_symbol away from it should be zero, and
// the largest of those, divided by the peak, is what this returns.
//
// A perfect untruncated raised cosine gives 0. A real design gives the
// truncation floor, which is the number worth tracking when sweeping
// coefficient precision: it says how much of the eye a given tap precision
// costs before any channel noise is involved.
//
// Pre: composite.size() >= 3, samples_per_symbol >= 1.
template <DspField T>
double peak_isi(const mtl::vec::dense_vector<T>& composite,
                std::size_t samples_per_symbol) {
	if (composite.size() < 3)
		throw std::invalid_argument("peak_isi: composite too short");
	if (samples_per_symbol < 1)
		throw std::invalid_argument("peak_isi: samples_per_symbol must be >= 1");

	std::size_t peak = 0;
	double peak_mag = std::abs(static_cast<double>(composite[0]));
	for (std::size_t i = 1; i < composite.size(); ++i) {
		const double m = std::abs(static_cast<double>(composite[i]));
		if (m > peak_mag) { peak_mag = m; peak = i; }
	}
	if (!(peak_mag > 0.0))
		throw std::runtime_error("peak_isi: composite is identically zero");

	double worst = 0.0;
	for (std::size_t i = peak % samples_per_symbol; i < composite.size();
	     i += samples_per_symbol) {
		if (i == peak) continue;
		worst = std::max(worst, std::abs(static_cast<double>(composite[i])));
	}
	return worst / peak_mag;
}

} // namespace sw::dsp::sdr
