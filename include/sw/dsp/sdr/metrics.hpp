#pragma once
// metrics.hpp: link-quality figures of merit — EVM, MER, BER, and the
// constellation impairments that separate a precision problem from an
// analog one.
//
// These are measurement functions. They hold no state and take batches, so
// the arithmetic they report on is the caller's, not theirs: every metric is
// accumulated in double regardless of the scalar type the symbols arrive in,
// which is what lets a posit16 link and a double reference be compared on
// the same axis. That mirrors the convention in <sw/dsp/analysis/>.
//
// WHAT EACH ONE TELLS YOU
//
//   EVM / MER   how far the received cloud sits from the ideal points, as a
//               single number. Two faces of one measurement: with a common
//               normalization, MER_dB = -EVM_dB exactly.
//   BER         what the link actually delivers, after slicing. The number
//               that matters, but a coarse one — it saturates at 0 long
//               before EVM stops being informative.
//   IqImbalance whether the error is STRUCTURED. A quantization or noise
//               problem scatters symbols isotropically; a gain, offset or
//               quadrature error moves them systematically, and this
//               separates the two. Worth reaching for when EVM is worse
//               than the arithmetic alone can explain.
//
// EVM NORMALIZATION is the classic trap: the same cloud yields different EVM
// numbers depending on the reference power used. Everything here normalizes
// by the MEAN REFERENCE SYMBOL POWER, the convention 3GPP uses. Standards
// that normalize by the peak constellation magnitude report smaller figures
// for the same signal, so cross-standard comparisons need care.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/sdr/constellation.hpp>

namespace sw::dsp::sdr {

// ============================================================================
// Results
// ============================================================================

// Error vector magnitude. `rms` and `peak` are fractions of the RMS
// reference amplitude; the percent and dB fields are the same numbers in the
// other two units standards quote, carried alongside so callers do not
// re-derive them inconsistently.
struct EvmResult {
	double rms         = 0.0;   // fraction, e.g. 0.05
	double peak        = 0.0;   // fraction
	double rms_percent = 0.0;   // 100 * rms
	double peak_percent= 0.0;
	double rms_db      = 0.0;   // 20*log10(rms)
	double peak_db     = 0.0;
	std::size_t count  = 0;     // symbols measured
};

// Bit error rate over a batch.
struct BerResult {
	std::size_t bit_errors = 0;
	std::size_t total_bits = 0;
	double      rate       = 0.0;   // bit_errors / total_bits, 0 when empty
};

// Structured constellation impairments, recovered by least squares against
// the reference. The model fitted is
//
//   [ Re(r) ]   [ a  b ] [ Re(s) ]   [ i_offset ]
//   [ Im(r) ] = [ c  d ] [ Im(s) ] + [ q_offset ]
//
// which is the general real-linear map plus a DC term — enough to express
// gain error, DC offset, common rotation and quadrature (phase) error, and
// to tell them apart.
struct IqImbalance {
	double i_offset = 0.0;   // DC offset on I, in reference-amplitude units
	double q_offset = 0.0;
	double gain_i   = 1.0;   // length of the image of the I basis vector
	double gain_q   = 1.0;
	double amplitude_imbalance_db = 0.0;   // 20*log10(gain_i / gain_q)
	double phase_imbalance_deg    = 0.0;   // departure of I-to-Q angle from 90
	double common_phase_deg       = 0.0;   // overall rotation of the cloud
	double residual_evm           = 0.0;   // EVM left after removing all of it
};

// ============================================================================
// Gaussian tail
// ============================================================================

// Q(x) = P(N(0,1) > x). Written through erfc rather than as 1 - CDF so it
// stays accurate in the far tail, which is exactly where BER curves live.
inline double q_function(double x) {
	return 0.5 * std::erfc(x / std::sqrt(2.0));
}

namespace detail {

template <typename Complex>
double sym_real(const Complex& c) { return static_cast<double>(c.real()); }
template <typename Complex>
double sym_imag(const Complex& c) { return static_cast<double>(c.imag()); }

inline void require_paired(std::size_t a, std::size_t b, const char* who) {
	if (a != b)
		throw std::invalid_argument(std::string(who) +
			": reference and received must be the same length, got " +
			std::to_string(a) + " and " + std::to_string(b));
	if (a == 0)
		throw std::invalid_argument(std::string(who) + ": empty input");
}

// Solve the symmetric 3x3 normal system M*x = y by Cramer's rule. Returns
// false if M is numerically singular, which happens when the reference does
// not exercise both axes.
inline bool solve3(const double M[3][3], const double y[3], double x[3]) {
	auto det3 = [](const double m[3][3]) {
		return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
		     - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
		     + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
	};
	const double d = det3(M);
	// Scale-aware singularity test: compare against the magnitude the
	// determinant would have if the columns were merely well conditioned.
	double norm = 0.0;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j) norm = std::max(norm, std::abs(M[i][j]));
	if (!(std::abs(d) > 1e-12 * norm * norm * norm)) return false;

	for (int col = 0; col < 3; ++col) {
		double Mc[3][3];
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j) Mc[i][j] = (j == col) ? y[i] : M[i][j];
		x[col] = det3(Mc) / d;
	}
	return true;
}

} // namespace detail

// ============================================================================
// EVM / MER
// ============================================================================

// Error vector magnitude of `received` against `reference`.
//
// Pre:  both spans the same non-zero length; the reference carries non-zero
//       average power.
// Post: rms = sqrt( mean|r-s|^2 / mean|s|^2 ),
//       peak = max|r-s| / sqrt(mean|s|^2).
//
// Both are normalized by the same RMS reference amplitude, so peak and rms
// are directly comparable and their ratio is the crest factor of the error.
template <typename Complex>
EvmResult evm(std::span<const Complex> reference,
              std::span<const Complex> received) {
	detail::require_paired(reference.size(), received.size(), "evm");

	double err_power = 0.0, ref_power = 0.0, worst = 0.0;
	for (std::size_t i = 0; i < reference.size(); ++i) {
		const double sr = detail::sym_real(reference[i]);
		const double si = detail::sym_imag(reference[i]);
		const double dr = detail::sym_real(received[i]) - sr;
		const double di = detail::sym_imag(received[i]) - si;
		const double e2 = dr * dr + di * di;
		err_power += e2;
		ref_power += sr * sr + si * si;
		worst = std::max(worst, e2);
	}
	if (!(ref_power > 0.0))
		throw std::invalid_argument("evm: reference has zero average power");

	const double n         = static_cast<double>(reference.size());
	const double ref_rms   = std::sqrt(ref_power / n);
	EvmResult r;
	r.count        = reference.size();
	r.rms          = std::sqrt(err_power / n) / ref_rms;
	r.peak         = std::sqrt(worst) / ref_rms;
	r.rms_percent  = 100.0 * r.rms;
	r.peak_percent = 100.0 * r.peak;
	// An exact match is a legitimate outcome; report it as -inf dB rather
	// than letting log10(0) surface as a domain error.
	r.rms_db  = (r.rms  > 0.0) ? 20.0 * std::log10(r.rms)  : -std::numeric_limits<double>::infinity();
	r.peak_db = (r.peak > 0.0) ? 20.0 * std::log10(r.peak) : -std::numeric_limits<double>::infinity();
	return r;
}

// Modulation error ratio: 10*log10( mean|s|^2 / mean|r-s|^2 ).
//
// The same measurement as RMS EVM under this library's normalization, so
// mer_db() == -evm().rms_db to within rounding. Both are provided because
// both appear in specifications, and silently offering only one invites a
// sign error at the call site.
template <typename Complex>
double mer_db(std::span<const Complex> reference,
              std::span<const Complex> received) {
	return -evm<Complex>(reference, received).rms_db;
}

// ============================================================================
// BER
// ============================================================================

// Bit error rate between two bit sequences (one bit per byte, values 0/1).
//
// Pre:  equal, non-zero lengths.
// Post: rate = bit_errors / total_bits.
inline BerResult ber(std::span<const std::uint8_t> transmitted,
                     std::span<const std::uint8_t> received) {
	detail::require_paired(transmitted.size(), received.size(), "ber");
	BerResult r;
	r.total_bits = transmitted.size();
	for (std::size_t i = 0; i < transmitted.size(); ++i)
		if ((transmitted[i] != 0) != (received[i] != 0)) ++r.bit_errors;
	r.rate = static_cast<double>(r.bit_errors) / static_cast<double>(r.total_bits);
	return r;
}

// ============================================================================
// Theoretical BER in AWGN
// ============================================================================

// Gray-coded bit error probability against Eb/N0 for an ideal AWGN channel.
//
// BPSK and QPSK are exact: Gray-coded QPSK is two independent BPSK channels
// in quadrature, so they share a curve. The higher orders are the standard
// nearest-neighbour approximations, which count only the dominant error
// events and are therefore slightly optimistic at low SNR — they tighten as
// Eb/N0 rises and are the curves quoted in the literature.
//
// Pre:  a named Modulation.
// Post: a probability in [0, 1].
inline double theoretical_ber_awgn(Modulation m, double ebn0_db) {
	const double ebn0 = std::pow(10.0, ebn0_db / 10.0);
	const double k    = static_cast<double>(bits_per_symbol(m));   // throws on a bad enum
	const double M    = static_cast<double>(modulation_order(m));

	switch (m) {
		case Modulation::bpsk:
		case Modulation::qpsk:
			// Exact for Gray-coded BPSK and QPSK.
			return q_function(std::sqrt(2.0 * ebn0));

		case Modulation::psk8: {
			// M-PSK nearest-neighbour bound, spread over k bits per symbol.
			const double arg = std::sqrt(2.0 * k * ebn0) * std::sin(pi / M);
			return (2.0 / k) * q_function(arg);
		}

		case Modulation::qam16:
		case Modulation::qam64:
		case Modulation::qam256: {
			// Square M-QAM with Gray coding on each axis.
			const double root_m = std::sqrt(M);
			const double arg = std::sqrt(3.0 * k * ebn0 / (M - 1.0));
			return (4.0 / k) * (1.0 - 1.0 / root_m) * q_function(arg);
		}
	}
	throw std::invalid_argument("theoretical_ber_awgn: unknown Modulation value");
}

// Es/N0 for a given Eb/N0: symbol energy is bits-per-symbol times bit energy.
inline double esn0_db_from_ebn0_db(Modulation m, double ebn0_db) {
	return ebn0_db + 10.0 * std::log10(static_cast<double>(bits_per_symbol(m)));
}
inline double ebn0_db_from_esn0_db(Modulation m, double esn0_db) {
	return esn0_db - 10.0 * std::log10(static_cast<double>(bits_per_symbol(m)));
}

// ============================================================================
// Structured impairments
// ============================================================================

// Recover gain, DC offset, common rotation and quadrature error by fitting
// the real-linear model in IqImbalance against the reference.
//
// Pre:  equal, non-zero lengths; the reference must exercise BOTH axes.
//       A reference confined to one axis — BPSK, whose symbols are all real
//       — leaves the fit rank-deficient and throws, because no measurement
//       of Q gain or quadrature error exists in that data.
// Post: the fitted parameters, plus the EVM that survives removing all of
//       them, which is the part of the error that is NOT structured.
//
// A clean link returns gain_i = gain_q = 1, zero offsets, zero imbalances,
// and a residual EVM equal to the raw EVM.
template <typename Complex>
IqImbalance iq_imbalance(std::span<const Complex> reference,
                         std::span<const Complex> received) {
	detail::require_paired(reference.size(), received.size(), "iq_imbalance");

	// Normal equations for the design matrix [Re(s), Im(s), 1]. The same
	// matrix serves both output rows, so it is accumulated once.
	double M[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
	double yi[3] = {0, 0, 0}, yq[3] = {0, 0, 0};
	double ref_power = 0.0;

	for (std::size_t n = 0; n < reference.size(); ++n) {
		const double si = detail::sym_real(reference[n]);
		const double sq = detail::sym_imag(reference[n]);
		const double ri = detail::sym_real(received[n]);
		const double rq = detail::sym_imag(received[n]);

		M[0][0] += si * si; M[0][1] += si * sq; M[0][2] += si;
		                    M[1][1] += sq * sq; M[1][2] += sq;
		                                        M[2][2] += 1.0;
		yi[0] += si * ri; yi[1] += sq * ri; yi[2] += ri;
		yq[0] += si * rq; yq[1] += sq * rq; yq[2] += rq;
		ref_power += si * si + sq * sq;
	}
	M[1][0] = M[0][1];
	M[2][0] = M[0][2];
	M[2][1] = M[1][2];

	double row_i[3], row_q[3];
	if (!detail::solve3(M, yi, row_i) || !detail::solve3(M, yq, row_q))
		throw std::invalid_argument(
			"iq_imbalance: the reference does not exercise both axes, so I/Q "
			"gain and quadrature error are not observable. Use a constellation "
			"with symbols off the real axis (QPSK or higher).");

	const double a = row_i[0], b = row_i[1];
	const double c = row_q[0], d = row_q[1];

	IqImbalance r;
	r.i_offset = row_i[2];
	r.q_offset = row_q[2];
	// Where each basis vector lands: gain is the length, angle the direction.
	r.gain_i = std::hypot(a, c);
	r.gain_q = std::hypot(b, d);
	const double angle_i = std::atan2(c, a);          // ideal 0
	const double angle_q = std::atan2(d, b);          // ideal +90 deg

	r.amplitude_imbalance_db = (r.gain_i > 0.0 && r.gain_q > 0.0)
		? 20.0 * std::log10(r.gain_i / r.gain_q) : 0.0;
	r.common_phase_deg = angle_i * 180.0 / pi;
	// Departure of the I-to-Q angle from a right angle, wrapped to
	// (-180, 180] so a small error near the wrap does not read as huge.
	double quad = (angle_q - angle_i) - pi / 2.0;
	while (quad >  pi) quad -= two_pi;
	while (quad < -pi) quad += two_pi;
	r.phase_imbalance_deg = quad * 180.0 / pi;

	// What is left once the fitted model is removed: the unstructured part.
	double residual = 0.0;
	for (std::size_t n = 0; n < reference.size(); ++n) {
		const double si = detail::sym_real(reference[n]);
		const double sq = detail::sym_imag(reference[n]);
		const double pi_ = a * si + b * sq + r.i_offset;
		const double pq_ = c * si + d * sq + r.q_offset;
		const double dr = detail::sym_real(received[n]) - pi_;
		const double dq = detail::sym_imag(received[n]) - pq_;
		residual += dr * dr + dq * dq;
	}
	r.residual_evm = std::sqrt(residual / ref_power);
	return r;
}

} // namespace sw::dsp::sdr
