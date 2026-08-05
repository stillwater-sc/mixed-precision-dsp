#pragma once
// constellation.hpp: QAM/PSK constellation mapping and demapping.
//
// A Constellation is an immutable table of M complex points, Gray-labelled
// and scaled to unit average power. Mapping and demapping are pure functions
// of their arguments — unlike the filters elsewhere in this library there is
// no delay line and no phase to carry between calls, so a single instance is
// safe to share across streams.
//
// Supported schemes (issue #95):
//
//   scheme    bits/sym   M    geometry
//   BPSK          1      2    2-PSK,  points +/-1
//   QPSK          2      4    4-PSK,  offset pi/4 -> (+/-1 +/- j)/sqrt(2)
//   8-PSK         3      8    8-PSK,  first point on the real axis
//   16-QAM        4     16    square, 4-PAM per axis
//   64-QAM        6     64    square, 8-PAM per axis
//   256-QAM       8    256    square, 16-PAM per axis
//
// GRAY LABELLING. Constellation-adjacent points differ in exactly one bit,
// which is what makes a symbol error usually cost a single bit error. The
// construction relies on the defining property of the reflected binary code:
// gray(k) and gray(k+1) differ in one bit. Position k on the circle (PSK) or
// on an axis (QAM) is therefore labelled gray(k), and mapping a label b back
// to its position is the inverse, gray_to_binary(b).
//
// NORMALIZATION. Every constellation is scaled so E[|s|^2] = 1 over
// equiprobable symbols, which lets an SNR or noise variance mean the same
// thing across schemes. PSK points already sit on the unit circle. Square
// M-QAM with levels +/-1, +/-3, ..., +/-(L-1) on each axis has
// E[|s|^2] = 2(M-1)/3, so the scale factor is sqrt(3 / (2(M-1))) — the
// familiar 1/sqrt(10), 1/sqrt(42), 1/sqrt(170) for 16/64/256-QAM.
//
// CONVENTIONS. Three choices that quietly ruin a receiver if mismatched:
//   * Bits are MSB-first, one bit per std::uint8_t, values 0 or 1.
//     For square QAM the leading half of the bits drives I, the trailing
//     half drives Q.
//   * A POSITIVE LLR means bit 0 is the more likely value: the returned
//     quantity is ln( P(b=0 | r) / P(b=1 | r) ).
//   * noise_variance is E[|n|^2] for the COMPLEX noise, not the per-
//     dimension variance. Complex AWGN of total variance N0 carries N0/2
//     in each of I and Q.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>

namespace sw::dsp::sdr {

// ============================================================================
// Modulation scheme
// ============================================================================

enum class Modulation {
	bpsk,      // 1 bit/symbol
	qpsk,      // 2
	psk8,      // 3
	qam16,     // 4
	qam64,     // 6
	qam256     // 8
};

// Bits per symbol for a scheme. Throws on an unnamed enumerator so a
// corrupted or future value cannot silently produce an empty constellation.
inline std::size_t bits_per_symbol(Modulation m) {
	switch (m) {
		case Modulation::bpsk:   return 1;
		case Modulation::qpsk:   return 2;
		case Modulation::psk8:   return 3;
		case Modulation::qam16:  return 4;
		case Modulation::qam64:  return 6;
		case Modulation::qam256: return 8;
	}
	throw std::invalid_argument("sdr::bits_per_symbol: unknown Modulation value");
}

// Number of constellation points, M = 2^bits_per_symbol.
inline std::size_t modulation_order(Modulation m) {
	return std::size_t{1} << bits_per_symbol(m);
}

// True for the square-QAM schemes, false for the PSK schemes. The two
// families are labelled differently: PSK Gray-codes around the circle,
// square QAM Gray-codes each axis independently.
inline bool is_square_qam(Modulation m) {
	return m == Modulation::qam16 || m == Modulation::qam64 ||
	       m == Modulation::qam256;
}

inline const char* to_string(Modulation m) {
	switch (m) {
		case Modulation::bpsk:   return "BPSK";
		case Modulation::qpsk:   return "QPSK";
		case Modulation::psk8:   return "8-PSK";
		case Modulation::qam16:  return "16-QAM";
		case Modulation::qam64:  return "64-QAM";
		case Modulation::qam256: return "256-QAM";
	}
	return "unknown";
}

namespace detail {

// Reflected binary (Gray) code: successive values differ in one bit.
inline constexpr std::size_t binary_to_gray(std::size_t b) {
	return b ^ (b >> 1);
}

// Inverse of binary_to_gray: recover the position from its Gray label.
inline constexpr std::size_t gray_to_binary(std::size_t g, std::size_t nbits) {
	std::size_t b = g;
	for (std::size_t shift = 1; shift < nbits; shift <<= 1) b ^= b >> shift;
	return b;
}

} // namespace detail

// ============================================================================
// Constellation
// ============================================================================

template <DspField T>
class Constellation {
public:
	using scalar_type = T;
	using symbol_type = complex_for_t<T>;

	// Build the table for `m`. Establishes every invariant:
	//   I1  points().size() == order() == 2^bits_per_symbol()
	//   I2  bits_per_symbol() agrees with the scheme
	//   I3  unit average power
	//   I4  all points distinct
	//   I5  constellation-adjacent points differ in exactly one bit
	// The invariants are properties of the construction below rather than
	// anything a caller can disturb — the object exposes no mutators — so
	// they are verified in the unit tests instead of by runtime assertion,
	// per this project's testing rules.
	explicit Constellation(Modulation m)
		: modulation_(m),
		  bits_(sw::dsp::sdr::bits_per_symbol(m)),   // throws on a bad enum
		  points_(sw::dsp::sdr::modulation_order(m)) {
		if (is_square_qam(m)) build_square_qam();
		else                  build_psk();
	}

	Modulation  modulation()      const { return modulation_; }
	std::size_t bits_per_symbol() const { return bits_; }
	std::size_t order()           const { return points_.size(); }

	// The table, indexed by bit pattern (MSB-first).
	const mtl::vec::dense_vector<symbol_type>& points() const { return points_; }

	// ------------------------------------------------------------------
	// Mapping
	// ------------------------------------------------------------------

	// Point for a bit pattern already packed into an index.
	// Pre: index < order().
	symbol_type symbol(std::size_t index) const {
		if (index >= points_.size())
			throw std::out_of_range("Constellation::symbol: index " +
				std::to_string(index) + " >= order " + std::to_string(points_.size()));
		return points_[index];
	}

	// Pack MSB-first bits into a symbol index.
	// Pre: bits.size() == bits_per_symbol(), every element in {0, 1}.
	std::size_t index_of(std::span<const std::uint8_t> bits) const {
		require_bit_count(bits.size(), "index_of");
		std::size_t index = 0;
		for (std::uint8_t b : bits) {
			if (b > 1)
				throw std::invalid_argument(
					"Constellation::index_of: bit value must be 0 or 1, got " +
					std::to_string(static_cast<int>(b)));
			index = (index << 1) | b;
		}
		return index;
	}

	// Bits -> point. Pre: as index_of().
	symbol_type map(std::span<const std::uint8_t> bits) const {
		return points_[index_of(bits)];
	}

	// Unpack a symbol index into MSB-first bits.
	// Pre: index < order(), out.size() == bits_per_symbol().
	void bits_of(std::size_t index, std::span<std::uint8_t> out) const {
		if (index >= points_.size())
			throw std::out_of_range("Constellation::bits_of: index out of range");
		require_bit_count(out.size(), "bits_of");
		for (std::size_t k = 0; k < bits_; ++k)
			out[k] = static_cast<std::uint8_t>((index >> (bits_ - 1 - k)) & 1u);
	}

	// ------------------------------------------------------------------
	// Hard-decision demapping
	// ------------------------------------------------------------------

	// Nearest point by Euclidean distance; returns its index.
	//
	// Exhaustive over M points, which is exact for every scheme including
	// the PSK ones. Square QAM admits an O(1) per-axis slicer; that is a
	// speed optimization to make once a profile says it matters, not a
	// correctness one.
	std::size_t demap_hard(symbol_type r) const {
		std::size_t best = 0;
		T best_d = distance_squared(r, points_[0]);
		for (std::size_t i = 1; i < points_.size(); ++i) {
			const T d = distance_squared(r, points_[i]);
			if (d < best_d) { best_d = d; best = i; }
		}
		return best;
	}

	// Nearest point, unpacked to bits. Pre: out.size() == bits_per_symbol().
	void demap_hard_bits(symbol_type r, std::span<std::uint8_t> out) const {
		bits_of(demap_hard(r), out);
	}

	// ------------------------------------------------------------------
	// Soft-decision demapping
	// ------------------------------------------------------------------

	// Exact log-likelihood ratios:
	//
	//   LLR_k = ln( sum_{s: b_k=0} exp(-|r-s|^2 / N0)
	//             / sum_{s: b_k=1} exp(-|r-s|^2 / N0) )
	//
	// evaluated with the max-subtraction form of log-sum-exp, so the
	// exponentials stay bounded no matter how small N0 is.
	//
	// Pre: noise_variance > 0, out.size() == bits_per_symbol().
	// Post: out[k] > 0 iff bit k is more likely 0.
	//
	// Requires exp() and log() for T. Being a template member it is only
	// instantiated when called, so a scalar type without transcendentals
	// can still use everything else on this class.
	void demap_llr(symbol_type r, T noise_variance, std::span<T> out) const {
		require_noise_variance(noise_variance, "demap_llr");
		require_bit_count(out.size(), "demap_llr");

		using std::exp;
		using std::log;

		for (std::size_t k = 0; k < bits_; ++k) {
			// Track each hypothesis' best exponent so the sums can be
			// shifted into a safe range before exponentiating.
			T max0 = lowest_metric(), max1 = lowest_metric();
			for (std::size_t i = 0; i < points_.size(); ++i) {
				const T metric = -distance_squared(r, points_[i]) / noise_variance;
				if (bit_is_set(i, k)) { if (metric > max1) max1 = metric; }
				else                  { if (metric > max0) max0 = metric; }
			}

			T sum0{}, sum1{};
			for (std::size_t i = 0; i < points_.size(); ++i) {
				const T metric = -distance_squared(r, points_[i]) / noise_variance;
				if (bit_is_set(i, k)) sum1 = sum1 + exp(metric - max1);
				else                  sum0 = sum0 + exp(metric - max0);
			}

			out[k] = (max0 + log(sum0)) - (max1 + log(sum1));
		}
	}

	// Max-log approximation:
	//
	//   LLR_k ~ ( min_{s: b_k=1} |r-s|^2 - min_{s: b_k=0} |r-s|^2 ) / N0
	//
	// Keeps the sign and ordering of the exact LLR and costs no
	// transcendentals, so it works for any DspField. Accurate to a fraction
	// of a dB at the SNRs where a coded link operates, and the usual choice
	// in a real receiver.
	//
	// Pre: noise_variance > 0, out.size() == bits_per_symbol().
	void demap_llr_maxlog(symbol_type r, T noise_variance, std::span<T> out) const {
		require_noise_variance(noise_variance, "demap_llr_maxlog");
		require_bit_count(out.size(), "demap_llr_maxlog");

		for (std::size_t k = 0; k < bits_; ++k) {
			T best0 = highest_metric(), best1 = highest_metric();
			for (std::size_t i = 0; i < points_.size(); ++i) {
				const T d = distance_squared(r, points_[i]);
				if (bit_is_set(i, k)) { if (d < best1) best1 = d; }
				else                  { if (d < best0) best0 = d; }
			}
			out[k] = (best1 - best0) / noise_variance;
		}
	}

	// Average power of the table, (1/M) * sum |p_i|^2. Exactly 1 by
	// construction; exposed so tests and precision sweeps can measure how
	// far a narrow scalar type drifts from it.
	T average_power() const {
		T acc{};
		for (std::size_t i = 0; i < points_.size(); ++i)
			acc = acc + magnitude_squared(points_[i]);
		return acc / static_cast<T>(static_cast<double>(points_.size()));
	}

private:
	// --- construction -------------------------------------------------

	// M points on the unit circle. Position k sits at angle
	// (2*pi*k)/M + offset and carries the label gray(k), so neighbours in
	// angle differ in one bit (I5). Writing the table by label means
	// indexing it with a bit pattern needs no further decoding.
	void build_psk() {
		const std::size_t M = points_.size();
		const double offset = (modulation_ == Modulation::qpsk)
			? pi / 4.0    // put QPSK on the diagonals: (+/-1 +/- j)/sqrt(2)
			: 0.0;
		for (std::size_t k = 0; k < M; ++k) {
			const double theta = two_pi * static_cast<double>(k) /
			                     static_cast<double>(M) + offset;
			points_[detail::binary_to_gray(k)] =
				symbol_type(static_cast<T>(std::cos(theta)),
				            static_cast<T>(std::sin(theta)));
		}
	}

	// Square M-QAM: L = sqrt(M) levels per axis, the leading half of the
	// bits on I and the trailing half on Q, each half Gray-coded along its
	// own axis. Level index p maps to amplitude (2p - (L-1)), giving the
	// odd-integer grid +/-1, +/-3, ..., then the whole table is scaled to
	// unit average power.
	void build_square_qam() {
		const std::size_t half = bits_ / 2;
		const std::size_t L    = std::size_t{1} << half;
		const double scale = std::sqrt(3.0 / (2.0 * (static_cast<double>(points_.size()) - 1.0)));

		for (std::size_t index = 0; index < points_.size(); ++index) {
			const std::size_t gi = index >> half;                     // I label
			const std::size_t gq = index & (L - 1);                   // Q label
			const double i_amp = level_amplitude(gi, half, L);
			const double q_amp = level_amplitude(gq, half, L);
			points_[index] = symbol_type(static_cast<T>(scale * i_amp),
			                             static_cast<T>(scale * q_amp));
		}
	}

	// Amplitude for a Gray-labelled level: undo the Gray code to recover
	// the position, then place it on the odd-integer grid.
	static double level_amplitude(std::size_t gray_label, std::size_t nbits,
	                               std::size_t levels) {
		const std::size_t p = detail::gray_to_binary(gray_label, nbits);
		return 2.0 * static_cast<double>(p) - (static_cast<double>(levels) - 1.0);
	}

	// --- helpers ------------------------------------------------------

	// Bit k of a symbol index, counting MSB-first to match the public
	// bit convention.
	bool bit_is_set(std::size_t index, std::size_t k) const {
		return ((index >> (bits_ - 1 - k)) & 1u) != 0u;
	}

	static T magnitude_squared(const symbol_type& s) {
		return s.real() * s.real() + s.imag() * s.imag();
	}

	static T distance_squared(const symbol_type& a, const symbol_type& b) {
		const T dr = a.real() - b.real();
		const T di = a.imag() - b.imag();
		return dr * dr + di * di;
	}

	// Seeds for running max / min searches. Written in terms of a large
	// finite magnitude rather than infinity so that scalar types without
	// an infinity representation (fixed-point, some posit configurations)
	// behave the same as the floating ones.
	static T lowest_metric()  { return static_cast<T>(-1e30); }
	static T highest_metric() { return static_cast<T>( 1e30); }

	void require_bit_count(std::size_t n, const char* who) const {
		if (n != bits_)
			throw std::invalid_argument(std::string("Constellation::") + who +
				": expected " + std::to_string(bits_) + " bits for " +
				to_string(modulation_) + ", got " + std::to_string(n));
	}

	static void require_noise_variance(T n0, const char* who) {
		if (!(n0 > T{}))
			throw std::invalid_argument(std::string("Constellation::") + who +
				": noise_variance must be > 0");
	}

	Modulation                          modulation_;
	std::size_t                         bits_;
	mtl::vec::dense_vector<symbol_type> points_;
};

} // namespace sw::dsp::sdr
