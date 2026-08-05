// test_sdr_constellation.cpp: QAM/PSK constellation mapping and demapping.
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)`.
//
// The class invariants I1-I5 from the design are asserted here rather than
// by runtime assertion in the header: Constellation exposes no mutators, so
// they are properties of the construction and a test is the right place to
// pin them.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/sdr/constellation.hpp>

#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/posit/posit.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using sw::dsp::sdr::Constellation;
using sw::dsp::sdr::Modulation;
using sw::dsp::sdr::to_string;

static void check(bool condition, const std::string& msg) {
	if (!condition) throw std::runtime_error("test failed: " + msg);
}

static const Modulation all_schemes[] = {
	Modulation::bpsk, Modulation::qpsk, Modulation::psk8,
	Modulation::qam16, Modulation::qam64, Modulation::qam256
};

// Hamming distance between two symbol indices over `nbits` bits.
static int hamming(std::size_t a, std::size_t b, std::size_t nbits) {
	int d = 0;
	for (std::size_t k = 0; k < nbits; ++k)
		if (((a >> k) & 1u) != ((b >> k) & 1u)) ++d;
	return d;
}

// A comfortable operating Es/N0 for a scheme, in dB.
//
// The required SNR climbs with constellation order — the issue's own design
// note puts 256-QAM at ~40 dB — so a single fixed SNR either leaves BPSK
// untested or drowns 256-QAM. 10*log10(M) tracks the shrinking minimum
// distance and the +12 dB margin puts every scheme comfortably clear of its
// error floor, which is what these tests are trying to exercise.
static double comfortable_esn0_db(const Constellation<double>& c) {
	return 10.0 * std::log10(static_cast<double>(c.order())) + 12.0;
}

// ---------------------------------------------------------------------------
// I1 / I2: table size and bits-per-symbol agree with the scheme
// ---------------------------------------------------------------------------
static void test_sizes() {
	const struct { Modulation m; std::size_t bits, order; } expect[] = {
		{Modulation::bpsk,   1,   2}, {Modulation::qpsk,   2,   4},
		{Modulation::psk8,   3,   8}, {Modulation::qam16,  4,  16},
		{Modulation::qam64,  6,  64}, {Modulation::qam256, 8, 256},
	};
	for (const auto& e : expect) {
		Constellation<double> c(e.m);
		check(c.bits_per_symbol() == e.bits,
		      std::string(to_string(e.m)) + " bits_per_symbol");
		check(c.order() == e.order, std::string(to_string(e.m)) + " order");
		check(c.points().size() == e.order,
		      std::string(to_string(e.m)) + " table size");
		check(c.order() == (std::size_t{1} << c.bits_per_symbol()),
		      std::string(to_string(e.m)) + " order == 2^bits");
	}
	std::cout << "  sizes: passed\n";
}

// ---------------------------------------------------------------------------
// I3: unit average power
// ---------------------------------------------------------------------------
static void test_unit_average_power() {
	for (Modulation m : all_schemes) {
		Constellation<double> c(m);
		double acc = 0.0;
		for (std::size_t i = 0; i < c.order(); ++i) {
			const auto p = c.symbol(i);
			acc += p.real() * p.real() + p.imag() * p.imag();
		}
		const double avg = acc / static_cast<double>(c.order());
		check(std::abs(avg - 1.0) < 1e-12,
		      std::string(to_string(m)) + " average power = " + std::to_string(avg));
		check(std::abs(c.average_power() - 1.0) < 1e-12,
		      std::string(to_string(m)) + " average_power() disagrees");
	}
	// The textbook QAM scale factors, as a check that the normalization is
	// the standard one and not merely self-consistent.
	const struct { Modulation m; double inner; } qam[] = {
		{Modulation::qam16,  1.0 / std::sqrt(10.0)},
		{Modulation::qam64,  1.0 / std::sqrt(42.0)},
		{Modulation::qam256, 1.0 / std::sqrt(170.0)},
	};
	for (const auto& q : qam) {
		Constellation<double> c(q.m);
		double smallest = 1e30;
		for (std::size_t i = 0; i < c.order(); ++i)
			smallest = std::min(smallest, std::abs(c.symbol(i).real()));
		check(std::abs(smallest - q.inner) < 1e-12,
		      std::string(to_string(q.m)) + " innermost level " +
		      std::to_string(smallest) + ", expected " + std::to_string(q.inner));
	}
	std::cout << "  unit_average_power: passed\n";
}

// ---------------------------------------------------------------------------
// I4: all points distinct
// ---------------------------------------------------------------------------
static void test_points_distinct() {
	for (Modulation m : all_schemes) {
		Constellation<double> c(m);
		for (std::size_t i = 0; i < c.order(); ++i) {
			for (std::size_t j = i + 1; j < c.order(); ++j) {
				const auto a = c.symbol(i), b = c.symbol(j);
				const double d = std::hypot(a.real() - b.real(), a.imag() - b.imag());
				check(d > 1e-9, std::string(to_string(m)) + " points " +
				      std::to_string(i) + " and " + std::to_string(j) + " coincide");
			}
		}
	}
	std::cout << "  points_distinct: passed\n";
}

// ---------------------------------------------------------------------------
// I5: Gray labelling — constellation-adjacent points differ in one bit
//
// "Adjacent" is defined per family: for PSK, neighbours in angle; for square
// QAM, neighbours along an axis. Checking nearest-neighbour-in-distance would
// be wrong for QAM, where the diagonal neighbour is also close but is allowed
// to differ in two bits.
// ---------------------------------------------------------------------------
static void test_gray_labelling() {
	// PSK: sort indices by angle and walk the circle.
	for (Modulation m : {Modulation::bpsk, Modulation::qpsk, Modulation::psk8}) {
		Constellation<double> c(m);
		std::vector<std::size_t> order(c.order());
		for (std::size_t i = 0; i < c.order(); ++i) order[i] = i;
		std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
			return std::atan2(c.symbol(a).imag(), c.symbol(a).real()) <
			       std::atan2(c.symbol(b).imag(), c.symbol(b).real());
		});
		for (std::size_t p = 0; p < order.size(); ++p) {
			const std::size_t q = (p + 1) % order.size();
			if (order.size() == 2 && p == 1) continue;   // BPSK: one pair only
			const int d = hamming(order[p], order[q], c.bits_per_symbol());
			check(d == 1, std::string(to_string(m)) +
			      ": angle-adjacent labels differ in " + std::to_string(d) + " bits");
		}
	}

	// Square QAM: neighbours along I at fixed Q, and along Q at fixed I.
	for (Modulation m : {Modulation::qam16, Modulation::qam64, Modulation::qam256}) {
		Constellation<double> c(m);
		const std::size_t half = c.bits_per_symbol() / 2;
		const std::size_t L    = std::size_t{1} << half;

		// Order the axis labels by the amplitude they produce, then check
		// that neighbouring amplitudes carry labels one bit apart.
		std::vector<std::size_t> axis(L);
		for (std::size_t g = 0; g < L; ++g) axis[g] = g;
		std::sort(axis.begin(), axis.end(), [&](std::size_t a, std::size_t b) {
			// index with label `a` on I and label 0 on Q
			return c.symbol((a << half) | 0).real() < c.symbol((b << half) | 0).real();
		});
		for (std::size_t p = 0; p + 1 < L; ++p) {
			const int d = hamming(axis[p], axis[p + 1], half);
			check(d == 1, std::string(to_string(m)) +
			      ": amplitude-adjacent axis labels differ in " +
			      std::to_string(d) + " bits");
		}
	}
	std::cout << "  gray_labelling: passed\n";
}

// ---------------------------------------------------------------------------
// Round trip: map -> demap recovers the bits, for every symbol of every scheme
// ---------------------------------------------------------------------------
static void test_round_trip() {
	for (Modulation m : all_schemes) {
		Constellation<double> c(m);
		std::vector<std::uint8_t> bits(c.bits_per_symbol()), back(c.bits_per_symbol());
		for (std::size_t i = 0; i < c.order(); ++i) {
			c.bits_of(i, back);
			check(c.index_of(back) == i,
			      std::string(to_string(m)) + " bits_of/index_of round trip");

			bits = back;
			const auto s = c.map(bits);
			check(c.demap_hard(s) == i,
			      std::string(to_string(m)) + " noiseless demap_hard mismatch at " +
			      std::to_string(i));

			c.demap_hard_bits(s, back);
			check(back == bits, std::string(to_string(m)) +
			      " noiseless bit round trip mismatch at " + std::to_string(i));
		}
	}
	std::cout << "  round_trip: passed\n";
}

// ---------------------------------------------------------------------------
// Hard decision under noise: at high SNR the symbol error rate is tiny, and
// every decision is genuinely the nearest point.
// ---------------------------------------------------------------------------
static void test_hard_decision_under_noise() {
	std::mt19937 rng(12345);
	for (Modulation m : all_schemes) {
		Constellation<double> c(m);
		const double n0 = std::pow(10.0, -comfortable_esn0_db(c) / 10.0);
		std::normal_distribution<double> gauss(0.0, std::sqrt(n0 / 2.0));

		std::size_t errors = 0;
		const std::size_t trials = 2000;
		std::uniform_int_distribution<std::size_t> pick(0, c.order() - 1);
		for (std::size_t t = 0; t < trials; ++t) {
			const std::size_t tx = pick(rng);
			const auto s = c.symbol(tx);
			const std::complex<double> r(s.real() + gauss(rng), s.imag() + gauss(rng));
			const std::size_t rx = c.demap_hard(r);
			if (rx != tx) ++errors;

			// Whatever it decided, it must be the true minimum-distance point.
			double best = 1e30;
			std::size_t best_i = 0;
			for (std::size_t i = 0; i < c.order(); ++i) {
				const auto p = c.symbol(i);
				const double d = (r.real() - p.real()) * (r.real() - p.real()) +
				                 (r.imag() - p.imag()) * (r.imag() - p.imag());
				if (d < best) { best = d; best_i = i; }
			}
			check(rx == best_i, std::string(to_string(m)) +
			      " demap_hard is not minimum-distance");
		}
		// A loose ceiling: this measures correctness, not a precise SER.
		check(errors * 100 < trials, std::string(to_string(m)) +
		      " symbol error rate too high at 30 dB: " + std::to_string(errors) +
		      "/" + std::to_string(trials));
	}
	std::cout << "  hard_decision_under_noise: passed\n";
}

// ---------------------------------------------------------------------------
// LLR sign agrees with the transmitted bit, and the exact and max-log forms
// agree on sign. Positive LLR means bit 0.
// ---------------------------------------------------------------------------
static void test_llr_signs() {
	std::mt19937 rng(999);
	for (Modulation m : all_schemes) {
		Constellation<double> c(m);
		const std::size_t nb = c.bits_per_symbol();
		const double n0 = std::pow(10.0, -comfortable_esn0_db(c) / 10.0);
		std::normal_distribution<double> gauss(0.0, std::sqrt(n0 / 2.0));
		std::uniform_int_distribution<std::size_t> pick(0, c.order() - 1);

		std::vector<std::uint8_t> tx_bits(nb);
		std::vector<double> llr(nb), llr_ml(nb);
		std::size_t sign_errors = 0, disagreements = 0;
		const std::size_t trials = 500;

		for (std::size_t t = 0; t < trials; ++t) {
			const std::size_t tx = pick(rng);
			c.bits_of(tx, tx_bits);
			const auto s = c.symbol(tx);
			const std::complex<double> r(s.real() + gauss(rng), s.imag() + gauss(rng));

			c.demap_llr(r, n0, llr);
			c.demap_llr_maxlog(r, n0, llr_ml);

			for (std::size_t k = 0; k < nb; ++k) {
				check(std::isfinite(llr[k]),
				      std::string(to_string(m)) + " exact LLR not finite");
				check(std::isfinite(llr_ml[k]),
				      std::string(to_string(m)) + " max-log LLR not finite");
				// Positive LLR <=> bit 0.
				const std::uint8_t decided    = (llr[k] > 0.0) ? 0 : 1;
				const std::uint8_t decided_ml = (llr_ml[k] > 0.0) ? 0 : 1;
				if (decided != tx_bits[k]) ++sign_errors;
				if (decided != decided_ml) ++disagreements;
			}
		}
		// Well clear of the error floor, the soft decisions should almost
		// always match the transmitted bits and the two LLR forms should
		// almost always agree with each other.
		const std::size_t total = trials * nb;
		check(sign_errors * 100 < total, std::string(to_string(m)) +
		      " LLR sign disagrees with the transmitted bit too often: " +
		      std::to_string(sign_errors) + "/" + std::to_string(total));
		check(disagreements * 100 < total, std::string(to_string(m)) +
		      " exact and max-log LLR disagree too often: " +
		      std::to_string(disagreements) + "/" + std::to_string(total));
	}
	std::cout << "  llr_signs: passed\n";
}

// ---------------------------------------------------------------------------
// LLR magnitude behaves: cleaner channel -> more confident, and a symbol
// sitting exactly on a decision boundary gives ~0.
// ---------------------------------------------------------------------------
static void test_llr_magnitude() {
	Constellation<double> c(Modulation::qpsk);
	std::vector<double> hi(2), lo(2);
	const auto s = c.symbol(0);

	c.demap_llr(s, 0.01, hi);   // low noise
	c.demap_llr(s, 1.00, lo);   // high noise
	for (std::size_t k = 0; k < 2; ++k)
		check(std::abs(hi[k]) > std::abs(lo[k]),
		      "LLR magnitude should grow as noise falls");

	// BPSK: the origin is equidistant from both points, so the LLR is 0.
	Constellation<double> b(Modulation::bpsk);
	std::vector<double> mid(1);
	b.demap_llr(std::complex<double>(0.0, 0.0), 0.1, mid);
	check(std::abs(mid[0]) < 1e-12,
	      "BPSK LLR at the origin should be 0, got " + std::to_string(mid[0]));

	// ...and it should be antisymmetric about that boundary.
	std::vector<double> plus(1), minus(1);
	b.demap_llr(std::complex<double>( 0.3, 0.0), 0.1, plus);
	b.demap_llr(std::complex<double>(-0.3, 0.0), 0.1, minus);
	check(std::abs(plus[0] + minus[0]) < 1e-9,
	      "BPSK LLR should be antisymmetric about the origin");
	check(plus[0] > 0.0, "BPSK: +0.3 lies nearer the bit-0 point, LLR must be > 0");

	std::cout << "  llr_magnitude: passed\n";
}

// ---------------------------------------------------------------------------
// Mixed precision: the table survives narrow scalar types.
// ---------------------------------------------------------------------------
template <typename T>
static void check_precision(const char* name, double power_tol) {
	for (Modulation m : all_schemes) {
		Constellation<T> c(m);
		check(c.order() == (std::size_t{1} << c.bits_per_symbol()),
		      std::string(name) + " " + to_string(m) + " order");

		const double avg = static_cast<double>(c.average_power());
		check(std::abs(avg - 1.0) < power_tol,
		      std::string(name) + " " + to_string(m) + " average power " +
		      std::to_string(avg) + " outside tolerance " + std::to_string(power_tol));

		// Noiseless round trip must still be exact: quantizing the table
		// moves every point, but it moves the transmitted point the same
		// way, so the nearest neighbour is unchanged.
		std::vector<std::uint8_t> bits(c.bits_per_symbol());
		for (std::size_t i = 0; i < c.order(); ++i) {
			c.bits_of(i, bits);
			check(c.demap_hard(c.map(bits)) == i,
			      std::string(name) + " " + to_string(m) +
			      " round trip failed at symbol " + std::to_string(i));
		}
	}
}

static void test_mixed_precision() {
	using posit32  = sw::universal::posit<32, 2>;
	using posit16  = sw::universal::posit<16, 2>;
	using cfloat32 = sw::universal::cfloat<32, 8, std::uint32_t, true, false, false>;

	check_precision<double>  ("double",   1e-12);
	check_precision<float>   ("float",    1e-6);
	check_precision<posit32> ("posit32",  1e-6);
	check_precision<cfloat32>("cfloat32", 1e-6);
	// posit16 has ~11 bits of mantissa near unity; 256-QAM's innermost
	// levels sit around 0.077, so the table quantizes visibly. The round
	// trip still has to hold.
	check_precision<posit16> ("posit16",  2e-2);

	std::cout << "  mixed_precision: passed\n";
}

// ---------------------------------------------------------------------------
// Contract violations are reported, not ignored
// ---------------------------------------------------------------------------
static void test_validation() {
	Constellation<double> c(Modulation::qam16);
	std::vector<std::uint8_t> wrong_size(3, 0), right_size(4, 0), out(4);

	bool caught = false;
	try { c.map(wrong_size); } catch (const std::invalid_argument&) { caught = true; }
	check(caught, "wrong bit count should throw");

	caught = false;
	std::vector<std::uint8_t> bad_bit = {0, 1, 2, 0};
	try { c.index_of(bad_bit); } catch (const std::invalid_argument&) { caught = true; }
	check(caught, "bit value > 1 should throw");

	caught = false;
	try { c.symbol(16); } catch (const std::out_of_range&) { caught = true; }
	check(caught, "index >= order should throw");

	caught = false;
	try { c.bits_of(99, out); } catch (const std::out_of_range&) { caught = true; }
	check(caught, "bits_of with a bad index should throw");

	caught = false;
	std::vector<double> llr(4);
	try { c.demap_llr(std::complex<double>(0, 0), 0.0, llr); }
	catch (const std::invalid_argument&) { caught = true; }
	check(caught, "zero noise variance should throw");

	caught = false;
	try { c.demap_llr(std::complex<double>(0, 0), -1.0, llr); }
	catch (const std::invalid_argument&) { caught = true; }
	check(caught, "negative noise variance should throw");

	caught = false;
	try { sw::dsp::sdr::bits_per_symbol(static_cast<Modulation>(99)); }
	catch (const std::invalid_argument&) { caught = true; }
	check(caught, "unknown Modulation should throw");

	std::cout << "  validation: passed\n";
}

// ---------------------------------------------------------------------------

int main() {
	try {
		std::cout << "SDR constellation tests\n";
		test_sizes();
		test_unit_average_power();
		test_points_distinct();
		test_gray_labelling();
		test_round_trip();
		test_hard_decision_under_noise();
		test_llr_signs();
		test_llr_magnitude();
		test_mixed_precision();
		test_validation();
		std::cout << "All SDR constellation tests passed.\n";
		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << "\n";
		return 1;
	}
}
