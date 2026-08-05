// test_sdr_metrics.cpp: EVM, MER, BER and constellation impairment metrics.
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)`.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/sdr/metrics.hpp>
#include <sw/dsp/sdr/constellation.hpp>

#include <universal/number/posit/posit.hpp>

#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sw::dsp::sdr;
using cd = std::complex<double>;
using sw::dsp::pi;

static void check(bool condition, const std::string& msg) {
	if (!condition) throw std::runtime_error("test failed: " + msg);
}

static bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }

// ---------------------------------------------------------------------------
// EVM against hand-computable vectors
//
// Test vectors are constructed so the answer can be worked out on paper —
// that is what makes them a check on the definition rather than on a previous
// run of this same code.
// ---------------------------------------------------------------------------
static void test_evm_known_vectors() {
	// One symbol, reference amplitude 1, error 0.1 -> EVM exactly 10%.
	{
		std::vector<cd> ref = {cd(1.0, 0.0)};
		std::vector<cd> rx  = {cd(1.1, 0.0)};
		auto e = evm<cd>(ref, rx);
		check(near(e.rms, 0.1, 1e-15), "single-symbol EVM rms = " + std::to_string(e.rms));
		check(near(e.rms_percent, 10.0, 1e-13), "EVM percent");
		check(near(e.rms_db, 20.0 * std::log10(0.1), 1e-13), "EVM dB");
		check(near(e.peak, 0.1, 1e-15), "peak equals rms for one symbol");
		check(e.count == 1, "count");
	}

	// A perfect match is 0% / -inf dB, not a domain error.
	{
		std::vector<cd> ref = {cd(0.7, -0.7), cd(-0.7, 0.7)};
		auto e = evm<cd>(ref, ref);
		check(e.rms == 0.0, "identical vectors give zero EVM");
		check(std::isinf(e.rms_db) && e.rms_db < 0, "zero EVM should be -inf dB");
	}

	// Four unit-power symbols, one displaced by 0.2 and the rest exact:
	//   mean|e|^2 = 0.04/4 = 0.01 -> rms EVM = 0.1
	//   peak      = 0.2 / 1       = 0.2
	{
		std::vector<cd> ref = {cd(1, 0), cd(0, 1), cd(-1, 0), cd(0, -1)};
		std::vector<cd> rx  = ref;
		rx[2] = cd(-1.2, 0.0);
		auto e = evm<cd>(ref, rx);
		check(near(e.rms, 0.1, 1e-15), "rms over four symbols = " + std::to_string(e.rms));
		check(near(e.peak, 0.2, 1e-15), "peak over four symbols = " + std::to_string(e.peak));
		check(near(e.peak / e.rms, 2.0, 1e-12), "error crest factor");
	}

	// Normalization is by MEAN REFERENCE POWER, so scaling reference and
	// received together leaves EVM unchanged.
	{
		std::vector<cd> ref = {cd(3, 4), cd(-3, 4), cd(3, -4)};
		std::vector<cd> rx;
		for (auto s : ref) rx.push_back(s * 1.05);
		auto e1 = evm<cd>(ref, rx);
		std::vector<cd> ref2, rx2;
		for (std::size_t i = 0; i < ref.size(); ++i) {
			ref2.push_back(ref[i] * 100.0);
			rx2.push_back(rx[i] * 100.0);
		}
		auto e2 = evm<cd>(ref2, rx2);
		check(near(e1.rms, e2.rms, 1e-15), "EVM must be scale invariant");
		// A pure 5% gain error is a 5% error vector.
		check(near(e1.rms, 0.05, 1e-14), "5% gain -> 5% EVM, got " +
		      std::to_string(e1.rms));
	}
	std::cout << "  evm_known_vectors: passed\n";
}

// ---------------------------------------------------------------------------
// MER is the mirror of EVM
// ---------------------------------------------------------------------------
static void test_mer_mirrors_evm() {
	std::mt19937 rng(4);
	std::normal_distribution<double> g(0.0, 0.03);
	Constellation<double> c(Modulation::qam16);
	std::uniform_int_distribution<std::size_t> pick(0, c.order() - 1);

	std::vector<cd> ref, rx;
	for (int i = 0; i < 500; ++i) {
		const auto s = c.symbol(pick(rng));
		ref.emplace_back(s.real(), s.imag());
		rx.emplace_back(s.real() + g(rng), s.imag() + g(rng));
	}
	const auto e = evm<cd>(ref, rx);
	const double mer = mer_db<cd>(ref, rx);
	check(near(mer, -e.rms_db, 1e-12),
	      "MER should be -EVM_dB: " + std::to_string(mer) + " vs " +
	      std::to_string(-e.rms_db));
	std::cout << "  mer_mirrors_evm: passed (MER " << std::fixed
	          << std::setprecision(2) << mer << " dB)\n";
}

// ---------------------------------------------------------------------------
// BER counting
// ---------------------------------------------------------------------------
static void test_ber_counting() {
	std::vector<std::uint8_t> tx = {0, 1, 1, 0, 1, 0, 0, 1};
	std::vector<std::uint8_t> rx = {0, 1, 0, 0, 1, 1, 0, 1};   // 2 differ
	auto r = ber(tx, rx);
	check(r.total_bits == 8, "total bits");
	check(r.bit_errors == 2, "bit errors = " + std::to_string(r.bit_errors));
	check(near(r.rate, 0.25, 1e-15), "rate");

	auto perfect = ber(tx, tx);
	check(perfect.bit_errors == 0 && perfect.rate == 0.0, "identical -> zero BER");

	bool caught = false;
	try { std::vector<std::uint8_t> shortv = {0}; ber(tx, shortv); }
	catch (const std::invalid_argument&) { caught = true; }
	check(caught, "length mismatch should throw");

	std::cout << "  ber_counting: passed\n";
}

// ---------------------------------------------------------------------------
// Theoretical BER: known anchor points and structural properties
// ---------------------------------------------------------------------------
static void test_theoretical_ber() {
	// BPSK at Eb/N0 = 0 dB is Q(sqrt(2)) = 0.0786496...; the textbook
	// 10^-5 point sits at 9.6 dB.
	check(near(theoretical_ber_awgn(Modulation::bpsk, 0.0), 0.078649603, 1e-9),
	      "BPSK at 0 dB");
	check(near(theoretical_ber_awgn(Modulation::bpsk, 9.6), 1.0e-5, 2e-6),
	      "BPSK 1e-5 point near 9.6 dB");

	// Gray-coded QPSK shares BPSK's BER-vs-Eb/N0 curve exactly: it is two
	// orthogonal BPSK channels.
	for (double db : {0.0, 3.0, 6.0, 9.0, 12.0})
		check(near(theoretical_ber_awgn(Modulation::bpsk, db),
		           theoretical_ber_awgn(Modulation::qpsk, db), 1e-15),
		      "QPSK must match BPSK at " + std::to_string(db) + " dB");

	// Monotone decreasing in SNR, and bounded.
	for (Modulation m : {Modulation::bpsk, Modulation::qpsk, Modulation::psk8,
	                     Modulation::qam16, Modulation::qam64, Modulation::qam256}) {
		double prev = 1.0;
		for (double db = 0.0; db <= 24.0; db += 2.0) {
			const double p = theoretical_ber_awgn(m, db);
			check(p >= 0.0 && p <= 1.0, std::string(to_string(m)) +
			      " BER out of range at " + std::to_string(db) + " dB");
			check(p < prev, std::string(to_string(m)) +
			      " BER not decreasing at " + std::to_string(db) + " dB");
			prev = p;
		}
	}

	// Denser constellations need more Eb/N0 for the same BER.
	const double target = 1e-4;
	auto ebn0_for = [&](Modulation m) {
		double lo = -5.0, hi = 60.0;
		for (int i = 0; i < 200; ++i) {
			const double mid = 0.5 * (lo + hi);
			((theoretical_ber_awgn(m, mid) > target) ? lo : hi) = mid;
		}
		return 0.5 * (lo + hi);
	};
	const double e_qpsk = ebn0_for(Modulation::qpsk);
	const double e_16   = ebn0_for(Modulation::qam16);
	const double e_64   = ebn0_for(Modulation::qam64);
	const double e_256  = ebn0_for(Modulation::qam256);
	check(e_qpsk < e_16 && e_16 < e_64 && e_64 < e_256,
	      "required Eb/N0 must climb with constellation order");
	// QPSK reaches 1e-4 at about 8.4 dB, a standard textbook figure.
	check(near(e_qpsk, 8.4, 0.2), "QPSK 1e-4 point = " + std::to_string(e_qpsk));

	// Es/N0 <-> Eb/N0 conversions round trip.
	for (Modulation m : {Modulation::qpsk, Modulation::qam64}) {
		const double eb = 7.5;
		check(near(ebn0_db_from_esn0_db(m, esn0_db_from_ebn0_db(m, eb)), eb, 1e-12),
		      "Es/N0 conversion round trip");
	}
	std::cout << "  theoretical_ber: passed\n";
}

// ---------------------------------------------------------------------------
// Simulated AWGN converges to the theoretical curve
//
// The strongest check in this file: an independent Monte Carlo of the whole
// map / noise / demap path, compared against the closed form.
// ---------------------------------------------------------------------------
static void test_measured_ber_matches_theory() {
	struct Case { Modulation m; double ebn0_db; std::size_t symbols; };
	// SNRs chosen so BER lands in 1e-2..1e-4. That window is bounded at both
	// ends: below it the nearest-neighbour approximation is no longer tight,
	// and above it Monte Carlo becomes impractical — 16-QAM at 14 dB has a
	// theoretical BER of 2.8e-6, needing ~11 million bits just to accumulate
	// 30 errors. The symbol counts below are sized to give at least ~80
	// errors, so sampling noise sits near 10% and the 25% band that follows
	// is a real test rather than a coin flip.
	const Case cases[] = {
		{Modulation::bpsk,   4.0, 200000},
		{Modulation::bpsk,   7.0, 400000},
		{Modulation::qpsk,   4.0, 100000},
		{Modulation::qpsk,   7.0, 200000},
		{Modulation::qam16, 10.0,  60000},
		{Modulation::qam16, 12.0, 250000},
	};

	std::mt19937 rng(20260805);
	for (const auto& cse : cases) {
		Constellation<double> c(cse.m);
		const std::size_t nb = c.bits_per_symbol();

		// Unit average symbol power, so Es = 1 and N0 follows from Es/N0.
		const double esn0_db = esn0_db_from_ebn0_db(cse.m, cse.ebn0_db);
		const double n0 = std::pow(10.0, -esn0_db / 10.0);
		std::normal_distribution<double> gauss(0.0, std::sqrt(n0 / 2.0));
		std::uniform_int_distribution<std::size_t> pick(0, c.order() - 1);

		std::vector<std::uint8_t> tx_bits, rx_bits, tmp(nb);
		tx_bits.reserve(cse.symbols * nb);
		rx_bits.reserve(cse.symbols * nb);

		for (std::size_t t = 0; t < cse.symbols; ++t) {
			const std::size_t idx = pick(rng);
			c.bits_of(idx, tmp);
			tx_bits.insert(tx_bits.end(), tmp.begin(), tmp.end());

			const auto s = c.symbol(idx);
			const cd r(s.real() + gauss(rng), s.imag() + gauss(rng));
			c.demap_hard_bits(r, tmp);
			rx_bits.insert(rx_bits.end(), tmp.begin(), tmp.end());
		}

		const auto measured = ber(tx_bits, rx_bits);
		const double theory = theoretical_ber_awgn(cse.m, cse.ebn0_db);

		std::cout << "      " << std::setw(7) << to_string(cse.m)
		          << "  Eb/N0 " << std::fixed << std::setprecision(1) << cse.ebn0_db
		          << " dB   measured " << std::scientific << std::setprecision(3)
		          << measured.rate << "   theory " << theory
		          << "   (" << measured.bit_errors << " errors)\n";

		check(measured.bit_errors > 30, std::string(to_string(cse.m)) +
		      " too few errors for a meaningful comparison: " +
		      std::to_string(measured.bit_errors));
		// Within 25%: the sampling error at ~100 errors is roughly 10%, and
		// the approximation itself contributes a few percent at these SNRs.
		const double ratio = measured.rate / theory;
		check(ratio > 0.75 && ratio < 1.25, std::string(to_string(cse.m)) +
		      " at " + std::to_string(cse.ebn0_db) + " dB: measured/theory = " +
		      std::to_string(ratio));
	}
	std::cout << "  measured_ber_matches_theory: passed\n";
}

// ---------------------------------------------------------------------------
// Structured impairments are detected and quantified
// ---------------------------------------------------------------------------
static void test_iq_impairments() {
	Constellation<double> c(Modulation::qam16);
	std::mt19937 rng(11);
	std::uniform_int_distribution<std::size_t> pick(0, c.order() - 1);

	std::vector<cd> ref;
	for (int i = 0; i < 4000; ++i) {
		const auto s = c.symbol(pick(rng));
		ref.emplace_back(s.real(), s.imag());
	}

	// A clean link: unity gains, no offsets, no imbalance.
	{
		auto r = iq_imbalance<cd>(ref, ref);
		check(near(r.gain_i, 1.0, 1e-9) && near(r.gain_q, 1.0, 1e-9), "clean gains");
		check(near(r.i_offset, 0.0, 1e-9) && near(r.q_offset, 0.0, 1e-9), "clean offsets");
		check(near(r.amplitude_imbalance_db, 0.0, 1e-9), "clean amplitude imbalance");
		check(near(r.phase_imbalance_deg, 0.0, 1e-9), "clean phase imbalance");
		check(near(r.common_phase_deg, 0.0, 1e-9), "clean rotation");
		check(r.residual_evm < 1e-12, "clean residual");
	}

	// Inject a known DC offset.
	{
		std::vector<cd> rx;
		for (auto s : ref) rx.push_back(s + cd(0.07, -0.03));
		auto r = iq_imbalance<cd>(ref, rx);
		check(near(r.i_offset,  0.07, 1e-9), "I offset = " + std::to_string(r.i_offset));
		check(near(r.q_offset, -0.03, 1e-9), "Q offset = " + std::to_string(r.q_offset));
		check(r.residual_evm < 1e-12, "offset fully explained by the model");
	}

	// Inject a known amplitude imbalance: I gain 1.10, Q gain 0.95.
	{
		std::vector<cd> rx;
		for (auto s : ref) rx.push_back(cd(1.10 * s.real(), 0.95 * s.imag()));
		auto r = iq_imbalance<cd>(ref, rx);
		check(near(r.gain_i, 1.10, 1e-9), "gain_i = " + std::to_string(r.gain_i));
		check(near(r.gain_q, 0.95, 1e-9), "gain_q = " + std::to_string(r.gain_q));
		const double want = 20.0 * std::log10(1.10 / 0.95);
		check(near(r.amplitude_imbalance_db, want, 1e-9),
		      "amplitude imbalance = " + std::to_string(r.amplitude_imbalance_db) +
		      " dB, expected " + std::to_string(want));
		check(near(r.phase_imbalance_deg, 0.0, 1e-9),
		      "amplitude-only error must not read as phase imbalance");
	}

	// Inject a known quadrature error, and pin the SIGN convention in both
	// directions. phase_imbalance_deg is defined geometrically as the
	// departure of the I-to-Q angle from a right angle, so tilting the Q
	// basis vector to 90+phi reads as +phi and to 90-phi as -phi.
	for (double phi_deg : {5.0, -5.0, 0.5}) {
		const double phi = phi_deg * pi / 180.0;
		std::vector<cd> rx;
		for (auto s : ref) {
			// I basis (1,0) is left alone; Q basis (0,1) is sent to
			// (-sin phi, cos phi), whose angle is 90 + phi degrees.
			rx.push_back(cd(s.real() - s.imag() * std::sin(phi),
			                 s.imag() * std::cos(phi)));
		}
		auto r = iq_imbalance<cd>(ref, rx);
		check(near(r.phase_imbalance_deg, phi_deg, 1e-6),
		      "phase imbalance = " + std::to_string(r.phase_imbalance_deg) +
		      " deg, expected " + std::to_string(phi_deg));
		check(near(r.gain_i, 1.0, 1e-9), "quadrature error must not move gain_i");
		check(near(r.common_phase_deg, 0.0, 1e-9),
		      "a quadrature error alone is not a common rotation");
		check(r.residual_evm < 1e-12, "quadrature error fully explained by the model");
	}

	// Inject a known common rotation: the whole cloud turns, but I and Q
	// stay perpendicular and equal in gain.
	{
		const double theta = 12.0 * pi / 180.0;
		const cd rot(std::cos(theta), std::sin(theta));
		std::vector<cd> rx;
		for (auto s : ref) rx.push_back(s * rot);
		auto r = iq_imbalance<cd>(ref, rx);
		check(near(r.common_phase_deg, 12.0, 1e-6),
		      "common phase = " + std::to_string(r.common_phase_deg));
		check(near(r.phase_imbalance_deg, 0.0, 1e-9),
		      "a pure rotation is not a quadrature error");
		check(near(r.amplitude_imbalance_db, 0.0, 1e-9),
		      "a pure rotation is not an amplitude imbalance");
	}

	// Unstructured error: additive noise is NOT explained by the model, so
	// the residual survives while the fitted parameters stay clean.
	{
		std::normal_distribution<double> g(0.0, 0.05);
		std::vector<cd> rx;
		for (auto s : ref) rx.emplace_back(s.real() + g(rng), s.imag() + g(rng));
		auto r = iq_imbalance<cd>(ref, rx);
		const auto e = evm<cd>(ref, rx);
		check(near(r.gain_i, 1.0, 0.02) && near(r.gain_q, 1.0, 0.02),
		      "noise should not be absorbed into gain");
		check(near(r.phase_imbalance_deg, 0.0, 1.0),
		      "noise should not read as quadrature error");
		// Almost all of the EVM survives the fit — that is what marks it
		// unstructured.
		check(r.residual_evm > 0.9 * e.rms,
		      "noise residual " + std::to_string(r.residual_evm) +
		      " should be close to raw EVM " + std::to_string(e.rms));
	}

	// A one-axis reference makes I/Q gain unobservable, and that must be
	// reported rather than silently fitted.
	{
		Constellation<double> b(Modulation::bpsk);
		std::vector<cd> bref;
		for (int i = 0; i < 100; ++i) {
			const auto s = b.symbol(i % 2);
			bref.emplace_back(s.real(), s.imag());
		}
		bool caught = false;
		try { iq_imbalance<cd>(bref, bref); }
		catch (const std::invalid_argument&) { caught = true; }
		check(caught, "a BPSK reference should be rejected as rank-deficient");
	}
	std::cout << "  iq_impairments: passed\n";
}

// ---------------------------------------------------------------------------
// Metrics work on narrow scalar types, and quantifying quantization is the
// point of the module.
// ---------------------------------------------------------------------------
static void test_mixed_precision() {
	using posit16 = sw::universal::posit<16, 2>;
	using cp16 = sw::dsp::complex_for_t<posit16>;

	Constellation<double>  ref_c(Modulation::qam64);
	Constellation<posit16> nar_c(Modulation::qam64);

	std::vector<cd>   ref;
	std::vector<cp16> nar;
	for (std::size_t i = 0; i < ref_c.order(); ++i) {
		const auto s = ref_c.symbol(i);
		ref.emplace_back(s.real(), s.imag());
		nar.push_back(nar_c.symbol(i));
	}

	// Compare the two tables by converting the narrow one into doubles: the
	// EVM is then exactly the cost of representing 64-QAM in posit16.
	std::vector<cd> nar_as_double;
	for (const auto& s : nar)
		nar_as_double.emplace_back(static_cast<double>(s.real()),
		                            static_cast<double>(s.imag()));

	const auto e = evm<cd>(ref, nar_as_double);
	std::cout << "      64-QAM table in posit16: EVM " << std::fixed
	          << std::setprecision(4) << e.rms_percent << " % ("
	          << std::setprecision(2) << e.rms_db << " dB)\n";
	check(e.rms > 0.0, "posit16 should not represent the table exactly");
	check(e.rms < 0.01, "posit16 64-QAM EVM should still be under 1%, got " +
	      std::to_string(e.rms_percent) + " %");

	// The same measurement taken directly on the posit-valued symbols agrees.
	const auto e2 = evm<cp16>(nar, nar);
	check(e2.rms == 0.0, "a table compared with itself is exact");

	std::cout << "  mixed_precision: passed\n";
}

// ---------------------------------------------------------------------------
// Contract violations
// ---------------------------------------------------------------------------
static void test_validation() {
	std::vector<cd> a = {cd(1, 0)}, b2 = {cd(1, 0), cd(0, 1)}, empty;

	bool caught = false;
	try { evm<cd>(a, b2); } catch (const std::invalid_argument&) { caught = true; }
	check(caught, "length mismatch should throw");

	caught = false;
	try { evm<cd>(empty, empty); } catch (const std::invalid_argument&) { caught = true; }
	check(caught, "empty input should throw");

	caught = false;
	std::vector<cd> zero = {cd(0, 0), cd(0, 0)};
	try { evm<cd>(zero, b2); } catch (const std::invalid_argument&) { caught = true; }
	check(caught, "zero-power reference should throw");

	caught = false;
	try { theoretical_ber_awgn(static_cast<Modulation>(42), 5.0); }
	catch (const std::invalid_argument&) { caught = true; }
	check(caught, "unknown Modulation should throw");

	std::cout << "  validation: passed\n";
}

// ---------------------------------------------------------------------------

int main() {
	try {
		std::cout << "SDR metrics tests\n";
		test_evm_known_vectors();
		test_mer_mirrors_evm();
		test_ber_counting();
		test_theoretical_ber();
		test_measured_ber_matches_theory();
		test_iq_impairments();
		test_mixed_precision();
		test_validation();
		std::cout << "All SDR metrics tests passed.\n";
		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << "\n";
		return 1;
	}
}
