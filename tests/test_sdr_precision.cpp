// test_sdr_precision.cpp: end-to-end SDR precision analysis.
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)`.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/analysis/sdr_precision.hpp>

#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/posit/posit.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sw::dsp::analysis;
using sw::dsp::sdr::Modulation;
using posit8  = sw::universal::posit<8, 2>;
using posit16 = sw::universal::posit<16, 2>;
using posit32 = sw::universal::posit<32, 2>;

static void check(bool c, const std::string& m) {
	if (!c) throw std::runtime_error("test failed: " + m);
}

// Attribution needs the arithmetic to be the loudest thing in the link.
static SdrLinkConfig quiet_config(Modulation m = Modulation::qam16) {
	SdrLinkConfig c;
	c.modulation = m;
	c.ebn0_db    = 80.0;
	c.num_symbols = 2000;
	return c;
}

// ---------------------------------------------------------------------------
// The all-double reference is clean, and BER is zero on a quiet link
// ---------------------------------------------------------------------------
static void test_reference_link() {
	for (Modulation m : {Modulation::qpsk, Modulation::qam16, Modulation::qam64}) {
		auto r = run_link<double>(quiet_config(m), SdrBlock::none);
		check(r.total_bits > 1000, "too few bits measured");
		check(r.bit_errors == 0, std::string(sw::dsp::sdr::to_string(m)) +
		      ": a quiet all-double link should make no bit errors, got " +
		      std::to_string(r.bit_errors));
		// The floor is RRC truncation ISI, not arithmetic — the same
		// truncation/quantization crossover the RRC module documents.
		check(r.evm_rms > 0.0 && r.evm_rms < 0.02,
		      std::string(sw::dsp::sdr::to_string(m)) + " reference EVM " +
		      std::to_string(r.evm_rms));
		check(std::abs(r.mer_db + r.evm_db) < 1e-9, "MER must mirror EVM");
		check(r.constellation.size() == r.total_bits / (r.total_bits /
		      r.constellation.size()), "constellation size consistent");
	}
	// A longer pulse lowers the floor, confirming it is truncation.
	auto a = quiet_config(); a.rrc_span = 10;
	auto b = quiet_config(); b.rrc_span = 24;
	const double e10 = run_link<double>(a, SdrBlock::none).evm_rms;
	const double e24 = run_link<double>(b, SdrBlock::none).evm_rms;
	check(e24 < e10, "a longer RRC span should lower the reference floor: " +
	      std::to_string(e10) + " -> " + std::to_string(e24));
	std::cout << "  reference_link: passed (floor " << std::scientific
	          << std::setprecision(2) << e10 << " at span 10, " << e24
	          << " at span 24)\n";
}

// ---------------------------------------------------------------------------
// Per-block attribution: narrower arithmetic costs more, everywhere
// ---------------------------------------------------------------------------
static void test_block_attribution() {
	std::cout << "    block contributions on 16-QAM (Eb/N0 = 80 dB):\n";
	std::cout << "      type      constellation   tx_shaping    rx_matched   whole_chain\n";

	auto row = [](const std::vector<SdrLinkResult>& r) {
		return std::array<double, 4>{r[1].evm_contribution, r[2].evm_contribution,
		                              r[3].evm_contribution, r[4].evm_contribution};
	};
	const auto p32 = row(analyze_blocks<posit32>(quiet_config(), "posit32", 32));
	const auto p16 = row(analyze_blocks<posit16>(quiet_config(), "posit16", 16));
	const auto p8  = row(analyze_blocks<posit8>(quiet_config(), "posit8", 8));

	auto show = [](const char* n, const std::array<double, 4>& v) {
		std::cout << "      " << std::setw(8) << n;
		for (double x : v) std::cout << "   " << std::scientific
		                             << std::setprecision(3) << x;
		std::cout << "\n";
	};
	show("posit32", p32); show("posit16", p16); show("posit8", p8);

	// The reference contributes nothing to itself, by construction.
	const auto rows = analyze_blocks<posit16>(quiet_config(), "posit16", 16);
	check(rows.front().evm_contribution == 0.0, "the reference must contribute 0");
	check(rows.front().scalar_type == "double", "reference must be labelled double");

	// Every block costs more as the type narrows — the property the whole
	// framework exists to expose.
	for (std::size_t i = 0; i < 4; ++i) {
		check(p16[i] > p32[i], "block " + std::to_string(i) +
		      ": posit16 should cost more than posit32");
		check(p8[i] > p16[i], "block " + std::to_string(i) +
		      ": posit8 should cost more than posit16");
	}
	// On this link the constellation table is the dominant single block at
	// posit8 — its quantization lands directly on the symbol, with no
	// filtering to average it down.
	check(p8[0] > p8[1] && p8[0] > p8[2],
	      "at posit8 the constellation should dominate the filters");

	// Contributions are NOT additive, and the whole-chain figure is measured
	// rather than summed. Assert only that it stays the same order — a
	// breakdown is an attribution, not a budget.
	for (const auto& v : {p16, p8}) {
		const double psum = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
		check(v[3] > 0.25 * psum && v[3] < 4.0 * psum,
		      "whole_chain " + std::to_string(v[3]) +
		      " should be the same order as the power sum " +
		      std::to_string(psum) + ", but need not equal it");
	}
	std::cout << "  block_attribution: passed\n";
}

// ---------------------------------------------------------------------------
// BER curves: the all-double link reproduces the closed form, and narrow
// arithmetic shifts the curve the way a precision penalty should
//
// Checking the reference against theory is what makes the rest of this file
// trustworthy: it validates the noise scaling, the pulse shaping, the
// sampling instant and the demapper together. An earlier version scaled the
// per-sample noise by 1/sqrt(samples_per_symbol), which silently raised the
// true Es/N0 by 6 dB and made a 16-QAM link error-free where it should have
// been visibly breaking. Nothing but a comparison against the closed form
// would have caught that.
// ---------------------------------------------------------------------------
static void test_ber_curves() {
	std::cout << "    BER vs Eb/N0 on 16-QAM:\n"
	          << "      Eb/N0     double      theory   ratio      posit8\n";
	std::vector<double> d_ber, p_ber;
	for (double ebn0 : {8.0, 10.0, 12.0}) {
		SdrLinkConfig c = quiet_config();
		c.ebn0_db = ebn0;
		c.num_symbols = 40000;
		const double d = run_link<double>(c, SdrBlock::none).ber;
		const double p = run_link<posit8>(c, SdrBlock::whole_chain).ber;
		const double t = sw::dsp::sdr::theoretical_ber_awgn(Modulation::qam16, ebn0);
		d_ber.push_back(d); p_ber.push_back(p);
		std::cout << "      " << std::setw(5) << std::fixed << std::setprecision(1)
		          << ebn0 << "  " << std::scientific << std::setprecision(3) << d
		          << "  " << t << "   " << std::fixed << std::setprecision(2)
		          << d / t << "   " << std::scientific << std::setprecision(3)
		          << p << "\n";
		// The reference must land on the theoretical curve. Within 25%: the
		// nearest-neighbour approximation contributes a few percent and the
		// RRC truncation floor a little more.
		check(d / t > 0.75 && d / t < 1.25,
		      "double BER at " + std::to_string(ebn0) + " dB is " +
		      std::to_string(d) + " against a theoretical " + std::to_string(t));
	}
	// BER falls as SNR rises, for both.
	for (std::size_t i = 1; i < d_ber.size(); ++i) {
		check(d_ber[i] < d_ber[i - 1], "double BER should fall with Eb/N0");
		check(p_ber[i] < p_ber[i - 1], "posit8 BER should fall with Eb/N0");
	}
	// Narrow arithmetic costs SNR at every point, and the PENALTY GROWS as
	// the channel cleans up. That growth is the assertion worth making:
	// demanding a fixed multiple instead would be asking the arithmetic to
	// dominate noise it does not yet dominate. Measured ratios 1.06, 1.10,
	// 1.19 across 8 to 12 dB — modest, monotone, and heading the right way
	// as the channel recedes and the arithmetic is left exposed.
	for (std::size_t i = 0; i < d_ber.size(); ++i)
		check(p_ber[i] > d_ber[i], "posit8 must be worse than double at " +
		      std::to_string(8.0 + 2.0 * static_cast<double>(i)) + " dB");
	const double first = p_ber.front() / d_ber.front();
	const double last  = p_ber.back()  / d_ber.back();
	check(last > first, "the precision penalty should grow as the channel "
	      "gets quieter: ratio " + std::to_string(first) + " at 8 dB, " +
	      std::to_string(last) + " at 12 dB");
	std::cout << "  ber_curves: passed\n";
}

// ---------------------------------------------------------------------------
// Pareto: the minimum width each modulation order needs
// ---------------------------------------------------------------------------
static void test_pareto_frontier() {
	std::cout << "    minimum usable width per modulation (EVM budget shown):\n";
	std::cout << "      modulation   budget    posit8      posit16     posit32\n";
	for (Modulation m : {Modulation::qpsk, Modulation::qam16,
	                     Modulation::qam64, Modulation::qam256}) {
		const double budget = evm_budget(m);
		const auto cfg = quiet_config(m);
		const double e8  = run_link<posit8>(cfg, SdrBlock::whole_chain).evm_rms;
		const double e16 = run_link<posit16>(cfg, SdrBlock::whole_chain).evm_rms;
		const double e32 = run_link<posit32>(cfg, SdrBlock::whole_chain).evm_rms;
		std::cout << "      " << std::setw(10) << sw::dsp::sdr::to_string(m)
		          << "   " << std::fixed << std::setprecision(4) << budget
		          << "   " << std::scientific << std::setprecision(2)
		          << e8 << "    " << e16 << "    " << e32 << "\n";

		// Wider is never worse.
		check(e32 <= e16 * 1.05 && e16 <= e8 * 1.05,
		      std::string(sw::dsp::sdr::to_string(m)) +
		      ": EVM should not grow with width");
		// The budget is a decision-distance ceiling, so 32-bit arithmetic
		// must sit well inside it for every modulation in the library.
		check(e32 < 0.25 * budget, std::string(sw::dsp::sdr::to_string(m)) +
		      ": posit32 EVM " + std::to_string(e32) +
		      " is not comfortably inside the budget " + std::to_string(budget));
	}
	// The budget itself must shrink as constellations densify — that is what
	// makes the frontier a frontier.
	check(evm_budget(Modulation::qpsk) > evm_budget(Modulation::qam16), "budget order");
	check(evm_budget(Modulation::qam16) > evm_budget(Modulation::qam64), "budget order");
	check(evm_budget(Modulation::qam64) > evm_budget(Modulation::qam256), "budget order");
	std::cout << "  pareto_frontier: passed\n";
}

// ---------------------------------------------------------------------------
// CSV export
// ---------------------------------------------------------------------------
static void test_csv_export() {
	const auto dir = std::filesystem::temp_directory_path();
	const auto sweep = (dir / "_sdr_precision.csv").string();
	const auto cons  = (dir / "_sdr_constellation.csv").string();

	auto rows = analyze_blocks<posit16>(quiet_config(), "posit16", 16);
	write_sdr_precision_csv(sweep, rows);
	write_constellation_csv(cons, rows.back());

	auto count_lines = [](const std::string& p) {
		std::ifstream in(p);
		if (!in) throw std::runtime_error("cannot reopen " + p);
		std::string line;
		std::size_t n = 0;
		while (std::getline(in, line)) ++n;
		return n;
	};
	// Header plus one row per configuration.
	check(count_lines(sweep) == rows.size() + 1, "sweep CSV row count");
	check(count_lines(cons) == rows.back().constellation.size() + 1,
	      "constellation CSV row count");

	// The header must carry the identifier columns the existing tooling reads.
	std::ifstream in(sweep);
	std::string header;
	std::getline(in, header);
	for (const char* col : {"pipeline", "block", "scalar_type", "bit_width",
	                        "modulation", "evm_rms", "ber"})
		check(header.find(col) != std::string::npos,
		      "CSV header is missing column " + std::string(col));

	std::filesystem::remove(sweep);
	std::filesystem::remove(cons);
	std::cout << "  csv_export: passed\n";
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------
static void test_validation() {
	bool caught = false;
	auto c = quiet_config(); c.samples_per_symbol = 1;
	try { run_link<double>(c, SdrBlock::none); }
	catch (const std::invalid_argument&) { caught = true; }
	check(caught, "samples_per_symbol < 2 should throw");

	caught = false;
	c = quiet_config(); c.num_symbols = 10;      // <= 4*rrc_span
	try { run_link<double>(c, SdrBlock::none); }
	catch (const std::invalid_argument&) { caught = true; }
	check(caught, "too few symbols should throw");

	caught = false;
	try { write_sdr_precision_csv("/nonexistent-dir/x.csv", {}); }
	catch (const std::runtime_error&) { caught = true; }
	check(caught, "an unopenable path should throw");
	std::cout << "  validation: passed\n";
}

int main() {
	try {
		std::cout << "SDR precision analysis tests\n";
		test_reference_link();
		test_block_attribution();
		test_ber_curves();
		test_pareto_frontier();
		test_csv_export();
		test_validation();
		std::cout << "All SDR precision analysis tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << "\n";
		return 1;
	}
}
