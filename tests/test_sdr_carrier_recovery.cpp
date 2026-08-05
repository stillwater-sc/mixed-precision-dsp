// test_sdr_carrier_recovery.cpp: Costas carrier recovery.
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)`.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/sdr/carrier_recovery.hpp>
#include <sw/dsp/sdr/constellation.hpp>
#include <sw/dsp/sdr/metrics.hpp>

#include <universal/number/cfloat/cfloat.hpp>
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

struct Outcome {
	bool   locked = false;
	double frequency = 0.0;
	double worst_error = 0.0;   // distance from the nearest constellation point
	double residual_evm = 0.0;
};

// Run `nsym` symbols of `mod` through a carrier offset and the loop.
template <typename State = double>
static Outcome run(Modulation mod, CarrierDetector /*det*/, double df, double phi0,
                   const CarrierRecoveryConfig<State>& cfg,
                   std::size_t nsym = 8000, double noise = 0.0, unsigned seed = 1) {
	Constellation<double> c(mod);
	std::mt19937 rng(seed);
	std::uniform_int_distribution<std::size_t> pick(0, c.order() - 1);
	std::normal_distribution<double> gauss(0.0, noise);

	CarrierRecovery<State, sw::dsp::complex_for_t<State>> cr(cfg);
	using CS = sw::dsp::complex_for_t<State>;

	std::vector<cd> out;
	double ph = phi0;
	for (std::size_t n = 0; n < nsym; ++n) {
		const auto s = c.symbol(pick(rng));
		const cd x(s.real() + gauss(rng), s.imag() + gauss(rng));
		const cd rot(std::cos(ph), std::sin(ph));
		const cd spun = x * rot;
		ph += df;
		const CS y = cr.process(CS(static_cast<State>(spun.real()),
		                            static_cast<State>(spun.imag())));
		out.emplace_back(static_cast<double>(y.real()), static_cast<double>(y.imag()));
	}

	Outcome o;
	o.locked = cr.is_locked();
	o.frequency = static_cast<double>(cr.frequency());
	check(cr.invariants_hold(), "invariants violated during the run");

	// Compare the settled tail against the nearest constellation point.
	std::vector<cd> ref, got;
	for (std::size_t i = out.size() * 3 / 4; i < out.size(); ++i) {
		const auto idx = c.demap_hard(out[i]);
		const auto p = c.symbol(idx);
		ref.emplace_back(p.real(), p.imag());
		got.push_back(out[i]);
		o.worst_error = std::max(o.worst_error, std::abs(out[i] - ref.back()));
	}
	o.residual_evm = evm<cd>(ref, got).rms;
	return o;
}

static CarrierRecoveryConfig<double> base_config() {
	CarrierRecoveryConfig<double> c;
	c.loop_bandwidth = 0.02;
	c.damping        = 0.707;
	c.detector       = CarrierDetector::qpsk;
	c.max_frequency  = 0.5;
	c.enable_afc     = true;
	c.afc_gain       = 0.01;
	c.lock_threshold = 0.1;
	c.lock_average_symbols = 64.0;
	return c;
}

// ---------------------------------------------------------------------------
// Locks a phase offset, with the ambiguity a Costas loop inevitably has
// ---------------------------------------------------------------------------
static void test_phase_lock() {
	for (double phi : {0.0, 0.3, 0.8, -0.5, 1.2}) {
		auto o = run(Modulation::qpsk, CarrierDetector::qpsk, 0.0, phi, base_config());
		check(o.locked, "phi " + std::to_string(phi) + ": never locked");
		check(o.worst_error < 1e-6, "phi " + std::to_string(phi) +
		      ": residual error " + std::to_string(o.worst_error));
		check(std::abs(o.frequency) < 1e-6, "phi " + std::to_string(phi) +
		      ": frequency drifted to " + std::to_string(o.frequency));
	}
	std::cout << "  phase_lock: passed\n";
}

// ---------------------------------------------------------------------------
// AFC widens the pull-in range — that is the whole reason it exists
// ---------------------------------------------------------------------------
static void test_afc_extends_pull_in() {
	std::cout << "    offset   PLL only   with AFC\n";
	bool saw_pll_fail = false;
	for (double df : {0.02, 0.06, 0.12, 0.25}) {
		auto no_afc = base_config();  no_afc.enable_afc = false;
		auto with   = base_config();  with.enable_afc   = true;

		auto a = run(Modulation::qpsk, CarrierDetector::qpsk, df, 0.3, no_afc);
		auto b = run(Modulation::qpsk, CarrierDetector::qpsk, df, 0.3, with);

		std::cout << "    " << std::setw(6) << std::fixed << std::setprecision(2) << df
		          << "   " << std::setw(8) << (a.locked ? "lock" : "FAIL")
		          << "   " << std::setw(8) << (b.locked ? "lock" : "FAIL") << "\n";

		// With AFC every offset must acquire, and land on the true frequency.
		check(b.locked, "AFC failed to acquire at df = " + std::to_string(df));
		check(std::abs(b.frequency - df) < 1e-4, "AFC frequency = " +
		      std::to_string(b.frequency) + ", expected " + std::to_string(df));
		check(b.worst_error < 1e-6, "AFC residual error " +
		      std::to_string(b.worst_error) + " at df = " + std::to_string(df));

		if (!a.locked) saw_pll_fail = true;
		// Where the bare PLL does acquire, it must be just as accurate.
		if (a.locked)
			check(std::abs(a.frequency - df) < 1e-4,
			      "PLL frequency = " + std::to_string(a.frequency));
	}
	// The point of the test: there is an offset the bare PLL cannot reach.
	check(saw_pll_fail, "the sweep should include an offset beyond the bare "
	      "PLL's pull-in range, otherwise it does not demonstrate what AFC adds");
	std::cout << "  afc_extends_pull_in: passed\n";
}

// ---------------------------------------------------------------------------
// Works across constellations, including a decision-directed 16-QAM
// ---------------------------------------------------------------------------
static void test_constellations() {
	struct Case { Modulation m; CarrierDetector d; const char* name; };
	const Case cases[] = {
		{Modulation::bpsk,  CarrierDetector::bpsk,              "BPSK"},
		{Modulation::qpsk,  CarrierDetector::qpsk,              "QPSK"},
		{Modulation::qam16, CarrierDetector::decision_directed, "16-QAM"},
	};
	for (const auto& c : cases) {
		auto cfg = base_config();
		cfg.detector = c.d;
		// 16-QAM's decision-directed detector needs a gentler loop: its
		// slicer is only reliable once the constellation is roughly upright.
		if (c.m == Modulation::qam16) { cfg.loop_bandwidth = 0.005; cfg.afc_gain = 0.002; }
		auto o = run(c.m, c.d, 0.01, 0.2, cfg, 12000);
		check(o.locked, std::string(c.name) + ": never locked");
		check(std::abs(o.frequency - 0.01) < 2e-3, std::string(c.name) +
		      ": frequency = " + std::to_string(o.frequency));
		check(o.residual_evm < 0.02, std::string(c.name) +
		      ": residual EVM " + std::to_string(o.residual_evm * 100.0) + "%");
		std::cout << "      " << std::setw(7) << c.name << ": EVM "
		          << std::fixed << std::setprecision(4) << o.residual_evm * 100.0
		          << "%, frequency " << std::setprecision(5) << o.frequency << "\n";
	}
	std::cout << "  constellations: passed\n";
}

// ---------------------------------------------------------------------------
// Steady-state phase error shrinks as the loop narrows
// ---------------------------------------------------------------------------
static void test_phase_noise_vs_bandwidth() {
	std::cout << "    Bn*T    residual EVM\n";
	double prev = 1e30;
	for (double bw : {0.05, 0.02, 0.005}) {
		auto cfg = base_config();
		cfg.loop_bandwidth = bw;
		// Light channel noise deliberately: at sigma = 0.05 the additive
		// noise alone contributes ~7% EVM and buries the loop's own phase
		// noise, which is the quantity under test — the trend was there but
		// only in the third digit.
		auto o = run(Modulation::qpsk, CarrierDetector::qpsk, 0.01, 0.2, cfg,
		             12000, 0.01, 4);
		std::cout << "    " << std::setw(5) << std::fixed << std::setprecision(3) << bw
		          << "    " << std::setprecision(4) << o.residual_evm * 100.0 << "%\n";
		check(o.locked, "bw " + std::to_string(bw) + " did not lock");
		check(o.residual_evm < prev, "a narrower loop should leave less phase "
		      "noise: bw " + std::to_string(bw) + " gave " +
		      std::to_string(o.residual_evm) + ", previous " + std::to_string(prev));
		prev = o.residual_evm;
	}
	std::cout << "  phase_noise_vs_bandwidth: passed\n";
}

// ---------------------------------------------------------------------------
// The phase accumulator stays bounded over a long run
//
// This is the precision lesson the timing loop's symbol clock carries, in
// another guise: an unbounded phase would grow until its increment fell
// below the ULP and the oscillator stopped turning.
// ---------------------------------------------------------------------------
static void test_phase_stays_wrapped() {
	auto cfg = base_config();
	CarrierRecovery<double> cr(cfg);
	Constellation<double> c(Modulation::qpsk);
	std::mt19937 rng(3);
	std::uniform_int_distribution<std::size_t> pick(0, c.order() - 1);

	double ph = 0.0;
	const double df = 0.05;
	for (std::size_t n = 0; n < 200000; ++n) {
		const auto s = c.symbol(pick(rng));
		const cd rot(std::cos(ph), std::sin(ph));
		const cd x = cd(s.real(), s.imag()) * rot;
		ph += df;
		cr.process(cd(x.real(), x.imag()));
		check(std::abs(cr.phase()) <= pi + 1e-9,
		      "phase left [-pi, pi] at symbol " + std::to_string(n) + ": " +
		      std::to_string(cr.phase()));
	}
	check(cr.is_locked(), "should still be locked after 200k symbols");
	check(std::abs(cr.frequency() - df) < 1e-4,
	      "frequency drifted over a long run: " + std::to_string(cr.frequency()));
	std::cout << "  phase_stays_wrapped: passed (200k symbols)\n";
}

// ---------------------------------------------------------------------------
// Precision sweep: residual phase noise against loop arithmetic precision
// ---------------------------------------------------------------------------
static void test_precision_sweep() {
	using posit32  = sw::universal::posit<32, 2>;
	using posit16  = sw::universal::posit<16, 2>;
	using cfloat32 = sw::universal::cfloat<32, 8, std::uint32_t, true, false, false>;

	auto make = []<typename S>() {
		CarrierRecoveryConfig<S> cfg;
		cfg.loop_bandwidth = S(0.02);
		cfg.damping        = S(0.707);
		cfg.detector       = CarrierDetector::qpsk;
		cfg.max_frequency  = S(0.5);
		cfg.enable_afc     = true;
		cfg.afc_gain       = S(0.01);
		cfg.lock_threshold = S(0.1);
		cfg.lock_average_symbols = S(64);
		return cfg;
	};

	std::cout << "    residual phase noise vs loop precision (Bn*T = 0.02, df = 0.03):\n";
	struct Row { const char* name; Outcome o; };
	std::vector<Row> rows;
	rows.push_back({"double  ", run<double>  (Modulation::qpsk, CarrierDetector::qpsk,
	                                           0.03, 0.2, make.operator()<double>(), 12000, 0.03, 5)});
	rows.push_back({"float   ", run<float>   (Modulation::qpsk, CarrierDetector::qpsk,
	                                           0.03, 0.2, make.operator()<float>(), 12000, 0.03, 5)});
	rows.push_back({"posit32 ", run<posit32> (Modulation::qpsk, CarrierDetector::qpsk,
	                                           0.03, 0.2, make.operator()<posit32>(), 12000, 0.03, 5)});
	rows.push_back({"cfloat32", run<cfloat32>(Modulation::qpsk, CarrierDetector::qpsk,
	                                           0.03, 0.2, make.operator()<cfloat32>(), 12000, 0.03, 5)});
	rows.push_back({"posit16 ", run<posit16> (Modulation::qpsk, CarrierDetector::qpsk,
	                                           0.03, 0.2, make.operator()<posit16>(), 12000, 0.03, 5)});

	for (const auto& r : rows)
		std::cout << "      " << r.name << ": EVM " << std::fixed << std::setprecision(4)
		          << r.o.residual_evm * 100.0 << "%   frequency "
		          << std::setprecision(5) << r.o.frequency
		          << "   locked = " << (r.o.locked ? "yes" : "no") << "\n";

	// Every type must genuinely track: locked, on the right frequency, and
	// leaving a sane constellation. Checking a noise statistic alone would
	// pass a loop that had frozen.
	for (const auto& r : rows) {
		const std::string who(r.name);
		check(r.o.locked, who + " did not lock");
		check(std::abs(r.o.frequency - 0.03) < 2e-3, who + " frequency = " +
		      std::to_string(r.o.frequency) + ", expected 0.03");
		check(r.o.residual_evm < 0.05, who + " residual EVM " +
		      std::to_string(r.o.residual_evm * 100.0) + "%");
	}
	// The 32-bit types sit on the same floor as double, set by channel noise
	// rather than by arithmetic.
	for (std::size_t i = 1; i + 1 < rows.size(); ++i)
		check(std::abs(rows[i].o.residual_evm - rows[0].o.residual_evm) <
		          0.25 * rows[0].o.residual_evm,
		      std::string(rows[i].name) + " should sit on the same EVM floor as double");
	std::cout << "  precision_sweep: passed\n";
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------
static void test_validation() {
	auto bad = [](auto mutate) {
		auto cfg = base_config();
		mutate(cfg);
		bool caught = false;
		try { CarrierRecovery<double> cr(cfg); }
		catch (const std::invalid_argument&) { caught = true; }
		return caught;
	};
	check(bad([](auto& c) { c.loop_bandwidth = 0.0; }), "zero bandwidth should throw");
	check(bad([](auto& c) { c.damping = 0.0; }),        "zero damping should throw");
	check(bad([](auto& c) { c.detector_gain = -1.0; }), "negative Kp should throw");
	check(bad([](auto& c) { c.max_frequency = 0.0; }),  "zero max_frequency should throw");
	check(bad([](auto& c) { c.afc_gain = 0.0; }),       "zero AFC gain should throw when enabled");
	check(bad([](auto& c) { c.lock_threshold = 0.0; }), "zero lock threshold should throw");

	// AFC gain is only required when AFC is on.
	{
		auto cfg = base_config();
		cfg.enable_afc = false;
		cfg.afc_gain = 0.0;
		CarrierRecovery<double> cr(cfg);
		check(cr.invariants_hold(), "disabled AFC should not require a gain");
	}
	// The integrator is bounded by max_frequency, so a hopeless offset rails
	// rather than running away.
	{
		auto cfg = base_config();
		cfg.max_frequency = 0.05;
		auto o = run(Modulation::qpsk, CarrierDetector::qpsk, 0.4, 0.0, cfg, 4000);
		check(std::abs(o.frequency) <= 0.05 + 1e-9,
		      "frequency should be clamped to max_frequency, got " +
		      std::to_string(o.frequency));
	}
	// reset()
	{
		CarrierRecovery<double> cr(base_config());
		for (int n = 0; n < 500; ++n) cr.process(cd(0.7, 0.7));
		cr.reset();
		check(cr.phase() == 0.0 && cr.frequency() == 0.0, "reset should clear the loop");
		check(cr.symbols_processed() == 0, "reset should clear the counter");
		check(!cr.is_locked(), "reset should clear lock");
	}
	std::cout << "  validation: passed\n";
}

// ---------------------------------------------------------------------------

int main() {
	try {
		std::cout << "SDR carrier recovery tests\n";
		test_phase_lock();
		test_afc_extends_pull_in();
		test_constellations();
		test_phase_noise_vs_bandwidth();
		test_phase_stays_wrapped();
		test_precision_sweep();
		test_validation();
		std::cout << "All SDR carrier recovery tests passed.\n";
		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << "\n";
		return 1;
	}
}
