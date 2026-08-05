// test_sdr_agc.cpp: automatic gain control.
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)`.
//
// The class invariants from the design are asserted through the public
// invariants_hold() predicate rather than by assert() inside the header,
// because CI runs Release where NDEBUG strips assertions.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/sdr/agc.hpp>
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
using sw::dsp::complex_for_t;

static void check(bool condition, const std::string& msg) {
	if (!condition) throw std::runtime_error("test failed: " + msg);
}
static bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }

// Default config with time constants expressed in samples (sample_rate = 1).
static AgcConfig<double> base_config() {
	AgcConfig<double> c;
	c.reference_level  = 1.0;
	c.sample_rate_hz   = 1.0;
	c.attack_time_s    = 20.0;    // samples
	c.decay_time_s     = 200.0;
	c.averaging_time_s = 10.0;
	c.min_gain_db      = -80.0;
	c.max_gain_db      =  80.0;
	return c;
}

// ---------------------------------------------------------------------------
// Settling: a constant-amplitude input is driven to the reference level
// ---------------------------------------------------------------------------
static void test_settles_to_reference() {
	for (double amp : {0.01, 0.1, 1.0, 10.0, 100.0}) {
		auto cfg = base_config();
		AutomaticGainControl<double> agc(cfg);

		double last = 0.0;
		for (int n = 0; n < 20000; ++n)
			last = std::abs(agc.process(amp));

		check(agc.invariants_hold(), "invariants after settling at amp " +
		      std::to_string(amp));
		check(near(last, cfg.reference_level, 1e-3),
		      "amp " + std::to_string(amp) + " settled to " + std::to_string(last) +
		      ", expected " + std::to_string(cfg.reference_level));
		// The gain it found must be the one that maps amp -> reference.
		const double want_db = 20.0 * std::log10(cfg.reference_level / amp);
		check(near(agc.gain_db(), want_db, 0.02),
		      "amp " + std::to_string(amp) + " gain " + std::to_string(agc.gain_db()) +
		      " dB, expected " + std::to_string(want_db));
	}

	// A non-unity reference is honoured too.
	{
		auto cfg = base_config();
		cfg.reference_level = 0.25;
		AutomaticGainControl<double> agc(cfg);
		double last = 0.0;
		for (int n = 0; n < 20000; ++n) last = std::abs(agc.process(3.0));
		check(near(last, 0.25, 1e-3), "reference 0.25 settled to " + std::to_string(last));
	}
	std::cout << "  settles_to_reference: passed\n";
}

// ---------------------------------------------------------------------------
// Attack is fast, decay is slow, and the asymmetry is the configured one
// ---------------------------------------------------------------------------
static void test_attack_decay_asymmetry() {
	auto cfg = base_config();
	cfg.attack_time_s = 20.0;
	cfg.decay_time_s  = 400.0;
	cfg.detector      = LevelDetector::magnitude;   // no detector lag to confuse the timing

	// Settle at unity, then step the input UP by 20 dB: the gain must come
	// down, and quickly (attack).
	AutomaticGainControl<double> up(cfg);
	for (int n = 0; n < 20000; ++n) up.process(1.0);
	const double g0 = up.gain_db();
	int n_attack = 0;
	for (; n_attack < 20000; ++n_attack) {
		up.process(10.0);                       // +20 dB
		if (std::abs(up.gain_db() - (g0 - 20.0)) < 1.0) break;
	}
	check(n_attack < 20000, "attack never converged");

	// Same starting point, step the input DOWN by 20 dB: gain rises, slowly.
	AutomaticGainControl<double> down(cfg);
	for (int n = 0; n < 20000; ++n) down.process(1.0);
	const double g1 = down.gain_db();
	int n_decay = 0;
	for (; n_decay < 200000; ++n_decay) {
		down.process(0.1);                      // -20 dB
		if (std::abs(down.gain_db() - (g1 + 20.0)) < 1.0) break;
	}
	check(n_decay < 200000, "decay never converged");

	// The configured ratio is 20:1; allow a wide band since the settling
	// threshold is a crude proxy for a time constant, but the ORDER must be
	// unambiguous.
	const double ratio = static_cast<double>(n_decay) / static_cast<double>(n_attack);
	check(ratio > 5.0, "decay should be much slower than attack, ratio = " +
	      std::to_string(ratio) + " (attack " + std::to_string(n_attack) +
	      " samples, decay " + std::to_string(n_decay) + ")");

	// Swapping the constants swaps the behaviour — proving the asymmetry is
	// driven by configuration and not baked in.
	auto swapped = cfg;
	swapped.attack_time_s = 400.0;
	swapped.decay_time_s  = 20.0;
	AutomaticGainControl<double> sw_agc(swapped);
	for (int n = 0; n < 40000; ++n) sw_agc.process(1.0);
	const double g2 = sw_agc.gain_db();
	int n_slow_attack = 0;
	for (; n_slow_attack < 200000; ++n_slow_attack) {
		sw_agc.process(10.0);
		if (std::abs(sw_agc.gain_db() - (g2 - 20.0)) < 1.0) break;
	}
	check(n_slow_attack > n_attack * 5,
	      "swapping the time constants should make the attack slow, got " +
	      std::to_string(n_slow_attack) + " vs " + std::to_string(n_attack));

	std::cout << "  attack_decay_asymmetry: passed (attack " << n_attack
	          << " samples, decay " << n_decay << ")\n";
}

// ---------------------------------------------------------------------------
// >60 dB dynamic range, no overflow, no NaN — the point of the log domain
// ---------------------------------------------------------------------------
static void test_wide_dynamic_range() {
	auto cfg = base_config();
	// The sweep below spans 320 dB of input, so the gain range has to cover
	// +/-160 dB for the loop to reach the reference at both ends. Anything
	// narrower rails, which is correct behaviour but tests the clamp rather
	// than the dynamic range.
	cfg.min_gain_db = -200.0;
	cfg.max_gain_db =  200.0;
	AutomaticGainControl<double> agc(cfg);

	// Sweep the input across 320 dB in 20 dB steps.
	for (int e = -8; e <= 8; ++e) {
		const double amp = std::pow(10.0, e);          // 1e-8 .. 1e8
		double last = 0.0;
		for (int n = 0; n < 60000; ++n) last = std::abs(agc.process(amp));
		check(std::isfinite(last), "output not finite at amp 1e" + std::to_string(e));
		check(std::isfinite(agc.gain_db()), "gain not finite at amp 1e" + std::to_string(e));
		check(agc.invariants_hold(), "invariants at amp 1e" + std::to_string(e));
		check(near(last, 1.0, 1e-2), "amp 1e" + std::to_string(e) +
		      " settled to " + std::to_string(last));
	}

	// Silence must not poison the state: log(0) would be -inf, so the loop
	// floors the level. The gain rails at max and everything stays finite.
	{
		AutomaticGainControl<double> quiet(cfg);
		for (int n = 0; n < 5000; ++n) {
			const double y = quiet.process(0.0);
			check(std::isfinite(y), "silence produced a non-finite output");
		}
		check(std::isfinite(quiet.gain_db()), "silence produced a non-finite gain");
		check(quiet.invariants_hold(), "invariants after silence");
		check(near(quiet.gain_db(), cfg.max_gain_db, 1e-6),
		      "silence should drive the gain to its ceiling, got " +
		      std::to_string(quiet.gain_db()));
		// And it recovers once signal returns.
		double last = 0.0;
		for (int n = 0; n < 60000; ++n) last = std::abs(quiet.process(1.0));
		check(near(last, 1.0, 1e-2), "did not recover after silence, got " +
		      std::to_string(last));
	}
	std::cout << "  wide_dynamic_range: passed\n";
}

// ---------------------------------------------------------------------------
// Gain limits are respected, from both directions and via set_gain_db()
// ---------------------------------------------------------------------------
static void test_gain_limits() {
	auto cfg = base_config();
	cfg.min_gain_db = -6.0;
	cfg.max_gain_db = 12.0;

	// A very weak signal wants far more than 12 dB; the loop must rail.
	{
		AutomaticGainControl<double> agc(cfg);
		for (int n = 0; n < 50000; ++n) agc.process(1e-6);
		check(near(agc.gain_db(), 12.0, 1e-6),
		      "gain should rail at max, got " + std::to_string(agc.gain_db()));
		check(agc.invariants_hold(), "invariants at the max rail");
	}
	// A very strong signal wants far less than -6 dB.
	{
		AutomaticGainControl<double> agc(cfg);
		for (int n = 0; n < 50000; ++n) agc.process(1e6);
		check(near(agc.gain_db(), -6.0, 1e-6),
		      "gain should rail at min, got " + std::to_string(agc.gain_db()));
		check(agc.invariants_hold(), "invariants at the min rail");
	}
	// set_gain_db() clamps rather than breaking I1 from outside.
	{
		AutomaticGainControl<double> agc(cfg);
		agc.set_gain_db(1000.0);
		check(near(agc.gain_db(), 12.0, 1e-9), "set_gain_db above max must clamp");
		check(agc.invariants_hold(), "invariants after clamping high");
		agc.set_gain_db(-1000.0);
		check(near(agc.gain_db(), -6.0, 1e-9), "set_gain_db below min must clamp");
		check(agc.invariants_hold(), "invariants after clamping low");
	}
	std::cout << "  gain_limits: passed\n";
}

// ---------------------------------------------------------------------------
// Detector choice: RMS smooths across a constellation, magnitude does not
// ---------------------------------------------------------------------------
static void test_detectors() {
	Constellation<double> c(Modulation::qam16);
	std::mt19937 rng(5);
	std::uniform_int_distribution<std::size_t> pick(0, c.order() - 1);

	// A 16-QAM stream has a spread of symbol magnitudes, so an
	// instantaneous detector makes the gain chase each symbol while an RMS
	// detector regulates the average.
	auto run = [&](LevelDetector d) {
		auto cfg = base_config();
		cfg.detector         = d;
		cfg.attack_time_s    = 50.0;
		cfg.decay_time_s     = 50.0;
		cfg.averaging_time_s = 50.0;
		AutomaticGainControl<double, std::complex<double>> agc(cfg);

		std::mt19937 local(7);
		std::uniform_int_distribution<std::size_t> p(0, c.order() - 1);
		// Settle.
		for (int n = 0; n < 20000; ++n) {
			const auto s = c.symbol(p(local));
			agc.process(std::complex<double>(s.real(), s.imag()));
		}
		// Then measure how much the gain still moves.
		double lo = 1e30, hi = -1e30;
		for (int n = 0; n < 5000; ++n) {
			const auto s = c.symbol(p(local));
			agc.process(std::complex<double>(s.real(), s.imag()));
			lo = std::min(lo, agc.gain_db());
			hi = std::max(hi, agc.gain_db());
		}
		return hi - lo;
	};

	const double ripple_rms = run(LevelDetector::rms);
	const double ripple_mag = run(LevelDetector::magnitude);
	check(ripple_rms < ripple_mag,
	      "RMS detection should ripple less than instantaneous magnitude on "
	      "QAM (" + std::to_string(ripple_rms) + " vs " +
	      std::to_string(ripple_mag) + " dB)");
	std::cout << "  detectors: passed (gain ripple: rms " << std::fixed
	          << std::setprecision(3) << ripple_rms << " dB, magnitude "
	          << ripple_mag << " dB)\n";
}

// ---------------------------------------------------------------------------
// Complex I/Q: the gain is real, the samples are not, and the phase survives
// ---------------------------------------------------------------------------
static void test_complex_samples() {
	auto cfg = base_config();
	AutomaticGainControl<double, std::complex<double>> agc(cfg);

	const double amp = 0.02;
	std::complex<double> last;
	for (int n = 0; n < 30000; ++n) {
		const double ph = sw::dsp::two_pi * 0.013 * static_cast<double>(n);
		last = agc.process(std::complex<double>(amp * std::cos(ph), amp * std::sin(ph)));
	}
	check(near(std::abs(last), 1.0, 1e-2),
	      "complex input settled to |y| = " + std::to_string(std::abs(last)));

	// Scaling by a real gain cannot rotate the sample.
	const double ph = sw::dsp::two_pi * 0.013 * 30000.0;
	const std::complex<double> in(amp * std::cos(ph), amp * std::sin(ph));
	const std::complex<double> out = agc.process(in);
	check(near(std::arg(out), std::arg(in), 1e-12),
	      "AGC must not rotate the sample: arg in " + std::to_string(std::arg(in)) +
	      ", arg out " + std::to_string(std::arg(out)));

	// It also runs with the library's universal-aware complex alias.
	{
		using cp32 = complex_for_t<sw::universal::posit<32, 2>>;
		AgcConfig<double> c2 = base_config();
		AutomaticGainControl<double, cp32> agc2(c2);
		cp32 y;
		for (int n = 0; n < 20000; ++n) y = agc2.process(cp32(0.05, 0.0));
		check(near(static_cast<double>(y.real()), 1.0, 2e-2),
		      "posit complex settled to " + std::to_string(static_cast<double>(y.real())));
	}
	std::cout << "  complex_samples: passed\n";
}

// ---------------------------------------------------------------------------
// Block processing matches sample-at-a-time
// ---------------------------------------------------------------------------
static void test_block_matches_scalar() {
	auto cfg = base_config();
	AutomaticGainControl<double> a(cfg), b(cfg);

	std::vector<double> in(2000);
	std::mt19937 rng(3);
	std::uniform_real_distribution<double> u(-0.3, 0.3);
	for (auto& v : in) v = u(rng);

	auto blocked = b.process_block(std::span<const double>(in));
	for (std::size_t i = 0; i < in.size(); ++i) {
		const double one = a.process(in[i]);
		check(near(one, blocked[i], 1e-15),
		      "block and scalar disagree at " + std::to_string(i));
	}
	check(near(a.gain_db(), b.gain_db(), 1e-12), "final gains disagree");

	// reset() really does restore the initial state.
	a.reset();
	check(near(a.gain_db(), cfg.initial_gain_db, 1e-12), "reset should restore gain");
	check(a.level() == 0.0, "reset should clear the detector");
	check(a.invariants_hold(), "invariants after reset");
	std::cout << "  block_matches_scalar: passed\n";
}

// ---------------------------------------------------------------------------
// Precision sweep: residual amplitude error against gain-state precision
//
// The mechanism is worth stating, because it is not "narrow types are just
// noisier". The loop step is log_gain += rate * error. As the loop
// approaches its target the error shrinks, and once rate*error falls below
// half a ULP of the gain state the update rounds to nothing and THE LOOP
// STALLS — permanently, at whatever offset it had reached.
//
// So the residual is not set by precision alone but by precision AND loop
// rate together: a slower loop (larger time constant, smaller rate) stalls
// further from target, in direct proportion. Measured, input 0.05:
//
//   tau       double     posit16
//     10    1.9e-15     3.4e-03
//    200    4.0e-14     8.6e-02
//   1000    2.2e-13     3.9e-01
//
// double is at machine epsilon regardless of rate; posit16 tracks tau almost
// exactly. That proportionality is what this test pins, because it is the
// mechanism — a fixed threshold would pass for the wrong reason the moment
// somebody retuned the loop.
// ---------------------------------------------------------------------------
template <typename State>
static double residual_error(const char* name, double tau, bool print = true) {
	AgcConfig<State> cfg;
	cfg.reference_level  = State(1);
	cfg.sample_rate_hz   = State(1);
	cfg.attack_time_s    = State(tau);
	cfg.decay_time_s     = State(tau);
	cfg.averaging_time_s = State(10);
	cfg.min_gain_db      = State(-80);
	cfg.max_gain_db      = State(80);

	AutomaticGainControl<State> agc(cfg);
	const State amp = State(0.05);
	for (int n = 0; n < 80000; ++n) agc.process(amp);

	double worst = 0.0;
	for (int n = 0; n < 4000; ++n) {
		const double y = static_cast<double>(agc.process(amp));
		worst = std::max(worst, std::abs(y - 1.0));
	}
	if (print) {
		std::cout << "      " << name << ": residual = " << std::scientific
		          << std::setprecision(3) << worst << "   (gain "
		          << std::fixed << std::setprecision(3)
		          << static_cast<double>(agc.gain_db()) << " dB, want 26.021)\n";
	}
	check(agc.invariants_hold(), std::string(name) + " invariants");
	return worst;
}

static void test_precision_sweep() {
	using posit32  = sw::universal::posit<32, 2>;
	using posit16  = sw::universal::posit<16, 2>;
	using cfloat32 = sw::universal::cfloat<32, 8, std::uint32_t, true, false, false>;

	std::cout << "    residual vs gain-state precision (tau = 200 samples):\n";
	const double d   = residual_error<double>  ("double  ", 200.0);
	const double f   = residual_error<float>   ("float   ", 200.0);
	const double p32 = residual_error<posit32> ("posit32 ", 200.0);
	const double c32 = residual_error<cfloat32>("cfloat32", 200.0);
	const double p16 = residual_error<posit16> ("posit16 ", 200.0);

	// double closes the loop to machine precision.
	check(d < 1e-9, "double residual too large: " + std::to_string(d));
	// The 32-bit types hold the level far tighter than any link needs —
	// well under a thousandth of a dB.
	check(f   < 1e-3, "float residual too large: "    + std::to_string(f));
	check(p32 < 1e-3, "posit32 residual too large: "  + std::to_string(p32));
	check(c32 < 1e-3, "cfloat32 residual too large: " + std::to_string(c32));
	// posit16 is where the gain state starts to quantize the loop.
	check(p16 > 100.0 * d, "posit16 should be visibly worse than double");

	// The mechanism: stall offset is proportional to the loop time constant.
	// A 20x slower loop should stall roughly 20x further out for posit16,
	// while double is unaffected.
	std::cout << "    stall scales with loop rate (posit16):\n";
	const double p16_fast = residual_error<posit16>("posit16 tau=10 ", 10.0);
	const double p16_slow = residual_error<posit16>("posit16 tau=200", 200.0, false);
	const double d_fast   = residual_error<double> ("double  tau=10 ", 10.0);

	check(p16_slow > 5.0 * p16_fast,
	      "posit16 residual should grow with the loop time constant: tau=10 gave " +
	      std::to_string(p16_fast) + ", tau=200 gave " + std::to_string(p16_slow));
	check(d_fast < 1e-9 && d < 1e-9,
	      "double should be at machine precision at both loop rates");
	// A fast loop keeps even posit16 inside a hundredth of a dB.
	check(p16_fast < 1e-2, "posit16 with a fast loop should still be tight, got " +
	      std::to_string(p16_fast));

	std::cout << "  precision_sweep: passed\n";
}

// ---------------------------------------------------------------------------
// Configuration is validated
// ---------------------------------------------------------------------------
static void test_validation() {
	auto bad = [](auto mutate) {
		auto cfg = base_config();
		mutate(cfg);
		bool caught = false;
		try { AutomaticGainControl<double> agc(cfg); }
		catch (const std::invalid_argument&) { caught = true; }
		return caught;
	};

	check(bad([](auto& c) { c.reference_level = 0.0; }),  "zero reference should throw");
	check(bad([](auto& c) { c.reference_level = -1.0; }), "negative reference should throw");
	check(bad([](auto& c) { c.attack_time_s = 0.0; }),    "zero attack should throw");
	check(bad([](auto& c) { c.decay_time_s = -1.0; }),    "negative decay should throw");
	check(bad([](auto& c) { c.averaging_time_s = 0.0; }), "zero averaging should throw");
	check(bad([](auto& c) { c.sample_rate_hz = 0.0; }),   "zero sample rate should throw");
	check(bad([](auto& c) { c.min_gain_db = 10.0; c.max_gain_db = -10.0; }),
	      "min > max should throw");

	// min == max is legal: a fixed-gain stage.
	{
		auto cfg = base_config();
		cfg.min_gain_db = cfg.max_gain_db = 3.0;
		AutomaticGainControl<double> agc(cfg);
		for (int n = 0; n < 1000; ++n) agc.process(0.001);
		check(near(agc.gain_db(), 3.0, 1e-9), "pinned gain should stay put");
		check(agc.invariants_hold(), "invariants with a pinned gain");
	}
	std::cout << "  validation: passed\n";
}

// ---------------------------------------------------------------------------

int main() {
	try {
		std::cout << "SDR AGC tests\n";
		test_settles_to_reference();
		test_attack_decay_asymmetry();
		test_wide_dynamic_range();
		test_gain_limits();
		test_detectors();
		test_complex_samples();
		test_block_matches_scalar();
		test_precision_sweep();
		test_validation();
		std::cout << "All SDR AGC tests passed.\n";
		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << "\n";
		return 1;
	}
}
