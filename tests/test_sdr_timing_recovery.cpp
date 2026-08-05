// test_sdr_timing_recovery.cpp: symbol timing recovery.
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)`.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/sdr/timing_recovery.hpp>
#include <sw/dsp/sdr/rrc.hpp>
#include <sw/dsp/sdr/constellation.hpp>
#include <sw/dsp/filter/fir/polyphase.hpp>

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

static void check(bool condition, const std::string& msg) {
	if (!condition) throw std::runtime_error("test failed: " + msg);
}

// ---------------------------------------------------------------------------
// Test bench: BPSK through an RRC pair at 2 samples/symbol, with a settable
// fractional timing offset and clock-rate error.
// ---------------------------------------------------------------------------
namespace bench {

constexpr std::size_t sps  = 2;
constexpr std::size_t span = 10;

// Cubic interpolation of a waveform at an arbitrary real position.
static double sample_at(const std::vector<double>& x, double t) {
	if (t < 1.0) t = 1.0;
	if (t > static_cast<double>(x.size()) - 3.0) t = static_cast<double>(x.size()) - 3.0;
	const std::size_t i = static_cast<std::size_t>(t);
	const double m = t - static_cast<double>(i);
	const double s[4] = {x[i - 1], x[i], x[i + 1], x[i + 2]};
	const double h0 = -m * (m - 1) * (m - 2) / 6.0;
	const double h1 =  (m + 1) * (m - 1) * (m - 2) / 2.0;
	const double h2 = -(m + 1) * m * (m - 2) / 2.0;
	const double h3 =  (m + 1) * m * (m - 1) / 6.0;
	return h0 * s[0] + h1 * s[1] + h2 * s[2] + h3 * s[3];
}

struct Signal {
	std::vector<double> rx;        // matched-filtered waveform
	std::vector<double> symbols;   // transmitted +/-1
	double group_delay;            // TX+RX delay in samples
};

// `offset_symbols` shifts the sampling phase; `rate_error` scales the symbol
// period (1e-4 = 100 ppm fast clock), which is what forces the integrator to
// do real work rather than just parking at nominal.
//
// The waveform is shaped at `os` times the receiver's rate and only then
// decimated to 2 samples/symbol. Applying the offset directly at 2 sps would
// mean cubic-interpolating a signal sitting close to Nyquist, where a cubic
// is a poor interpolator: that alone closed the eye from 0.99 to 0.78 and
// looked exactly like a timing loop failing to converge. Oversampling first
// keeps the bench's own error near 1e-4, so what the tests measure is the
// loop rather than the harness.
static Signal make(std::size_t num_symbols, double alpha, double offset_symbols,
                   double rate_error, unsigned seed, double noise_sigma = 0.0) {
	constexpr std::size_t os = 8;                 // extra oversampling
	constexpr std::size_t fine_sps = sps * os;    // 16 samples/symbol
	const std::size_t fine_taps = span * fine_sps + 1;

	auto h = rrc_filter<double>(fine_taps, fine_sps, alpha);
	std::mt19937 rng(seed);
	std::bernoulli_distribution coin(0.5);
	std::normal_distribution<double> gauss(0.0, noise_sigma);

	mtl::vec::dense_vector<double> sym(num_symbols);
	std::vector<double> tx(num_symbols);
	for (std::size_t i = 0; i < num_symbols; ++i) {
		tx[i] = coin(rng) ? 1.0 : -1.0;
		sym[i] = tx[i];
	}

	sw::dsp::PolyphaseInterpolator<double> up(h, fine_sps);
	auto wave = up.process_block(std::span<const double>(sym.data(), sym.size()));
	sw::dsp::FIRFilter<double> mf(h);
	std::vector<double> clean(wave.size());
	for (std::size_t i = 0; i < wave.size(); ++i) clean[i] = mf.process(wave[i]);

	// Decimate to the receiver's 2 sps, advancing os*(1+rate_error) fine
	// samples per output and starting offset_symbols into the first symbol.
	Signal s;
	s.symbols = tx;
	s.group_delay = static_cast<double>(fine_taps - 1) / static_cast<double>(os);
	const double step = static_cast<double>(os) * (1.0 + rate_error);
	const double start = 2.0 + offset_symbols * static_cast<double>(fine_sps);
	const std::size_t n_out = static_cast<std::size_t>(
		(static_cast<double>(clean.size()) - 4.0 - start) / step);
	s.rx.resize(n_out);
	for (std::size_t n = 0; n < n_out; ++n)
		s.rx[n] = sample_at(clean, start + static_cast<double>(n) * step) + gauss(rng);
	return s;
}

} // namespace bench

// Run the loop over a signal and report what came out.
struct RunResult {
	std::size_t symbols = 0;
	std::size_t locked_at = 0;        // symbol index where lock was declared
	std::size_t converged_at = 0;     // first symbol after which the eye stays open
	bool        locked = false;
	double      final_omega = 0.0;
	double      min_abs_late = 0.0;   // smallest |y| over the settled tail
	double      mu_jitter = 0.0;      // std of mu over the settled tail
	std::vector<double> out;
};

template <typename State>
static RunResult run(const bench::Signal& sig, const TimingRecoveryConfig<State>& cfg) {
	TimingRecovery<State> tr(cfg);
	RunResult r;
	std::vector<double> mus;
	for (double v : sig.rx) {
		auto [ready, y] = tr.process(static_cast<State>(v));
		if (!ready) continue;
		++r.symbols;
		r.out.push_back(static_cast<double>(y));
		mus.push_back(static_cast<double>(tr.mu()));
		if (!r.locked && tr.is_locked()) { r.locked = true; r.locked_at = r.symbols; }
	}
	r.final_omega = static_cast<double>(tr.omega());

	const std::size_t tail = r.out.size() / 4;      // last quarter = settled
	const std::size_t first = r.out.size() - tail;
	r.min_abs_late = 1e30;
	for (std::size_t i = first; i < r.out.size(); ++i)
		r.min_abs_late = std::min(r.min_abs_late, std::abs(r.out[i]));

	// mu wraps at 1, so measure jitter on the unwrapped sequence.
	double mean = 0.0;
	std::vector<double> unwrapped;
	double acc = 0.0, prev = mus[first];
	for (std::size_t i = first; i < mus.size(); ++i) {
		double d = mus[i] - prev;
		if (d >  0.5) d -= 1.0;
		if (d < -0.5) d += 1.0;
		acc += d;
		unwrapped.push_back(acc);
		prev = mus[i];
	}
	for (double v : unwrapped) mean += v;
	mean /= static_cast<double>(unwrapped.size());
	double var = 0.0;
	for (double v : unwrapped) var += (v - mean) * (v - mean);
	r.mu_jitter = std::sqrt(var / static_cast<double>(unwrapped.size()));

	// Convergence time, measured from the signal rather than from the lock
	// flag. is_locked() cannot fire before its own averaging window has
	// filled, which floors it at a few hundred symbols and hides any
	// dependence on loop bandwidth — the very thing the bandwidth test is
	// trying to observe. This walks back from the end to the last symbol
	// whose eye was closed.
	r.converged_at = r.out.size();
	for (std::size_t i = r.out.size(); i-- > 0; ) {
		if (std::abs(r.out[i]) < 0.7) { r.converged_at = i + 1; break; }
	}
	return r;
}

static TimingRecoveryConfig<double> base_config() {
	TimingRecoveryConfig<double> c;
	c.samples_per_symbol = 2.0;
	c.loop_bandwidth     = 0.01;
	c.damping            = 0.707;
	c.detector           = TimingDetector::gardner;
	c.max_deviation      = 0.05;
	c.lock_threshold     = 0.05;
	c.lock_average_symbols = 64.0;
	return c;
}

// ---------------------------------------------------------------------------
// Locks from any initial timing offset
// ---------------------------------------------------------------------------
static void test_locks_from_any_offset() {
	for (double offset : {0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9}) {
		auto sig = bench::make(3000, 0.35, offset, 0.0, 1);
		auto r = run(sig, base_config());

		check(r.locked, "offset " + std::to_string(offset) + ": never locked");
		// A closed eye reads as a small |y|; an open one sits near 1.
		check(r.min_abs_late > 0.8,
		      "offset " + std::to_string(offset) + ": eye did not open, min|y| = " +
		      std::to_string(r.min_abs_late));
		// No clock error, so the rate estimate should stay at nominal.
		check(std::abs(r.final_omega - 2.0) < 0.01,
		      "offset " + std::to_string(offset) + ": omega drifted to " +
		      std::to_string(r.final_omega));
	}
	std::cout << "  locks_from_any_offset: passed\n";
}

// ---------------------------------------------------------------------------
// Tracks a clock-rate error — the integrator's job
// ---------------------------------------------------------------------------
static void test_tracks_clock_rate_error() {
	for (double ppm : {-2000.0, -500.0, 500.0, 2000.0}) {
		const double rate = ppm * 1e-6;
		auto sig = bench::make(6000, 0.35, 0.2, rate, 2);
		auto cfg = base_config();
		cfg.loop_bandwidth = 0.02;
		auto r = run(sig, cfg);

		check(r.locked, std::to_string(ppm) + " ppm: never locked");
		check(r.min_abs_late > 0.75, std::to_string(ppm) +
		      " ppm: eye did not open, min|y| = " + std::to_string(r.min_abs_late));
		// The integrator must have absorbed the rate error. Note the
		// direction: the bench scales the RECEIVER'S SAMPLE PERIOD by
		// (1+rate), so a symbol spans 2/(1+rate) receiver samples — a fast
		// receiver clock takes MORE samples per symbol, not fewer.
		const double want = 2.0 / (1.0 + rate);
		check(std::abs(r.final_omega - want) < 0.002,
		      std::to_string(ppm) + " ppm: omega = " + std::to_string(r.final_omega) +
		      ", expected " + std::to_string(want));
	}
	std::cout << "  tracks_clock_rate_error: passed\n";
}

// ---------------------------------------------------------------------------
// Symbols come out right: a noiseless link makes no decision errors
// ---------------------------------------------------------------------------
static void test_symbol_recovery() {
	auto sig = bench::make(3000, 0.35, 0.33, 0.0, 3);
	auto r = run(sig, base_config());
	check(r.locked, "did not lock");

	// Align the recovered stream against the transmitted one by searching a
	// small lag window — the loop's group delay is not known a priori.
	std::size_t best_lag = 0, best_matches = 0;
	const std::size_t window = r.out.size() / 2;
	for (std::size_t lag = 0; lag < 40 && lag + window < r.out.size(); ++lag) {
		std::size_t matches = 0;
		for (std::size_t i = 0; i < window; ++i) {
			const double y = r.out[r.out.size() - window + i];
			const std::size_t ti = sig.symbols.size() - window + i - lag;
			if (ti < sig.symbols.size() &&
			    ((y > 0.0) == (sig.symbols[ti] > 0.0))) ++matches;
		}
		if (matches > best_matches) { best_matches = matches; best_lag = lag; }
	}
	const double accuracy = static_cast<double>(best_matches) /
	                        static_cast<double>(window);
	check(accuracy > 0.999, "symbol accuracy " + std::to_string(accuracy) +
	      " at lag " + std::to_string(best_lag) + ", expected > 0.999");
	std::cout << "  symbol_recovery: passed (" << std::fixed << std::setprecision(4)
	          << accuracy * 100.0 << "% at lag " << best_lag << ")\n";
}

// ---------------------------------------------------------------------------
// Acquisition time and jitter scale with loop bandwidth, in opposite
// directions — the classic loop trade, and both halves of the acceptance
// criterion.
//
// Acquisition is measured as the number of symbols the INTEGRATOR needs to
// absorb a clock-rate error, not as the time for the eye to open. The eye
// can open almost immediately if the loop happens to start near the right
// phase, which makes it useless as a bandwidth-sensitive measure; pulling
// omega from nominal to the true rate is unambiguous work that a narrow loop
// genuinely takes longer to do.
// ---------------------------------------------------------------------------
static void test_bandwidth_tradeoff() {
	const double rate = 1500e-6;
	const double want_omega = 2.0 / (1.0 + rate);

	std::cout << "    bandwidth   omega@90%   mu jitter\n";
	std::vector<std::pair<double, std::pair<std::size_t, double>>> results;

	for (double bw : {0.002, 0.005, 0.02, 0.05}) {
		auto sig = bench::make(12000, 0.35, 0.2, rate, 4, 0.02);
		TimingRecoveryConfig<double> cfg = base_config();
		cfg.loop_bandwidth = bw;

		TimingRecovery<double> tr(cfg);
		std::size_t symbols = 0, reached = 0;
		std::vector<double> mus, outs;
		// 90% of the way from nominal to the true rate.
		const double target = 2.0 + 0.9 * (want_omega - 2.0);
		for (double v : sig.rx) {
			auto [ready, y] = tr.process(v);
			if (!ready) continue;
			++symbols;
			outs.push_back(y);
			mus.push_back(tr.mu());
			if (reached == 0 && std::abs(tr.omega() - 2.0) >= std::abs(target - 2.0))
				reached = symbols;
		}
		check(reached > 0, "bw " + std::to_string(bw) +
		      ": omega never reached 90% of the rate error");

		// Jitter is measured on a SEPARATE run with no rate error. Under a
		// rate error mu sweeps continuously through [0,1) and wraps, so its
		// standard deviation converges to that of a uniform distribution —
		// 0.289 regardless of loop bandwidth, which measures the wrap and
		// nothing else. With the clock matched, mu is stationary and its
		// spread is the timing jitter the loop actually contributes.
		auto quiet = bench::make(12000, 0.35, 0.2, 0.0, 4, 0.02);
		const double jitter = run(quiet, cfg).mu_jitter;

		std::cout << "    " << std::setw(9) << std::fixed << std::setprecision(3) << bw
		          << "   " << std::setw(9) << reached
		          << "   " << std::scientific << std::setprecision(3) << jitter << "\n";
		results.emplace_back(bw, std::make_pair(reached, jitter));
	}

	// Acquisition gets faster as bandwidth widens, monotonically.
	for (std::size_t i = 1; i < results.size(); ++i) {
		check(results[i].second.first < results[i - 1].second.first,
		      "acquisition should speed up with bandwidth: bw " +
		      std::to_string(results[i].first) + " took " +
		      std::to_string(results[i].second.first) + " symbols, bw " +
		      std::to_string(results[i - 1].first) + " took " +
		      std::to_string(results[i - 1].second.first));
	}
	// Jitter gets worse as bandwidth widens, monotonically — the other half.
	for (std::size_t i = 1; i < results.size(); ++i) {
		check(results[i].second.second > results[i - 1].second.second,
		      "jitter should grow with bandwidth: bw " +
		      std::to_string(results[i].first) + " gave " +
		      std::to_string(results[i].second.second) + ", bw " +
		      std::to_string(results[i - 1].first) + " gave " +
		      std::to_string(results[i - 1].second.second));
	}
	std::cout << "  bandwidth_tradeoff: passed\n";
}

// ---------------------------------------------------------------------------
// Mueller-Muller works at 1 sample/symbol, where Gardner cannot
// ---------------------------------------------------------------------------
static void test_mueller_muller() {
	// M&M is decision-directed, so it needs a roughly correct constellation
	// — it is run here at 2 sps on the same bench, which is its easy case.
	auto sig = bench::make(4000, 0.35, 0.3, 0.0, 6);
	auto cfg = base_config();
	cfg.detector = TimingDetector::mueller_muller;
	cfg.loop_bandwidth = 0.005;
	auto r = run(sig, cfg);
	check(r.locked, "Mueller-Muller did not lock");
	check(r.min_abs_late > 0.7, "Mueller-Muller eye did not open, min|y| = " +
	      std::to_string(r.min_abs_late));

	// Gardner must refuse 1 sample/symbol: it has no midpoint to read.
	bool caught = false;
	try {
		auto bad = base_config();
		bad.samples_per_symbol = 1.0;
		bad.detector = TimingDetector::gardner;
		TimingRecovery<double> tr(bad);
	} catch (const std::invalid_argument&) { caught = true; }
	check(caught, "Gardner at 1 sample/symbol should throw");

	// Mueller-Muller accepts it.
	{
		auto ok = base_config();
		ok.samples_per_symbol = 1.0;
		ok.detector = TimingDetector::mueller_muller;
		TimingRecovery<double> tr(ok);
		check(tr.invariants_hold(), "M&M at 1 sps should construct cleanly");
	}
	std::cout << "  mueller_muller: passed\n";
}

// ---------------------------------------------------------------------------
// Complex I/Q
// ---------------------------------------------------------------------------
static void test_complex_qpsk() {
	using cd = std::complex<double>;
	// Shaped at 16 samples/symbol and decimated to 2, for the same reason
	// bench::make() does: interpolating a 2 sps signal to apply the offset
	// would inject more error than the loop itself contributes.
	constexpr std::size_t os = 8, fine_sps = bench::sps * os;
	const std::size_t fine_taps = bench::span * fine_sps + 1;
	const std::size_t nsym = 4000;

	auto h = rrc_filter<double>(fine_taps, fine_sps, 0.35);
	Constellation<double> c(Modulation::qpsk);

	std::mt19937 rng(9);
	std::uniform_int_distribution<std::size_t> pick(0, c.order() - 1);
	mtl::vec::dense_vector<double> si(nsym), sq(nsym);
	for (std::size_t i = 0; i < nsym; ++i) {
		const auto s = c.symbol(pick(rng));
		si[i] = s.real();
		sq[i] = s.imag();
	}
	auto shape = [&](const mtl::vec::dense_vector<double>& in) {
		sw::dsp::PolyphaseInterpolator<double> up(h, fine_sps);
		auto w = up.process_block(std::span<const double>(in.data(), in.size()));
		sw::dsp::FIRFilter<double> mf(h);
		std::vector<double> out(w.size());
		for (std::size_t i = 0; i < w.size(); ++i) out[i] = mf.process(w[i]);
		return out;
	};
	auto wi = shape(si), wq = shape(sq);

	TimingRecovery<double, cd> tr(base_config());
	std::vector<cd> out;
	const double start = 2.0 + 0.35 * static_cast<double>(fine_sps);
	const std::size_t n_out = static_cast<std::size_t>(
		(static_cast<double>(wi.size()) - 4.0 - start) / static_cast<double>(os));
	for (std::size_t n = 0; n < n_out; ++n) {
		const double t = start + static_cast<double>(n * os);
		const cd x(bench::sample_at(wi, t), bench::sample_at(wq, t));
		auto [ready, y] = tr.process(x);
		if (ready) out.push_back(y);
	}
	check(tr.is_locked(), "complex QPSK did not lock");
	check(tr.invariants_hold(), "invariants after complex run");

	// Settled symbols must sit near the QPSK ring of radius 1.
	double worst = 0.0;
	for (std::size_t i = out.size() * 3 / 4; i < out.size(); ++i)
		worst = std::max(worst, std::abs(std::abs(out[i]) - 1.0));
	check(worst < 0.05, "QPSK magnitudes off the unit circle by " +
	      std::to_string(worst));
	std::cout << "  complex_qpsk: passed (worst radius error " << std::fixed
	          << std::setprecision(4) << worst << ")\n";
}

// ---------------------------------------------------------------------------
// Precision sweep: timing jitter against loop-filter arithmetic precision
//
// The assertions here check that each type actually TRACKS — omega lands on
// the true samples-per-symbol and the eye is open — not merely that some
// jitter number came out small. An earlier version of this test compared
// jitter alone and passed posit16 with flying colours while the loop had in
// fact railed at its deviation clamp and frozen solid: a frozen loop has
// exactly zero jitter. Measuring the response rather than a statistic of it
// is the difference.
// ---------------------------------------------------------------------------
struct PrecisionResult { double jitter; double omega; double eye; bool locked; };

template <typename State>
static PrecisionResult precision_for(const char* name) {
	auto sig = bench::make(8000, 0.35, 0.4, 200e-6, 7, 0.02);
	TimingRecoveryConfig<State> cfg;
	cfg.samples_per_symbol = State(2);
	cfg.loop_bandwidth     = State(0.01);
	cfg.damping            = State(0.707);
	cfg.detector           = TimingDetector::gardner;
	cfg.max_deviation      = State(0.05);
	cfg.lock_threshold     = State(0.05);
	cfg.lock_average_symbols = State(64);

	auto r = run(sig, cfg);
	std::cout << "      " << name << ": jitter = " << std::scientific
	          << std::setprecision(3) << r.mu_jitter
	          << "   omega = " << std::fixed << std::setprecision(6) << r.final_omega
	          << "   min|y| = " << std::setprecision(4) << r.min_abs_late
	          << "   locked = " << (r.locked ? "yes" : "no") << "\n";
	return {r.mu_jitter, r.final_omega, r.min_abs_late, r.locked};
}

static void test_precision_sweep() {
	using posit32  = sw::universal::posit<32, 2>;
	using posit16  = sw::universal::posit<16, 2>;
	using cfloat32 = sw::universal::cfloat<32, 8, std::uint32_t, true, false, false>;

	// The bench runs 200 ppm fast, so the loop should settle at 2/(1+200e-6).
	const double want_omega = 2.0 / (1.0 + 200e-6);

	std::cout << "    timing jitter vs loop-filter precision (Bn*T = 0.01, 200 ppm):\n";
	const auto d   = precision_for<double>  ("double  ");
	const auto f   = precision_for<float>   ("float   ");
	const auto p32 = precision_for<posit32> ("posit32 ");
	const auto c32 = precision_for<cfloat32>("cfloat32");
	const auto p16 = precision_for<posit16> ("posit16 ");

	for (auto [name, r] : {std::pair<const char*, PrecisionResult>{"double", d},
	                        {"float", f}, {"posit32", p32},
	                        {"cfloat32", c32}, {"posit16", p16}}) {
		const std::string who(name);
		check(r.locked, who + " did not lock");
		// Actually tracking, not railed at the clamp: a railed loop sits at
		// 2*(1 -/+ 0.05) = 1.9 or 2.1.
		check(std::abs(r.omega - want_omega) < 0.005,
		      who + " omega = " + std::to_string(r.omega) + ", expected " +
		      std::to_string(want_omega) + " (railed loops sit at 1.9 or 2.1)");
		// Eye open: a frozen or mistracking loop closes it completely.
		check(r.eye > 0.7, who + " eye did not open, min|y| = " +
		      std::to_string(r.eye));
		check(std::isfinite(r.jitter) && r.jitter > 0.0,
		      who + " jitter is not a finite positive number — a frozen loop "
		      "reports exactly zero");
	}

	// Every type from posit16 up sits on the same jitter floor, which at this
	// loop bandwidth is set by detector self-noise rather than by arithmetic.
	for (auto [name, r] : {std::pair<const char*, PrecisionResult>{"float", f},
	                        {"posit32", p32}, {"cfloat32", c32}, {"posit16", p16}}) {
		check(std::abs(r.jitter - d.jitter) < 0.25 * d.jitter,
		      std::string(name) + " jitter should sit on the same floor as "
		      "double (" + std::to_string(r.jitter) + " vs " +
		      std::to_string(d.jitter) + ")");
	}
	std::cout << "  precision_sweep: passed\n";
}

// ---------------------------------------------------------------------------
// Configuration validation
// ---------------------------------------------------------------------------
static void test_validation() {
	auto bad = [](auto mutate) {
		auto cfg = base_config();
		mutate(cfg);
		bool caught = false;
		try { TimingRecovery<double> tr(cfg); }
		catch (const std::invalid_argument&) { caught = true; }
		return caught;
	};
	check(bad([](auto& c) { c.samples_per_symbol = 0.5; }), "sps < 1 should throw");
	check(bad([](auto& c) { c.loop_bandwidth = 0.0; }),     "zero bandwidth should throw");
	check(bad([](auto& c) { c.damping = -1.0; }),           "negative damping should throw");
	check(bad([](auto& c) { c.detector_gain = 0.0; }),      "zero detector gain should throw");
	check(bad([](auto& c) { c.max_deviation = 0.0; }),      "zero deviation should throw");
	check(bad([](auto& c) { c.max_deviation = 1.5; }),      "deviation >= 1 should throw");

	// reset() returns the loop to its initial state.
	{
		auto sig = bench::make(1000, 0.35, 0.3, 0.0, 11);
		TimingRecovery<double> tr(base_config());
		for (double v : sig.rx) tr.process(v);
		check(tr.symbols_emitted() > 0, "should have emitted symbols");
		tr.reset();
		check(tr.symbols_emitted() == 0, "reset should clear the symbol count");
		check(std::abs(tr.omega() - 2.0) < 1e-12, "reset should restore omega");
		check(!tr.is_locked(), "reset should clear lock");
		check(tr.invariants_hold(), "invariants after reset");
	}
	std::cout << "  validation: passed\n";
}

// ---------------------------------------------------------------------------

int main() {
	try {
		std::cout << "SDR timing recovery tests\n";
		test_locks_from_any_offset();
		test_tracks_clock_rate_error();
		test_symbol_recovery();
		test_bandwidth_tradeoff();
		test_mueller_muller();
		test_complex_qpsk();
		test_precision_sweep();
		test_validation();
		std::cout << "All SDR timing recovery tests passed.\n";
		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << "\n";
		return 1;
	}
}
