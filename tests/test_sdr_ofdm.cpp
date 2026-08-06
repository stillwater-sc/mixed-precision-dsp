// test_sdr_ofdm.cpp: OFDM modulation and demodulation.
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)`.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/sdr/ofdm.hpp>
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

static void check(bool condition, const std::string& msg) {
	if (!condition) throw std::runtime_error("test failed: " + msg);
}

static OfdmConfig base_config() {
	OfdmConfig c;
	c.fft_size          = 64;
	c.cyclic_prefix     = 16;
	c.guard_subcarriers = 8;
	c.pilot_spacing     = 4;
	return c;
}

// Random constellation points for one symbol's data subcarriers.
static std::vector<cd> random_data(const OfdmLayout& L, Modulation m, unsigned seed) {
	Constellation<double> c(m);
	std::mt19937 rng(seed);
	std::uniform_int_distribution<std::size_t> pick(0, c.order() - 1);
	std::vector<cd> v(L.num_data());
	for (auto& s : v) {
		const auto p = c.symbol(pick(rng));
		s = cd(p.real(), p.imag());
	}
	return v;
}

// Convolve one symbol with a channel and return the first symbol_length()
// samples. The cyclic prefix absorbs the leading transient provided the
// channel is no longer than the prefix — which is the condition OFDM is
// built on and every case here respects.
static std::vector<cd> apply_channel(const std::vector<cd>& wave,
                                     const std::vector<cd>& h,
                                     std::size_t take, double noise_sigma = 0.0,
                                     unsigned seed = 7) {
	std::vector<cd> out(wave.size() + h.size() - 1, cd{});
	for (std::size_t i = 0; i < wave.size(); ++i)
		for (std::size_t j = 0; j < h.size(); ++j)
			out[i + j] += wave[i] * h[j];
	out.resize(take);
	if (noise_sigma > 0.0) {
		std::mt19937 rng(seed);
		std::normal_distribution<double> g(0.0, noise_sigma);
		for (auto& s : out) s += cd(g(rng), g(rng));
	}
	return out;
}

// EVM of the recovered data symbols against what was sent.
static double symbol_evm(const std::vector<cd>& tx,
                         const mtl::vec::dense_vector<cd>& rx) {
	std::vector<cd> got(rx.size());
	for (std::size_t i = 0; i < rx.size(); ++i) got[i] = rx[i];
	return evm<cd>(tx, got).rms;
}

// ---------------------------------------------------------------------------
// Layout: subcarriers are allocated the way the configuration says
// ---------------------------------------------------------------------------
static void test_layout() {
	const auto cfg = base_config();
	OfdmLayout L(cfg);

	check(L.symbol_length() == cfg.fft_size + cfg.cyclic_prefix, "symbol length");
	check(L.num_data() + L.num_pilots() == L.active().size(),
	      "every active subcarrier is either data or pilot");

	// DC is never active.
	for (std::size_t k : L.active()) check(k != 0, "DC must not be active");
	// The guard band around N/2 is nulled.
	const std::size_t half = cfg.guard_subcarriers / 2;
	for (std::size_t k : L.active()) {
		const std::size_t d = (k > cfg.fft_size / 2) ? (k - cfg.fft_size / 2)
		                                             : (cfg.fft_size / 2 - k);
		check(d > half, "subcarrier " + std::to_string(k) + " is inside the guard band");
	}
	// Data and pilot sets are disjoint.
	for (std::size_t p : L.pilots())
		for (std::size_t d : L.data())
			check(p != d, "subcarrier " + std::to_string(p) + " is both pilot and data");

	std::cout << "  layout: passed (" << L.active().size() << " active, "
	          << L.num_data() << " data, " << L.num_pilots() << " pilot)\n";
}

// ---------------------------------------------------------------------------
// Ideal channel: the round trip is exact
// ---------------------------------------------------------------------------
static void test_ideal_round_trip() {
	for (Modulation m : {Modulation::qpsk, Modulation::qam16, Modulation::qam64}) {
		const auto cfg = base_config();
		OfdmModulator<double> mod(cfg);
		OfdmDemodulator<double> dem(cfg);
		const auto& L = mod.layout();

		const auto tx = random_data(L, m, 3);
		auto wave = mod.modulate(std::span<const cd>(tx));
		check(wave.size() == L.symbol_length(), "modulated length");

		auto rx = dem.demodulate(std::span<const cd>(wave.data(), wave.size()));
		check(rx.size() == L.num_data(), "demodulated count");

		double worst = 0.0;
		for (std::size_t i = 0; i < tx.size(); ++i)
			worst = std::max(worst, std::abs(rx[i] - tx[i]));
		check(worst < 1e-12, std::string(to_string(m)) +
		      " ideal round trip worst error " + std::to_string(worst));
	}
	std::cout << "  ideal_round_trip: passed\n";
}

// ---------------------------------------------------------------------------
// The cyclic prefix turns multipath into a per-subcarrier gain
//
// This is the property OFDM exists for, so it gets the most attention. A
// FLAT channel is the clean case: the estimate is a constant, linear
// interpolation of a constant is exact, and equalization must therefore
// recover the symbols to machine precision no matter how the pilots are
// spaced. That isolates the equalizer from the interpolator.
// ---------------------------------------------------------------------------
static void test_flat_channel_is_exact() {
	for (cd gain : {cd(0.8, 0.2), cd(-1.5, 0.0), cd(0.0, 2.0)}) {
		for (std::size_t spacing : {2u, 4u, 8u}) {
			auto cfg = base_config();
			cfg.pilot_spacing = spacing;
			OfdmModulator<double> mod(cfg);
			OfdmDemodulator<double> dem(cfg);
			const auto& L = mod.layout();

			const auto tx = random_data(L, Modulation::qam16, 5);
			auto wave = mod.modulate(std::span<const cd>(tx));
			std::vector<cd> w(wave.begin(), wave.end());
			auto ch = apply_channel(w, {gain}, L.symbol_length());
			auto rx = dem.demodulate(std::span<const cd>(ch.data(), ch.size()));

			const double e = symbol_evm(tx, rx);
			check(e < 1e-12, "flat channel gain " + std::to_string(gain.real()) +
			      "+" + std::to_string(gain.imag()) + "j, spacing " +
			      std::to_string(spacing) + ": EVM " + std::to_string(e));
		}
	}
	std::cout << "  flat_channel_is_exact: passed\n";
}

// ---------------------------------------------------------------------------
// Frequency-selective multipath: equalization works, and the residual is
// interpolation error that behaves the way interpolation error should
// ---------------------------------------------------------------------------
static void test_multipath_equalization() {
	const std::vector<cd> two_tap   = {cd(1, 0), cd(0.3, 0)};
	const std::vector<cd> three_tap = {cd(1, 0), cd(0, 0.5), cd(-0.3, 0)};

	std::cout << "    pilot spacing    2-tap EVM    3-tap EVM\n";
	std::vector<double> e2, e3;
	for (std::size_t spacing : {2u, 4u, 8u}) {
		auto cfg = base_config();
		cfg.pilot_spacing = spacing;
		double got[2];
		const std::vector<cd>* chans[2] = {&two_tap, &three_tap};
		for (int c = 0; c < 2; ++c) {
			OfdmModulator<double> mod(cfg);
			OfdmDemodulator<double> dem(cfg);
			const auto& L = mod.layout();
			const auto tx = random_data(L, Modulation::qam16, 11);
			auto wave = mod.modulate(std::span<const cd>(tx));
			std::vector<cd> w(wave.begin(), wave.end());
			auto ch = apply_channel(w, *chans[c], L.symbol_length());
			auto rx = dem.demodulate(std::span<const cd>(ch.data(), ch.size()));
			got[c] = symbol_evm(tx, rx);
		}
		e2.push_back(got[0]);
		e3.push_back(got[1]);
		std::cout << "    " << std::setw(13) << spacing << "    " << std::scientific
		          << std::setprecision(3) << got[0] << "    " << got[1] << "\n";
	}

	// Denser pilots estimate the channel better.
	check(e2.front() < e2.back(), "2-tap: tighter pilot spacing should reduce EVM (" +
	      std::to_string(e2.front()) + " vs " + std::to_string(e2.back()) + ")");
	check(e3.front() < e3.back(), "3-tap: tighter pilot spacing should reduce EVM (" +
	      std::to_string(e3.front()) + " vs " + std::to_string(e3.back()) + ")");
	// A longer channel varies faster across frequency, so it is harder to
	// interpolate at any given pilot density.
	for (std::size_t i = 0; i < e2.size(); ++i)
		check(e3[i] > e2[i], "the 3-tap channel should be harder to estimate "
		      "than the 2-tap one at spacing index " + std::to_string(i));
	// At the tightest spacing the link is usable for 16-QAM, whose decision
	// boundaries sit at a third of the symbol spacing.
	check(e2.front() < 0.05, "2-tap EVM at the tightest spacing is " +
	      std::to_string(e2.front()));

	std::cout << "  multipath_equalization: passed\n";
}

// ---------------------------------------------------------------------------
// Bits survive an AWGN channel
// ---------------------------------------------------------------------------
static void test_bits_through_awgn() {
	const auto cfg = base_config();
	OfdmModulator<double> mod(cfg);
	OfdmDemodulator<double> dem(cfg);
	const auto& L = mod.layout();
	Constellation<double> c(Modulation::qpsk);
	const std::size_t nb = c.bits_per_symbol();

	std::mt19937 rng(21);
	std::uniform_int_distribution<std::size_t> pick(0, c.order() - 1);
	// Es/N0 = 20 dB per subcarrier.
	const double n0 = std::pow(10.0, -20.0 / 10.0);

	std::vector<std::uint8_t> tx_bits, rx_bits, tmp(nb);
	for (std::size_t sym = 0; sym < 200; ++sym) {
		std::vector<cd> tx(L.num_data());
		std::vector<std::size_t> idx(L.num_data());
		for (std::size_t i = 0; i < tx.size(); ++i) {
			idx[i] = pick(rng);
			const auto p = c.symbol(idx[i]);
			tx[i] = cd(p.real(), p.imag());
			c.bits_of(idx[i], tmp);
			tx_bits.insert(tx_bits.end(), tmp.begin(), tmp.end());
		}
		auto wave = mod.modulate(std::span<const cd>(tx));
		std::vector<cd> w(wave.begin(), wave.end());
		// The IFFT spreads each subcarrier's energy over N samples, so the
		// time-domain noise scale that gives a target per-subcarrier Es/N0
		// carries a 1/sqrt(N) factor.
		const double sigma = std::sqrt(n0 / 2.0) /
		                     std::sqrt(static_cast<double>(cfg.fft_size));
		auto ch = apply_channel(w, {cd(1, 0)}, L.symbol_length(), sigma,
		                        static_cast<unsigned>(sym + 1));
		auto rx = dem.demodulate(std::span<const cd>(ch.data(), ch.size()));
		for (std::size_t i = 0; i < rx.size(); ++i) {
			c.demap_hard_bits(rx[i], tmp);
			rx_bits.insert(rx_bits.end(), tmp.begin(), tmp.end());
		}
	}
	const auto r = ber(tx_bits, rx_bits);
	std::cout << "  bits_through_awgn: passed (" << r.bit_errors << " errors in "
	          << r.total_bits << " bits, BER " << std::scientific
	          << std::setprecision(2) << r.rate << ")\n";
	// 20 dB on QPSK is comfortable; the point is that bits get through, not a
	// precise BER — the pilots and equalizer add their own small penalty.
	check(r.rate < 1e-3, "BER " + std::to_string(r.rate) + " too high at 20 dB");
}

// ---------------------------------------------------------------------------
// PAPR: the characteristic OFDM cost, and it grows with subcarrier count
// ---------------------------------------------------------------------------
static void test_papr() {
	std::cout << "    fft size   mean PAPR over 200 symbols\n";
	double prev = 0.0;
	for (std::size_t N : {16u, 64u, 256u}) {
		auto cfg = base_config();
		cfg.fft_size = N;
		cfg.cyclic_prefix = N / 4;
		cfg.guard_subcarriers = N / 8;
		OfdmModulator<double> mod(cfg);
		const auto& L = mod.layout();

		double sum = 0.0;
		const std::size_t trials = 200;
		for (std::size_t t = 0; t < trials; ++t) {
			const auto tx = random_data(L, Modulation::qam16,
			                             static_cast<unsigned>(t + 100));
			auto wave = mod.modulate(std::span<const cd>(tx));
			sum += papr_db<cd>(std::span<const cd>(wave.data(), wave.size()));
		}
		const double mean = sum / static_cast<double>(trials);
		std::cout << "    " << std::setw(8) << N << "   " << std::fixed
		          << std::setprecision(2) << mean << " dB\n";
		check(mean > 3.0 && mean < 15.0, "PAPR " + std::to_string(mean) +
		      " dB at N = " + std::to_string(N) + " is outside any sane range");
		check(mean > prev, "PAPR should grow with subcarrier count: N = " +
		      std::to_string(N) + " gave " + std::to_string(mean) +
		      ", previous " + std::to_string(prev));
		prev = mean;
	}
	std::cout << "  papr: passed\n";
}

// ---------------------------------------------------------------------------
// Precision sweep: EVM against FFT arithmetic precision
//
// Subcarrier orthogonality is exactly what the transform's precision buys, so
// the residual here IS the intercarrier interference the issue asks about:
// with an ideal channel there is nothing else left to measure.
// ---------------------------------------------------------------------------
template <typename State>
static double ici_evm(const char* name) {
	const auto cfg = base_config();
	OfdmModulator<State> mod(cfg);
	OfdmDemodulator<State> dem(cfg);
	const auto& L = mod.layout();
	using CS = sw::dsp::complex_for_t<State>;

	Constellation<double> c(Modulation::qam64);
	std::mt19937 rng(31);
	std::uniform_int_distribution<std::size_t> pick(0, c.order() - 1);

	std::vector<cd> ref(L.num_data());
	std::vector<CS> tx(L.num_data());
	for (std::size_t i = 0; i < ref.size(); ++i) {
		const auto p = c.symbol(pick(rng));
		ref[i] = cd(p.real(), p.imag());
		tx[i] = CS(static_cast<State>(p.real()), static_cast<State>(p.imag()));
	}
	auto wave = mod.modulate(std::span<const CS>(tx));
	auto rx = dem.demodulate(std::span<const CS>(wave.data(), wave.size()));

	std::vector<cd> got(rx.size());
	for (std::size_t i = 0; i < rx.size(); ++i)
		got[i] = cd(static_cast<double>(rx[i].real()),
		            static_cast<double>(rx[i].imag()));
	const double e = evm<cd>(ref, got).rms;
	std::cout << "      " << name << ": EVM " << std::scientific
	          << std::setprecision(3) << e << "  (" << std::fixed
	          << std::setprecision(1) << 20.0 * std::log10(e) << " dB)\n";
	return e;
}

static void test_precision_sweep() {
	using posit32  = sw::universal::posit<32, 2>;
	using posit16  = sw::universal::posit<16, 2>;
	using cfloat32 = sw::universal::cfloat<32, 8, std::uint32_t, true, false, false>;

	std::cout << "    intercarrier interference vs FFT precision (64-QAM, ideal channel):\n";
	const double d   = ici_evm<double>  ("double  ");
	const double f   = ici_evm<float>   ("float   ");
	const double p32 = ici_evm<posit32> ("posit32 ");
	const double c32 = ici_evm<cfloat32>("cfloat32");
	const double p16 = ici_evm<posit16> ("posit16 ");

	check(d < 1e-12, "double ICI should be at machine precision, got " +
	      std::to_string(d));
	// The 32-bit types land near their own epsilon, orders below any
	// constellation's decision distance.
	for (auto [name, v] : {std::pair<const char*, double>{"float", f},
	                        {"posit32", p32}, {"cfloat32", c32}})
		check(v < 1e-5, std::string(name) + " ICI " + std::to_string(v) +
		      " is worse than its arithmetic explains");
	// posit16 degrades but must stay well inside 64-QAM's decision distance,
	// which is 2/sqrt(42) = 0.309 between neighbouring points.
	check(p16 > d, "posit16 should show more ICI than double");
	check(p16 < 0.03, "posit16 ICI " + std::to_string(p16) +
	      " encroaches on 64-QAM's decision distance");
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
		try { OfdmLayout L(cfg); }
		catch (const std::invalid_argument&) { caught = true; }
		return caught;
	};
	check(bad([](auto& c) { c.fft_size = 63; }),          "non-power-of-two should throw");
	check(bad([](auto& c) { c.fft_size = 4; }),           "tiny fft_size should throw");
	check(bad([](auto& c) { c.cyclic_prefix = 0; }),      "zero prefix should throw");
	check(bad([](auto& c) { c.cyclic_prefix = 128; }),    "prefix > fft_size should throw");
	check(bad([](auto& c) { c.pilot_spacing = 1; }),      "spacing 1 leaves no data");
	check(bad([](auto& c) { c.guard_subcarriers = 64; }), "guard >= fft_size should throw");
	check(bad([](auto& c) { c.guard_subcarriers = 62; }), "guard leaving too few subcarriers");

	// Wrong block sizes are rejected on both halves.
	{
		const auto cfg = base_config();
		OfdmModulator<double> mod(cfg);
		OfdmDemodulator<double> dem(cfg);
		std::vector<cd> wrong(3);
		bool caught = false;
		try { mod.modulate(std::span<const cd>(wrong)); }
		catch (const std::invalid_argument&) { caught = true; }
		check(caught, "wrong data count should throw");

		caught = false;
		try { dem.demodulate(std::span<const cd>(wrong)); }
		catch (const std::invalid_argument&) { caught = true; }
		check(caught, "wrong symbol length should throw");
	}

	// PAPR rejects what it cannot measure.
	{
		bool caught = false;
		std::vector<cd> empty;
		try { papr_db<cd>(std::span<const cd>(empty)); }
		catch (const std::invalid_argument&) { caught = true; }
		check(caught, "empty block should throw");

		caught = false;
		std::vector<cd> zeros(8, cd{});
		try { papr_db<cd>(std::span<const cd>(zeros)); }
		catch (const std::invalid_argument&) { caught = true; }
		check(caught, "an all-zero block has no PAPR and should throw");
	}
	std::cout << "  validation: passed\n";
}

// ---------------------------------------------------------------------------

int main() {
	try {
		std::cout << "SDR OFDM tests\n";
		test_layout();
		test_ideal_round_trip();
		test_flat_channel_is_exact();
		test_multipath_equalization();
		test_bits_through_awgn();
		test_papr();
		test_precision_sweep();
		test_validation();
		std::cout << "All SDR OFDM tests passed.\n";
		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << "\n";
		return 1;
	}
}
