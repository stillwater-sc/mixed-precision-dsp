// test_sdr_channelizer.cpp: oversampled analysis/synthesis filter bank.
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)`.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/sdr/channelizer.hpp>

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

// Push `in` through analysis then synthesis, returning the wideband output.
template <typename State = double>
static std::vector<double> round_trip(std::size_t M, const std::vector<double>& in) {
	OversampledChannelizer<State, State, State> an(M);
	OversampledSynthesizer<State, State, State> sy(M);
	using CS = sw::dsp::complex_for_t<State>;
	std::vector<double> out;
	std::vector<State> block(an.hop());
	for (std::size_t n = 0; n + an.hop() <= in.size(); n += an.hop()) {
		for (std::size_t i = 0; i < an.hop(); ++i)
			block[i] = static_cast<State>(in[n + i]);
		auto ch = an.process(std::span<const State>(block.data(), block.size()));
		auto w  = sy.process(std::span<const CS>(ch.data(), M));
		for (std::size_t i = 0; i < w.size(); ++i)
			out.push_back(static_cast<double>(w[i]));
	}
	return out;
}

static double reconstruction_error(std::size_t M, const std::vector<double>& in,
                                   const std::vector<double>& out) {
	const std::size_t d = OversampledSynthesizer<double>::cascade_delay(M);
	double e = 0.0, t = 0.0;
	for (std::size_t i = 500; i < in.size() && i + d < out.size() && i < 4000; ++i) {
		const double r = out[i + d] - in[i];
		e += r * r;
		t += in[i] * in[i];
	}
	return std::sqrt(e / t);
}

static std::vector<double> noise(std::size_t n, unsigned seed = 1) {
	std::mt19937 rng(seed);
	std::uniform_real_distribution<double> u(-1.0, 1.0);
	std::vector<double> v(n);
	for (auto& x : v) x = u(rng);
	return v;
}

// ---------------------------------------------------------------------------
// Perfect reconstruction, at machine precision
// ---------------------------------------------------------------------------
static void test_perfect_reconstruction() {
	std::cout << "       M   hop   delay   reconstruction error\n";
	for (std::size_t M : {4u, 8u, 16u, 32u, 64u}) {
		const auto in = noise(6000);
		const auto out = round_trip(M, in);
		const double err = reconstruction_error(M, in, out);
		const std::size_t d = OversampledSynthesizer<double>::cascade_delay(M);
		std::cout << "    " << std::setw(4) << M << "  " << std::setw(4) << M / 2
		          << "   " << std::setw(5) << d << "   " << std::scientific
		          << std::setprecision(3) << err << "\n";
		// 2x oversampling plus the squared-overlap-add normalization make
		// this exact, not merely good.
		check(err < 1e-12, "M = " + std::to_string(M) +
		      " reconstruction error " + std::to_string(err) +
		      ", expected machine precision");
	}
	std::cout << "  perfect_reconstruction: passed\n";
}

// ---------------------------------------------------------------------------
// The delay formula is exact, not approximate
// ---------------------------------------------------------------------------
static void test_delay_is_exact() {
	for (std::size_t M : {8u, 16u, 32u}) {
		const auto in = noise(4000, 7);
		const auto out = round_trip(M, in);
		// Search a window and confirm the predicted delay is the best one.
		std::size_t best = 0;
		double best_err = 1e30;
		for (std::size_t d = 0; d < 2 * M && d + 3000 < out.size(); ++d) {
			double e = 0.0;
			for (std::size_t i = 500; i < 3000; ++i) {
				const double r = out[i + d] - in[i];
				e += r * r;
			}
			if (e < best_err) { best_err = e; best = d; }
		}
		check(best == OversampledSynthesizer<double>::cascade_delay(M),
		      "M = " + std::to_string(M) + ": measured delay " +
		      std::to_string(best) + ", predicted " +
		      std::to_string(OversampledSynthesizer<double>::cascade_delay(M)));
	}
	std::cout << "  delay_is_exact: passed\n";
}

// ---------------------------------------------------------------------------
// Analysis isolates a narrowband tone into the channel that contains it
// ---------------------------------------------------------------------------
static void test_channel_isolation() {
	const std::size_t M = 16;
	std::cout << "    tone in channel   peak channel   adjacent rejection\n";
	// Targets kept away from DC and from M/2. A real input is
	// conjugate-symmetric, so a tone in channel c also appears in M-c; when
	// c is within a bin or two of either edge the two images leak into the
	// same bins and the peak is genuinely ambiguous. That is a property of
	// a real-valued wideband signal, not of the bank.
	for (std::size_t target : {3u, 5u}) {
		// A tone at the centre of channel `target`.
		const double f = static_cast<double>(target) / static_cast<double>(M);
		std::vector<double> in(8000);
		for (std::size_t n = 0; n < in.size(); ++n)
			in[n] = std::cos(sw::dsp::two_pi * f * static_cast<double>(n));

		OversampledChannelizer<double> an(M);
		std::vector<double> energy(M, 0.0);
		std::size_t frames = 0;
		for (std::size_t n = 0; n + an.hop() <= in.size(); n += an.hop()) {
			auto ch = an.process(std::span<const double>(in.data() + n, an.hop()));
			if (++frames < 40) continue;          // let the window fill
			for (std::size_t c = 0; c < M; ++c) energy[c] += std::norm(ch[c]);
		}

		std::size_t peak = 0;
		for (std::size_t c = 0; c < M; ++c) if (energy[c] > energy[peak]) peak = c;
		check(peak == target, "tone in channel " + std::to_string(target) +
		      " peaked in channel " + std::to_string(peak));

		// Rejection three channels away. A single transform's worth of
		// window is not a sharp filter — measured 15-20 dB — and that is the
		// trade this pair makes for exact reconstruction. Sharper channel
		// responses need a long prototype, which is what
		// sw::dsp::multirate::Channelizer provides for analysis-only use.
		// The assertion is therefore that energy concentrates where it
		// should, not that this is a selective filter bank.
		// Measured on the DC side of the tone. Going the other way runs
		// into the conjugate mirror at M-target: for target 6 the mirror
		// sits at 10, so "three channels up" is channel 9, right beside it,
		// and reads 7 dB of leakage from the image rather than the skirt of
		// the channel under test.
		const std::size_t adj = target - 2;
		const double rej = 10.0 * std::log10(energy[peak] / energy[adj]);
		std::cout << "    " << std::setw(15) << target << "   " << std::setw(12) << peak
		          << "   " << std::fixed << std::setprecision(1) << rej << " dB\n";
		check(rej > 12.0, "channel " + std::to_string(target) +
		      " rejection only " + std::to_string(rej) + " dB");
	}
	std::cout << "  channel_isolation: passed\n";
}

// ---------------------------------------------------------------------------
// A channel can be modified in isolation and the change survives resynthesis
// ---------------------------------------------------------------------------
static void test_channel_processing() {
	const std::size_t M = 16;
	const auto in = noise(6000, 3);

	OversampledChannelizer<double> an(M);
	OversampledSynthesizer<double> sy(M);
	std::vector<double> zeroed, passed;
	for (std::size_t n = 0; n + an.hop() <= in.size(); n += an.hop()) {
		auto ch = an.process(std::span<const double>(in.data() + n, an.hop()));
		auto copy = ch;
		// Null one channel and its conjugate mirror.
		copy[5] = cd{};
		copy[M - 5] = cd{};
		auto w = sy.process(std::span<const cd>(copy.data(), M));
		for (std::size_t i = 0; i < w.size(); ++i) zeroed.push_back(w[i]);
	}
	passed = round_trip(M, in);

	// Nulling a channel must change the output, but only partially — the
	// rest of the band has to survive.
	double diff = 0.0, total = 0.0;
	for (std::size_t i = 500; i < 4000 && i < zeroed.size(); ++i) {
		const double d = zeroed[i] - passed[i];
		diff += d * d;
		total += passed[i] * passed[i];
	}
	const double frac = std::sqrt(diff / total);
	check(frac > 0.05, "nulling a channel should change the output, got " +
	      std::to_string(frac));
	check(frac < 0.9, "nulling one of 16 channels should not destroy the "
	      "signal, got " + std::to_string(frac));
	std::cout << "  channel_processing: passed (nulling 1 of " << M
	          << " changed the output by " << std::fixed << std::setprecision(3)
	          << frac * 100.0 << "%)\n";
}

// ---------------------------------------------------------------------------
// Precision sweep: reconstruction error against filter and FFT precision
// ---------------------------------------------------------------------------
static void test_precision_sweep() {
	using posit32  = sw::universal::posit<32, 2>;
	using posit16  = sw::universal::posit<16, 2>;
	using cfloat32 = sw::universal::cfloat<32, 8, std::uint32_t, true, false, false>;

	const std::size_t M = 16;
	const auto in = noise(6000, 11);

	std::cout << "    reconstruction error vs bank precision (M = 16):\n";
	struct Row { const char* name; double err; };
	std::vector<Row> rows = {
		{"double  ", reconstruction_error(M, in, round_trip<double>(M, in))},
		{"float   ", reconstruction_error(M, in, round_trip<float>(M, in))},
		{"posit32 ", reconstruction_error(M, in, round_trip<posit32>(M, in))},
		{"cfloat32", reconstruction_error(M, in, round_trip<cfloat32>(M, in))},
		{"posit16 ", reconstruction_error(M, in, round_trip<posit16>(M, in))},
	};
	for (const auto& r : rows)
		std::cout << "      " << r.name << ": " << std::scientific
		          << std::setprecision(3) << r.err << "\n";

	check(rows[0].err < 1e-12, "double should reconstruct exactly");
	// The 32-bit types are limited by their own epsilon, not by the
	// structure: roughly 1e-7 for a 24-bit significand.
	for (std::size_t i = 1; i + 1 < rows.size(); ++i)
		check(rows[i].err < 1e-5, std::string(rows[i].name) +
		      " reconstruction error " + std::to_string(rows[i].err) +
		      " is worse than its arithmetic explains");
	// posit16 degrades, but must stay usable and finite.
	check(std::isfinite(rows.back().err) && rows.back().err < 0.05,
	      "posit16 reconstruction error " + std::to_string(rows.back().err));
	check(rows.back().err > rows[0].err, "posit16 should be worse than double");
	std::cout << "  precision_sweep: passed\n";
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------
static void test_validation() {
	bool caught = false;
	try { OversampledChannelizer<double> a(7); }      // odd M
	catch (const std::invalid_argument&) { caught = true; }
	check(caught, "odd M should throw: the hop is M/2");

	caught = false;
	try { OversampledChannelizer<double> a(0); }
	catch (const std::invalid_argument&) { caught = true; }
	check(caught, "M = 0 should throw");

	caught = false;
	try { OversampledSynthesizer<double> s(7); }
	catch (const std::invalid_argument&) { caught = true; }
	check(caught, "odd M should throw for the synthesizer too");

	// Wrong block sizes are rejected on both halves.
	{
		OversampledChannelizer<double> a(8);
		std::vector<double> wrong(3, 0.0);
		caught = false;
		try { a.process(std::span<const double>(wrong)); }
		catch (const std::invalid_argument&) { caught = true; }
		check(caught, "a block that is not hop() long should throw");
	}
	{
		OversampledSynthesizer<double> s(8);
		std::vector<cd> wrong(3);
		caught = false;
		try { s.process(std::span<const cd>(wrong)); }
		catch (const std::invalid_argument&) { caught = true; }
		check(caught, "a channel vector that is not M long should throw");
	}

	// reset() returns both halves to a clean state, so a second run over the
	// same input reproduces the first exactly.
	{
		const std::size_t M = 8;
		const auto in = noise(2000, 5);
		OversampledChannelizer<double> a(M);
		OversampledSynthesizer<double> s(M);
		auto run = [&]() {
			std::vector<double> out;
			for (std::size_t n = 0; n + a.hop() <= in.size(); n += a.hop()) {
				auto ch = a.process(std::span<const double>(in.data() + n, a.hop()));
				auto w = s.process(std::span<const cd>(ch.data(), M));
				for (std::size_t i = 0; i < w.size(); ++i) out.push_back(w[i]);
			}
			return out;
		};
		const auto first = run();
		a.reset(); s.reset();
		const auto second = run();
		for (std::size_t i = 0; i < first.size(); ++i)
			check(std::abs(first[i] - second[i]) < 1e-15,
			      "reset should make a rerun identical, differs at " +
			      std::to_string(i));
	}
	std::cout << "  validation: passed\n";
}

// ---------------------------------------------------------------------------

int main() {
	try {
		std::cout << "SDR channelizer tests\n";
		test_perfect_reconstruction();
		test_delay_is_exact();
		test_channel_isolation();
		test_channel_processing();
		test_precision_sweep();
		test_validation();
		std::cout << "All SDR channelizer tests passed.\n";
		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << "\n";
		return 1;
	}
}
