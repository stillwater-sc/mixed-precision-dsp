// test_sdr_rrc.cpp: root-raised-cosine / raised-cosine pulse shaping.
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)`.
//
// Reference coefficients come from scikit-commpy's rrcosfilter, renormalized
// to unit energy to match MATLAB rcosdesign()'s convention and ours. They are
// an independent implementation, so agreeing with them catches a transcription
// error; the zero-ISI and spectrum checks below are what catch a wrong formula.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/sdr/rrc.hpp>
#include <sw/dsp/sdr/constellation.hpp>
#include <sw/dsp/filter/fir/polyphase.hpp>

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

using sw::dsp::sdr::rrc_filter;
using sw::dsp::sdr::raised_cosine_filter;
using sw::dsp::sdr::peak_isi;
using sw::dsp::sdr::PulseNormalization;

static void check(bool condition, const std::string& msg) {
	if (!condition) throw std::runtime_error("test failed: " + msg);
}

// Full linear convolution.
template <typename T>
static mtl::vec::dense_vector<double> convolve(const mtl::vec::dense_vector<T>& a,
                                                const mtl::vec::dense_vector<T>& b) {
	mtl::vec::dense_vector<double> out(a.size() + b.size() - 1, 0.0);
	for (std::size_t i = 0; i < a.size(); ++i)
		for (std::size_t j = 0; j < b.size(); ++j)
			out[i + j] += static_cast<double>(a[i]) * static_cast<double>(b[j]);
	return out;
}

// ---------------------------------------------------------------------------
// The closed form agrees with an independent implementation
//
// Checked at the TAP-FUNCTION level, at matching abscissae, deliberately.
// scikit-commpy evaluates on the grid (arange(N) - N/2)/Fs, which for an ODD
// tap count sits at half-sample offsets and so contains no tap at t = 0 —
// its 33-tap alpha=0.25 pulse spans x = -4.125 .. 3.875. That grid is not
// centred and is not Nyquist-optimal: its composite measures 7.3e-3 peak ISI
// where a sample-centred grid of the same length measures 1.6e-3. MATLAB's
// rcosdesign() centres on a sample, as this library does.
//
// So comparing whole tap vectors would compare grid conventions, not
// formulas. Evaluating both at the same x isolates the formula — including
// the two singular branches, which an even N puts directly on commpy's grid.
// ---------------------------------------------------------------------------
static void test_reference_formula() {
	struct Ref { double alpha, x, value; };
	// from commpy.filters.rrcosfilter(32, alpha, 1.0, 4.0), read at x = t.
	// Unnormalized, so these check the closed form itself.
	const Ref refs[] = {
		{0.25,  0.00,  1.068309886183791e+00},   // t = 0 limit
		{0.25,  0.25,  9.431653207443821e-01},
		{0.25,  0.50,  6.217974105091317e-01},
		{0.25,  1.00, -6.423715577699857e-02},
		{0.25, -0.75,  2.378618293149634e-01},
		{0.25,  2.00,  5.305164769729841e-02},
		{0.25, -3.00, -3.751317983987938e-02},
		// alpha = 1 puts x = 0.25 exactly on the |4*a*x| = 1 singularity.
		{1.00,  0.00,  1.273239544735163e+00},   // t = 0 limit
		{1.00,  0.25,  1.000000000000000e+00},   // singular limit
		{1.00,  0.50,  4.244131815783876e-01},
		{1.00,  1.00, -8.488263631567752e-02},
		{1.00,  2.00, -2.021015150373274e-02},
		{1.00, -3.00, -8.903773040106033e-03},
		// alpha = 0 degenerates to a sinc.
		{0.00,  0.00,  1.000000000000000e+00},
		{0.00,  0.25,  9.003163161571061e-01},
		{0.00,  0.50,  6.366197723675814e-01},
		{0.00, -0.75,  3.001054387190354e-01},
		{0.00,  1.00,  0.0},                     // sinc zero
		{0.00,  2.00,  0.0},
	};
	for (const auto& r : refs) {
		const double got = sw::dsp::sdr::detail::rrc_tap(r.x, r.alpha);
		check(std::isfinite(got), "rrc_tap not finite at x=" + std::to_string(r.x) +
		      " alpha=" + std::to_string(r.alpha));
		check(std::abs(got - r.value) < 1e-12,
		      "rrc_tap(alpha=" + std::to_string(r.alpha) + ", x=" +
		      std::to_string(r.x) + ") = " + std::to_string(got) +
		      ", reference " + std::to_string(r.value));
	}

	// And the designer really does place tap n at x = (n - centre)/sps and
	// scale by a single constant, so the shape it emits is this formula.
	const std::size_t N = 33, sps = 4;
	const double alpha = 0.25;
	auto h = rrc_filter<double>(N, sps, alpha);
	const double centre = 0.5 * (static_cast<double>(N) - 1.0);
	const double scale = h[N / 2] / sw::dsp::sdr::detail::rrc_tap(0.0, alpha);
	for (std::size_t n = 0; n < N; ++n) {
		const double x = (static_cast<double>(n) - centre) / static_cast<double>(sps);
		const double want = sw::dsp::sdr::detail::rrc_tap(x, alpha) * scale;
		check(std::abs(h[n] - want) < 1e-14,
		      "designer tap " + std::to_string(n) + " does not follow the closed form");
	}
	std::cout << "  reference_formula: passed\n";
}

// ---------------------------------------------------------------------------
// Structural properties: symmetry, normalization, singular branches finite
// ---------------------------------------------------------------------------
static void test_structure() {
	for (double alpha : {0.0, 0.1, 0.22, 0.25, 0.35, 0.5, 0.75, 1.0}) {
		for (std::size_t sps : {2u, 4u, 8u}) {
			const std::size_t N = 8 * sps + 1;
			auto h = rrc_filter<double>(N, sps, alpha);
			const std::string tag = "alpha=" + std::to_string(alpha) +
			                        " sps=" + std::to_string(sps);

			// Every tap finite — this is what a mishandled singularity breaks.
			for (std::size_t i = 0; i < N; ++i)
				check(std::isfinite(h[i]), tag + " tap " + std::to_string(i) +
				      " is not finite (singularity mishandled?)");

			// Symmetric about the centre: the pulse is even in t.
			for (std::size_t i = 0; i < N / 2; ++i)
				check(std::abs(h[i] - h[N - 1 - i]) < 1e-15,
				      tag + " not symmetric at " + std::to_string(i));

			// Unit energy by default.
			double energy = 0.0;
			for (std::size_t i = 0; i < N; ++i) energy += h[i] * h[i];
			check(std::abs(energy - 1.0) < 1e-12,
			      tag + " energy = " + std::to_string(energy));

			// The peak is the centre tap.
			for (std::size_t i = 0; i < N; ++i)
				check(std::abs(h[i]) <= std::abs(h[N / 2]) + 1e-15,
				      tag + " peak is not the centre tap");
		}
	}

	// unit_dc_gain really does give unit DC gain.
	auto g = rrc_filter<double>(33, 4, 0.25, PulseNormalization::unit_dc_gain);
	double sum = 0.0;
	for (std::size_t i = 0; i < g.size(); ++i) sum += g[i];
	check(std::abs(sum - 1.0) < 1e-12, "unit_dc_gain sum = " + std::to_string(sum));

	std::cout << "  structure: passed\n";
}

// ---------------------------------------------------------------------------
// The defining property: RRC (x) RRC is Nyquist — its samples at every
// non-zero symbol offset vanish, so the composite carries no ISI.
//
// The residual at finite length is set by TRUNCATION, and how much
// truncation costs depends strongly on rolloff: the pulse tails decay like
// 1/t^3 for a healthy alpha but only like 1/t as alpha -> 0, where the pulse
// degenerates to a sinc and a brick-wall filter genuinely needs infinite
// support. So a single threshold across all rolloffs would be meaningless.
// The bounds below sit just above what a 16-symbol span actually achieves,
// and the span sweep that follows is the real check: if the residual were a
// formula error rather than truncation, it would not fall with length.
// ---------------------------------------------------------------------------
static void test_zero_isi() {
	// Measured peak ISI at span = 16, worst over sps in {2,4,8}:
	//   alpha 0.00 -> 5.7e-2      alpha 0.25 -> 1.6e-3
	//   alpha 0.10 -> 1.3e-2      alpha 0.35 -> 1.8e-3
	//   alpha 0.20 -> 6.1e-3      alpha 1.00 -> 6.3e-4
	const struct { double alpha, bound; } cases[] = {
		{0.00, 1.0e-1}, {0.10, 2.0e-2}, {0.20, 1.0e-2},
		{0.25, 3.0e-3}, {0.35, 3.0e-3}, {0.50, 3.0e-3}, {1.00, 3.0e-3},
	};

	for (const auto& c : cases) {
		for (std::size_t sps : {2u, 4u, 8u}) {
			const std::size_t N = 16 * sps + 1;
			auto h = rrc_filter<double>(N, sps, c.alpha);
			auto composite = convolve(h, h);

			const std::string tag = "alpha=" + std::to_string(c.alpha) +
			                        " sps=" + std::to_string(sps);

			// Unit-energy RRC means the composite peaks at exactly 1.
			const std::size_t centre = composite.size() / 2;
			check(std::abs(composite[centre] - 1.0) < 1e-12,
			      tag + " composite peak = " + std::to_string(composite[centre]) +
			      ", expected 1");

			const double isi = peak_isi(composite, sps);
			check(isi < c.bound, tag + " peak ISI = " + std::to_string(isi) +
			      ", expected < " + std::to_string(c.bound));
		}
	}

	// The residual really is truncation: lengthening the pulse has to shrink
	// it, at every rolloff including the pathological alpha = 0.
	for (double alpha : {0.0, 0.25, 0.35}) {
		const std::size_t sps = 4;
		double prev = 1e30;
		for (std::size_t span : {8u, 16u, 32u, 64u}) {
			auto h = rrc_filter<double>(span * sps + 1, sps, alpha);
			const double isi = peak_isi(convolve(h, h), sps);
			check(isi < prev, "alpha=" + std::to_string(alpha) +
			      ": peak ISI did not improve going to span " +
			      std::to_string(span) + " (" + std::to_string(isi) +
			      " vs " + std::to_string(prev) + ")");
			prev = isi;
		}
	}
	std::cout << "  zero_isi: passed\n";
}

// ---------------------------------------------------------------------------
// The composite is a raised cosine, not merely some Nyquist pulse.
//
// Compared shape-wise, both scaled to unit peak, so this is about the pulse
// rather than either function's normalization convention.
//
// The bound is expressed against the composite's own peak ISI rather than as
// a fixed number, because the two have the same cause: truncating the RRC
// before convolving is exactly what separates the composite from the ideal
// raised cosine. Measured across alpha in [0.2, 0.5] and spans 16 to 64 the
// deviation stays within about 4x the ISI and the two shrink together, so
// tying them makes the test track the physics instead of a tolerance that
// would need retuning for every rolloff.
// ---------------------------------------------------------------------------
static void test_composite_is_raised_cosine() {
	auto unit_peak = [](auto& v) {
		double p = 0.0;
		for (std::size_t i = 0; i < v.size(); ++i)
			p = std::max(p, std::abs(static_cast<double>(v[i])));
		for (std::size_t i = 0; i < v.size(); ++i) v[i] = v[i] / p;
	};

	for (double alpha : {0.2, 0.25, 0.35, 0.5}) {
		double prev_dev = 1e30;
		for (std::size_t span : {16u, 32u, 64u}) {
			const std::size_t sps = 4, N = span * sps + 1;
			auto h = rrc_filter<double>(N, sps, alpha);
			auto composite = convolve(h, h);
			const double isi = peak_isi(composite, sps);

			auto rc = raised_cosine_filter<double>(composite.size(), sps, alpha);
			unit_peak(composite);
			unit_peak(rc);

			double worst = 0.0;
			for (std::size_t i = 0; i < composite.size(); ++i)
				worst = std::max(worst, std::abs(composite[i] - rc[i]));

			const std::string tag = "alpha=" + std::to_string(alpha) +
			                        " span=" + std::to_string(span);
			check(worst < 5.0 * isi + 1e-6, tag +
			      ": composite deviates from the analytic raised cosine by " +
			      std::to_string(worst) + ", more than 5x its own peak ISI (" +
			      std::to_string(isi) + ") — that would mean a cause other "
			      "than truncation");
			check(worst < prev_dev, tag +
			      ": deviation did not shrink with span (" + std::to_string(worst) +
			      " vs " + std::to_string(prev_dev) + ")");
			prev_dev = worst;
		}
	}

	// The analytic raised cosine is itself Nyquist: 1 at the centre, 0 at
	// every other symbol instant, to machine precision — no truncation
	// argument needed, because it is evaluated rather than convolved.
	for (double alpha : {0.0, 0.25, 0.5, 1.0}) {
		const std::size_t sps = 4, N = 16 * sps + 1;
		auto rc = raised_cosine_filter<double>(N, sps, alpha,
		                                        PulseNormalization::unit_dc_gain);
		const double isi = peak_isi(rc, sps);
		check(isi < 1e-12, "analytic RC alpha=" + std::to_string(alpha) +
		      " is not Nyquist: peak ISI " + std::to_string(isi));
	}
	std::cout << "  composite_is_raised_cosine: passed\n";
}

// ---------------------------------------------------------------------------
// Eye diagram: shape BPSK symbols, matched-filter them, and confirm the eye
// is widest at the correct sampling phase.
// ---------------------------------------------------------------------------
static void test_eye_opens_at_correct_instant() {
	const std::size_t sps = 8, span = 10, N = span * sps + 1;
	const double alpha = 0.35;
	auto h = rrc_filter<double>(N, sps, alpha);

	// Random +/-1 symbols.
	std::mt19937 rng(2024);
	std::bernoulli_distribution coin(0.5);
	const std::size_t num_symbols = 400;
	mtl::vec::dense_vector<double> symbols(num_symbols);
	for (std::size_t i = 0; i < num_symbols; ++i) symbols[i] = coin(rng) ? 1.0 : -1.0;

	// TX shaping and RX matched filtering, both at the full sample rate so
	// every phase can be inspected. Two passes of the same symmetric pulse
	// is exactly the TX/RX matched pair.
	sw::dsp::PolyphaseInterpolator<double> shaper(h, sps);
	auto tx = shaper.process_block(std::span<const double>(symbols.data(), symbols.size()));

	sw::dsp::FIRFilter<double> matched(h);
	mtl::vec::dense_vector<double> rx(tx.size());
	for (std::size_t i = 0; i < tx.size(); ++i) rx[i] = matched.process(tx[i]);

	// Group delay of the pair, in samples: each filter contributes (N-1)/2.
	const std::size_t delay = (N - 1);
	// Skip the leading and trailing transients.
	const std::size_t first = delay + 4 * sps;
	const std::size_t last  = tx.size() - 4 * sps;

	// For each sampling phase, the eye opening is the smallest |y| seen.
	std::vector<double> opening(sps, 1e30);
	for (std::size_t phase = 0; phase < sps; ++phase) {
		for (std::size_t i = first + phase; i < last; i += sps)
			opening[phase] = std::min(opening[phase], std::abs(rx[i]));
	}

	// The correct instant is where the pulse pair peaks: index `delay`,
	// i.e. phase (delay % sps).
	const std::size_t best_phase = static_cast<std::size_t>(
		std::max_element(opening.begin(), opening.end()) - opening.begin());
	check(best_phase == delay % sps,
	      "eye is widest at phase " + std::to_string(best_phase) +
	      ", expected " + std::to_string(delay % sps));

	// The eye must be properly open there, and visibly narrower halfway
	// between symbol instants.
	const double best = opening[best_phase];
	const double worst = opening[(best_phase + sps / 2) % sps];
	check(best > 0.8, "eye opening at the correct instant is only " +
	      std::to_string(best) + ", expected > 0.8");
	check(worst < best * 0.6, "eye at the half-symbol offset (" +
	      std::to_string(worst) + ") is not clearly narrower than at the "
	      "sampling instant (" + std::to_string(best) + ")");

	std::cout << "  eye_opens_at_correct_instant: passed (opening "
	          << best << " vs " << worst << " mid-symbol)\n";
}

// ---------------------------------------------------------------------------
// End to end with the constellation module: QPSK through the shaped link
// recovers every symbol.
// ---------------------------------------------------------------------------
static void test_qpsk_link_round_trip() {
	using sw::dsp::sdr::Constellation;
	using sw::dsp::sdr::Modulation;

	const std::size_t sps = 4, span = 10, N = span * sps + 1;
	auto h = rrc_filter<double>(N, sps, 0.35);
	Constellation<double> c(Modulation::qpsk);

	std::mt19937 rng(77);
	std::uniform_int_distribution<std::size_t> pick(0, c.order() - 1);
	const std::size_t num_symbols = 300;
	std::vector<std::size_t> tx_idx(num_symbols);
	mtl::vec::dense_vector<double> si(num_symbols), sq(num_symbols);
	for (std::size_t i = 0; i < num_symbols; ++i) {
		tx_idx[i] = pick(rng);
		const auto s = c.symbol(tx_idx[i]);
		si[i] = s.real();
		sq[i] = s.imag();
	}

	// Shape and matched-filter I and Q independently.
	auto shape_and_match = [&](const mtl::vec::dense_vector<double>& sym) {
		sw::dsp::PolyphaseInterpolator<double> up(h, sps);
		auto wave = up.process_block(std::span<const double>(sym.data(), sym.size()));
		sw::dsp::FIRFilter<double> mf(h);
		mtl::vec::dense_vector<double> out(wave.size());
		for (std::size_t i = 0; i < wave.size(); ++i) out[i] = mf.process(wave[i]);
		return out;
	};
	auto ri = shape_and_match(si);
	auto rq = shape_and_match(sq);

	// Sample at the composite peak and demap.
	const std::size_t delay = N - 1;
	std::size_t errors = 0, counted = 0;
	for (std::size_t k = 0; k + 1 < num_symbols; ++k) {
		const std::size_t idx = delay + k * sps;
		if (idx >= ri.size()) break;
		// Skip the first few symbols while the filters fill.
		if (k < span) continue;
		const std::complex<double> r(ri[idx], rq[idx]);
		if (c.demap_hard(r) != tx_idx[k]) ++errors;
		++counted;
	}
	check(counted > 100, "too few symbols evaluated");
	check(errors == 0, "noiseless QPSK link had " + std::to_string(errors) +
	      " symbol errors out of " + std::to_string(counted));

	std::cout << "  qpsk_link_round_trip: passed (" << counted
	          << " symbols, 0 errors)\n";
}

// ---------------------------------------------------------------------------
// Precision sweep: residual ISI against coefficient precision.
//
// The finding worth recording is that the two error sources trade places.
// At a short span, TRUNCATION dominates and coefficient precision is
// irrelevant — double and posit16 give the same ISI to three digits. Lengthen
// the pulse and truncation falls away, at which point each type hits a floor
// set by its own quantization and stops improving:
//
//   span   double/float/posit32/cfloat32   posit16    posit8
//     16              1.289e-03           1.287e-03  4.880e-03
//     64              1.397e-04           1.397e-04  4.903e-03
//    256              3.533e-06           8.191e-05  4.904e-03
//
// So "how much precision do the coefficients need" has no answer on its own;
// it depends on how long the filter is. posit8 is quantization-limited at
// every length, posit16 from about span 128, and the 32-bit types never are
// over the range that matters.
// ---------------------------------------------------------------------------
template <typename T>
static double isi_for(const char* name, std::size_t span, std::size_t sps, double alpha) {
	auto h = rrc_filter<T>(span * sps + 1, sps, alpha);
	auto composite = convolve(h, h);
	const double isi = peak_isi(composite, sps);
	std::cout << "      " << name << ": peak ISI = " << isi << "\n";
	return isi;
}

static void test_precision_sweep() {
	using posit32  = sw::universal::posit<32, 2>;
	using posit16  = sw::universal::posit<16, 2>;
	using posit8   = sw::universal::posit<8, 2>;
	using cfloat32 = sw::universal::cfloat<32, 8, std::uint32_t, true, false, false>;

	const std::size_t sps = 4;
	const double alpha = 0.35;

	// --- short pulse: truncation dominates, precision does not matter ---
	std::cout << "    span 16 (truncation-limited):\n";
	const double s_d   = isi_for<double>  ("double  ", 16, sps, alpha);
	const double s_f   = isi_for<float>   ("float   ", 16, sps, alpha);
	const double s_p32 = isi_for<posit32> ("posit32 ", 16, sps, alpha);
	const double s_c32 = isi_for<cfloat32>("cfloat32", 16, sps, alpha);
	const double s_p16 = isi_for<posit16> ("posit16 ", 16, sps, alpha);
	const double s_p8  = isi_for<posit8>  ("posit8  ", 16, sps, alpha);

	check(s_d < 2e-3, "double ISI at span 16 unexpectedly high: " + std::to_string(s_d));
	for (auto [name, v] : {std::pair<const char*, double>{"float", s_f},
	                        {"posit32", s_p32}, {"cfloat32", s_c32},
	                        {"posit16", s_p16}}) {
		check(std::abs(v - s_d) < 0.02 * s_d, std::string(name) +
		      " should sit on the same truncation floor as double at span 16 (" +
		      std::to_string(v) + " vs " + std::to_string(s_d) + ")");
	}
	// posit8 is coarse enough to break through even here.
	check(s_p8 > 2.0 * s_d, "posit8 should already exceed the truncation floor "
	      "at span 16, got " + std::to_string(s_p8));

	// --- long pulse: truncation recedes and each type meets its own floor ---
	std::cout << "    span 256 (quantization-limited):\n";
	const double l_d   = isi_for<double>  ("double  ", 256, sps, alpha);
	const double l_p32 = isi_for<posit32> ("posit32 ", 256, sps, alpha);
	const double l_p16 = isi_for<posit16> ("posit16 ", 256, sps, alpha);
	const double l_p8  = isi_for<posit8>  ("posit8  ", 256, sps, alpha);

	// double keeps improving with length; the narrow types do not.
	check(l_d < s_d / 100.0, "double should improve sharply with span: " +
	      std::to_string(s_d) + " -> " + std::to_string(l_d));
	check(std::abs(l_p32 - l_d) < 0.02 * l_d,
	      "posit32 should still track double at span 256");
	check(l_p16 > 5.0 * l_d, "posit16 should now be quantization-limited, "
	      "well above double (" + std::to_string(l_p16) + " vs " +
	      std::to_string(l_d) + ")");
	check(l_p8 > l_p16, "posit8 must be worse than posit16 (" +
	      std::to_string(l_p8) + " vs " + std::to_string(l_p16) + ")");

	// posit8 is quantization-limited from the outset, so lengthening the
	// pulse buys it essentially nothing.
	check(std::abs(l_p8 - s_p8) < 0.2 * s_p8,
	      "posit8 ISI should be flat against span (" + std::to_string(s_p8) +
	      " -> " + std::to_string(l_p8) + ")");

	std::cout << "  precision_sweep: passed\n";
}

// ---------------------------------------------------------------------------
// Contract violations are reported
// ---------------------------------------------------------------------------
static void test_validation() {
	bool caught = false;
	try { rrc_filter<double>(32, 4, 0.25); }        // even
	catch (const std::invalid_argument&) { caught = true; }
	check(caught, "even num_taps should throw");

	caught = false;
	try { rrc_filter<double>(1, 4, 0.25); }
	catch (const std::invalid_argument&) { caught = true; }
	check(caught, "num_taps < 3 should throw");

	caught = false;
	try { rrc_filter<double>(33, 0, 0.25); }
	catch (const std::invalid_argument&) { caught = true; }
	check(caught, "samples_per_symbol 0 should throw");

	caught = false;
	try { rrc_filter<double>(33, 4, -0.1); }
	catch (const std::invalid_argument&) { caught = true; }
	check(caught, "negative rolloff should throw");

	caught = false;
	try { rrc_filter<double>(33, 4, 1.5); }
	catch (const std::invalid_argument&) { caught = true; }
	check(caught, "rolloff > 1 should throw");

	caught = false;
	try { raised_cosine_filter<double>(32, 4, 0.25); }
	catch (const std::invalid_argument&) { caught = true; }
	check(caught, "raised_cosine_filter should validate too");

	caught = false;
	try {
		mtl::vec::dense_vector<double> tiny(2, 1.0);
		peak_isi(tiny, 4);
	} catch (const std::invalid_argument&) { caught = true; }
	check(caught, "peak_isi with a too-short composite should throw");

	std::cout << "  validation: passed\n";
}

// ---------------------------------------------------------------------------

int main() {
	try {
		std::cout << "SDR RRC pulse-shaping tests\n";
		test_reference_formula();
		test_structure();
		test_zero_isi();
		test_composite_is_raised_cosine();
		test_eye_opens_at_correct_instant();
		test_qpsk_link_round_trip();
		test_precision_sweep();
		test_validation();
		std::cout << "All SDR RRC tests passed.\n";
		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << "\n";
		return 1;
	}
}
