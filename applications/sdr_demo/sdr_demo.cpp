// sdr_demo.cpp: end-to-end SDR link, swept across number systems
//
// The capstone for the Software Defined Radio epic (#85). It runs the full
// transmit/receive chain
//
//   bits -> constellation map -> RRC pulse shaping (upsample x sps)
//        -> AWGN -> RRC matched filter -> symbol sampling -> demap -> bits
//
// with the three-scalar model applied to the shaping and matched filters, so
// the coefficients, the accumulators and the samples each carry the number
// system under test. Unlike analysis/sdr_precision.hpp — which PROJECTS one
// block's coefficients into a narrow type and keeps the surrounding
// arithmetic at double, so a result is attributable to that block — this demo
// runs the arithmetic itself in the narrow type. That is the harder question
// and the one an implementer actually faces: not "what does quantizing these
// taps cost" but "what happens if the DSP runs in this format".
//
// Both views appear below. The sweep is the narrow-arithmetic chain; the
// closing section calls analysis::analyze_blocks() for the per-block
// attribution, which needs the projection model to mean anything.
//
// WHAT MAKES A CONFIGURATION USABLE is implementation loss, not raw EVM.
// Every configuration carries the same RRC truncation ISI and the same
// channel noise, and at a working Eb/N0 that floor dwarfs the arithmetic.
// Comparing a narrow chain's EVM against the double chain's EVM at the same
// operating point turns the two into a single number,
//
//   loss_dB = 10*log10( EVM_narrow^2 / EVM_double^2 )
//
// which is the extra Eb/N0 the narrow implementation needs to reach the same
// error rate. A configuration is called usable when that stays under 1 dB.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/analysis/sdr_precision.hpp>
#include <sw/dsp/filter/fir/fir_filter.hpp>
#include <sw/dsp/filter/fir/polyphase.hpp>
#include <sw/dsp/sdr/constellation.hpp>
#include <sw/dsp/sdr/metrics.hpp>
#include <sw/dsp/sdr/rrc.hpp>

#include <common/demo_output.hpp>

#if __has_include(<bit>)
#include <bit>
#endif
#include <sw/universal/number/cfloat/cfloat.hpp>
#include <sw/universal/number/fixpnt/fixpnt.hpp>
#include <sw/universal/number/posit/posit.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace sw::dsp;
using sw::dsp::sdr::Modulation;

// ============================================================================
// Number systems under test
// ============================================================================
//
// Four families at four widths. The cfloat parameters are chosen so the
// 16- and 32-bit members ARE the IEEE binary16 and binary32 formats, which
// makes the cfloat column a continuation of the IEEE column rather than a
// separate story; the 12- and 8-bit members are the natural interpolations.
//
// Fixed-point needs its binary point placed by hand. The shaped waveform
// peaks around 0.8, but the sample type also has to hold the channel noise,
// whose tail at the low-order operating points reaches several sigma on top
// of that — so every fixpnt member keeps two integer bits and spends the rest
// on fraction. The demo prints the resulting clipping rate, which is how you
// check the placement rather than assume it. All of them saturate rather than
// wrap: a clipped sample is a recoverable impairment, a wrapped one is not.

using cf32 = sw::universal::cfloat<32, 8, std::uint32_t, true, false, false>;
using cf16 = sw::universal::cfloat<16, 5, std::uint16_t, true, false, false>;
using cf12 = sw::universal::cfloat<12, 4, std::uint16_t, true, false, false>;
using cf8  = sw::universal::cfloat<8,  4, std::uint8_t,  true, false, false>;

using p32   = sw::universal::posit<32, 2>;
using p16   = sw::universal::posit<16, 2>;
using p12   = sw::universal::posit<12, 2>;
using p8    = sw::universal::posit<8,  2>;
using p8e0  = sw::universal::posit<8,  0>;

using q32 = sw::universal::fixpnt<32, 29, sw::universal::Saturate, std::uint32_t>;
using q16 = sw::universal::fixpnt<16, 13, sw::universal::Saturate, std::uint16_t>;
using q12 = sw::universal::fixpnt<12,  9, sw::universal::Saturate, std::uint16_t>;
using q8  = sw::universal::fixpnt<8,   5, sw::universal::Saturate, std::uint8_t>;

// ============================================================================
// Parameters
// ============================================================================

struct DemoParams {
	std::size_t samples_per_symbol = 4;
	double      rolloff            = 0.35;
	std::size_t rrc_span           = 10;      // symbols
	std::size_t num_symbols        = 2000;
	// Symbols for the reference-vs-theory check only. A BER of 1e-3 needs
	// something like 1e5 bits before the count stops being dominated by its
	// own Poisson spread, which is far more than the narrow sweep can afford
	// — but the reference chain runs in double, where it is nearly free.
	std::size_t ber_symbols        = 60000;
	unsigned    seed               = 1;
	// Channel used for the arithmetic-only measurement. Loud enough that the
	// arithmetic, not the channel, is what the EVM is reporting.
	double      quiet_ebn0_db      = 80.0;
	// The BER the operating point is defined by. Solved per modulation from
	// the closed form, so every scheme is compared at the same place on its
	// own curve rather than at an arbitrary shared Eb/N0.
	double      target_ber         = 1e-3;
	// Implementation loss a configuration may spend and still count as usable.
	double      loss_budget_db     = 1.0;
};

const Modulation kModulations[] = {
	Modulation::qpsk, Modulation::qam16, Modulation::qam64, Modulation::qam256
};

// ============================================================================
// The link
// ============================================================================

struct ChainResult {
	double      evm_rms    = 0.0;
	double      evm_db     = 0.0;
	double      mer_db     = 0.0;
	std::size_t bit_errors = 0;
	std::size_t total_bits = 0;
	double      ber        = 0.0;
	// Fraction of transmit samples the sample type could not hold. Always
	// zero for a floating-point type, whose range is nowhere near exhausted
	// by a normalized waveform; the number exists for fixed-point, where the
	// binary point is placed by hand and getting it wrong shows up here
	// rather than as a mysteriously bad EVM.
	double      clip_fraction = 0.0;
	double      peak_tx       = 0.0;   // largest transmit sample, absolute
	std::vector<std::complex<double>> constellation;
};

// Run one link with CoeffScalar/StateScalar/SampleScalar applied to both
// filters and to the sample stream between them.
//
// `tx_scale` backs the whole waveform off from full scale — signal and noise
// together, since a level change ahead of the converter moves both and leaves
// Eb/N0 alone. It is how an un-AGC'd front end sees a weak signal, and the
// measurement compensates for it before computing EVM, so what the number
// reports is purely the arithmetic's ability to carry a small signal.
//
// Pre:  samples_per_symbol >= 2, num_symbols > 4*rrc_span, tx_scale > 0.
// Post: EVM, MER and BER over the settled portion of the stream, plus the
//       received constellation and the transmit-side clipping.
template <typename Coeff, typename State, typename Sample>
ChainResult run_chain(const DemoParams& p, Modulation mod, double ebn0_db,
                      double tx_scale = 1.0) {
	if (p.samples_per_symbol < 2)
		throw std::invalid_argument("run_chain: samples_per_symbol must be >= 2");
	if (p.num_symbols <= 4 * p.rrc_span)
		throw std::invalid_argument(
			"run_chain: num_symbols must exceed 4*rrc_span so the measurement "
			"window excludes the filter transients");
	if (!(tx_scale > 0.0))
		throw std::invalid_argument("run_chain: tx_scale must be positive");

	const std::size_t sps   = p.samples_per_symbol;
	const std::size_t ntaps = p.rrc_span * sps + 1;

	// The constellation and the ideal reference stay in double: quantizing
	// the table is part of what the sample type costs, so folding it into
	// the reference would hide exactly what is being measured.
	sdr::Constellation<double> map(mod);
	const std::size_t nb = map.bits_per_symbol();

	// Designed in double and projected to Coeff by rrc_filter itself, which
	// keeps a narrow tap set as close to the ideal pulse as its own precision
	// allows instead of compounding design error onto representation error.
	auto taps = sdr::rrc_filter<Coeff>(ntaps, sps, p.rolloff);

	std::mt19937 rng(p.seed);
	std::uniform_int_distribution<std::size_t> pick(0, map.order() - 1);
	mtl::vec::dense_vector<Sample> si(p.num_symbols), sq(p.num_symbols);
	std::vector<std::complex<double>> ideal(p.num_symbols);
	std::vector<std::uint8_t> tx_bits, tmp(nb);
	tx_bits.reserve(p.num_symbols * nb);

	for (std::size_t n = 0; n < p.num_symbols; ++n) {
		const std::size_t k = pick(rng);
		const auto s = map.symbol(k);
		si[n] = static_cast<Sample>(s.real() * tx_scale);
		sq[n] = static_cast<Sample>(s.imag() * tx_scale);
		ideal[n] = std::complex<double>(s.real(), s.imag());
		map.bits_of(k, tmp);
		tx_bits.insert(tx_bits.end(), tmp.begin(), tmp.end());
	}

	// Per-dimension noise variance is N0/2 with NO samples-per-symbol factor.
	// Both RRC filters carry unit energy, so the transmit filter puts Es = 1
	// into each symbol and the matched filter passes the noise variance
	// through unchanged; the symbol-rate SNR is the sample-rate SNR.
	const double esn0_db = sdr::esn0_db_from_ebn0_db(mod, ebn0_db);
	const double n0      = std::pow(10.0, -esn0_db / 10.0);
	std::normal_distribution<double> gauss(0.0, std::sqrt(n0 / 2.0) * tx_scale);

	const double sample_max = static_cast<double>(std::numeric_limits<Sample>::max());
	double      peak    = 0.0;
	std::size_t clipped = 0, sample_count = 0;

	auto shape_and_match = [&](const mtl::vec::dense_vector<Sample>& sym) {
		PolyphaseInterpolator<Coeff, State, Sample> up(taps, sps);
		auto wave = up.process_block(std::span<const Sample>(sym.data(), sym.size()));
		FIRFilter<Coeff, State, Sample> mf(taps);
		mtl::vec::dense_vector<Sample> out(wave.size());
		for (std::size_t i = 0; i < wave.size(); ++i) {
			// Clipping is counted on the value the channel hands the sample
			// type, noise included: a converter that saturates on peaks does
			// so on signal-plus-noise, and counting only the clean waveform
			// would miss exactly the case that matters.
			const double v = static_cast<double>(wave[i]) + gauss(rng);
			peak = std::max(peak, std::abs(v));
			if (std::abs(v) >= sample_max) ++clipped;
			++sample_count;
			out[i] = mf.process(static_cast<Sample>(v));
		}
		return out;
	};
	auto ri = shape_and_match(si);
	auto rq = shape_and_match(sq);

	const std::size_t delay = ntaps - 1;          // TX + RX group delay
	const std::size_t skip  = 2 * p.rrc_span;     // let both filters settle

	std::vector<std::complex<double>> got, want;
	std::vector<std::uint8_t> rx_bits, want_bits;
	for (std::size_t k = skip; k + skip < p.num_symbols; ++k) {
		const std::size_t s = delay + k * sps;
		if (s >= ri.size()) break;
		// Undo the backoff before measuring: recovering the level is the
		// AGC's job, not the arithmetic's, and leaving it in would report a
		// gain error as if it were precision loss.
		const std::complex<double> y(static_cast<double>(ri[s]) / tx_scale,
		                             static_cast<double>(rq[s]) / tx_scale);
		got.push_back(y);
		want.push_back(ideal[k]);
		map.demap_hard_bits(y, tmp);
		rx_bits.insert(rx_bits.end(), tmp.begin(), tmp.end());
		want_bits.insert(want_bits.end(),
		                 tx_bits.begin() + static_cast<long>(k * nb),
		                 tx_bits.begin() + static_cast<long>((k + 1) * nb));
	}
	if (got.empty())
		throw std::runtime_error("run_chain: no symbols survived the transients");

	ChainResult r;
	const auto e = sdr::evm<std::complex<double>>(want, got);
	r.evm_rms = e.rms;
	r.evm_db  = e.rms_db;
	r.mer_db  = -e.rms_db;
	const auto b = sdr::ber(want_bits, rx_bits);
	r.bit_errors    = b.bit_errors;
	r.total_bits    = b.total_bits;
	r.ber           = b.rate;
	r.peak_tx       = peak;
	r.clip_fraction = sample_count ? static_cast<double>(clipped) /
	                                 static_cast<double>(sample_count) : 0.0;
	r.constellation = std::move(got);
	return r;
}

// ============================================================================
// Sweep rows
// ============================================================================

struct SweepRow {
	std::string config;        // "posit<16,2>"
	std::string family;        // "posit"
	int         bit_width = 0;
	Modulation  modulation = Modulation::qam16;

	double evm_quiet      = 0.0;   // EVM on a near-noiseless channel
	double evm_arith      = 0.0;   // quiet EVM with the double floor removed
	double ebn0_op_db     = 0.0;   // operating point for this modulation
	double evm_op         = 0.0;   // EVM there
	double loss_db        = 0.0;   // implementation loss vs the double chain
	double ber_op         = 0.0;
	std::size_t bit_errors = 0;
	std::size_t total_bits = 0;
	double ber_theory     = 0.0;
	double clip_fraction  = 0.0;
	bool   usable         = false;
};

// The all-double chain at each modulation: the floor every narrow chain is
// measured against.
struct Reference {
	double ebn0_op_db = 0.0;
	// Measured with the sweep's symbol count and seed, so a narrow chain sees
	// the SAME noise realization. Most of the sampling variance then cancels
	// in the ratio and the implementation loss is a difference of arithmetic
	// rather than a difference of draws.
	double      evm_quiet  = 0.0;
	double      evm_op     = 0.0;
	// The theory check, run separately over many more symbols because a BER
	// of 1e-3 needs them and the double chain can afford them.
	double      ber_check  = 0.0;
	std::size_t ber_bits   = 0;
	std::size_t ber_errors = 0;
};

// Eb/N0 at which the closed form predicts `target` for this modulation.
// Monotone in Eb/N0, so bisection is exact to the width of the bracket.
double ebn0_for_ber(Modulation m, double target) {
	double lo = -10.0, hi = 60.0;
	for (int i = 0; i < 200; ++i) {
		const double mid = 0.5 * (lo + hi);
		if (sdr::theoretical_ber_awgn(m, mid) > target) lo = mid;
		else                                            hi = mid;
	}
	return 0.5 * (lo + hi);
}

// Error left after removing a common floor, subtracted in POWER. The floor
// here is the double chain's own EVM — RRC truncation ISI plus whatever the
// channel contributed — and it is routinely larger than the arithmetic being
// measured, so a raw EVM comparison says nothing.
double excess_evm(double narrow, double reference) {
	const double d = narrow * narrow - reference * reference;
	return (d > 0.0) ? std::sqrt(d) : 0.0;
}

// ============================================================================
// Console output
// ============================================================================

std::string fmt_sci(double v, int prec = 2) {
	std::ostringstream os;
	if (v <= 0.0) { os << "0"; return os.str(); }
	os << std::scientific << std::setprecision(prec) << v;
	return os.str();
}

void print_sweep(Modulation m, const std::vector<SweepRow>& rows,
                 const Reference& ref, double budget_db) {
	std::cout << "\n" << std::string(104, '=') << "\n";
	std::cout << "  " << sdr::to_string(m)
	          << "   operating point Eb/N0 = " << std::fixed << std::setprecision(2)
	          << ref.ebn0_op_db << " dB (double chain BER "
	          << fmt_sci(ref.ber_check) << ", EVM " << std::fixed
	          << std::setprecision(4) << ref.evm_op << ")\n";
	std::cout << std::string(104, '=') << "\n";
	std::cout << std::left  << std::setw(20) << "Number system"
	          << std::right << std::setw(8)  << "bits"
	          << std::right << std::setw(13) << "EVM arith"
	          << std::right << std::setw(12) << "EVM @op"
	          << std::right << std::setw(11) << "loss(dB)"
	          << std::right << std::setw(12) << "BER"
	          << std::right << std::setw(9)  << "errors"
	          << std::right << std::setw(9)  << "clip%"
	          << std::right << std::setw(9)  << "usable"
	          << "\n" << std::string(104, '-') << "\n";

	for (const auto& r : rows) {
		if (r.modulation != m) continue;
		std::cout << std::left  << std::setw(20) << r.config
		          << std::right << std::setw(8)  << r.bit_width
		          << std::right << std::setw(13) << fmt_sci(r.evm_arith)
		          << std::right << std::setw(12) << std::fixed << std::setprecision(4) << r.evm_op
		          << std::right << std::setw(11) << std::fixed << std::setprecision(2) << r.loss_db
		          << std::right << std::setw(12) << fmt_sci(r.ber_op)
		          << std::right << std::setw(9)  << r.bit_errors
		          << std::right << std::setw(9)  << std::fixed << std::setprecision(2)
		                                         << 100.0 * r.clip_fraction
		          << std::right << std::setw(9)  << (r.usable ? "yes" : "NO")
		          << "\n";
	}
	std::size_t bits_here = 0;
	for (const auto& r : rows)
		if (r.modulation == m) { bits_here = r.total_bits; break; }

	std::cout << std::string(104, '-') << "\n";
	std::cout << "  usable = implementation loss <= " << std::fixed << std::setprecision(1)
	          << budget_db << " dB, measured on EVM. The BER column is the same "
	             "run's error count over\n"
	             "  " << bits_here
	          << " bits — indicative only at this rate, which is why usability "
	             "is decided on EVM.\n";
	std::cout << "  clip% = transmit samples the sample type could not hold "
	             "(fixed-point scaling check).\n";
}

// The claim the epic rests on: at a fixed bit width, which family carries the
// densest constellation. Printed as the highest usable modulation per cell, so
// a family that buys an extra step shows it directly.
void print_modulation_ceiling(const std::vector<SweepRow>& rows) {
	std::cout << "\n" << std::string(104, '=') << "\n";
	std::cout << "  Highest usable modulation, by number system and bit width\n";
	std::cout << std::string(104, '=') << "\n";

	std::map<std::pair<std::string, int>, Modulation> best;
	std::map<std::pair<std::string, int>, bool> any;
	for (const auto& r : rows) {
		const auto key = std::make_pair(r.family, r.bit_width);
		if (!r.usable) { any.emplace(key, false); continue; }
		any[key] = true;
		auto it = best.find(key);
		if (it == best.end() ||
		    sdr::bits_per_symbol(r.modulation) > sdr::bits_per_symbol(it->second))
			best[key] = r.modulation;
	}

	std::cout << std::left << std::setw(20) << "Family";
	for (int b : {8, 12, 16, 32})
		std::cout << std::right << std::setw(14) << (std::to_string(b) + "-bit");
	std::cout << "\n" << std::string(76, '-') << "\n";

	for (const char* fam : {"IEEE", "cfloat", "posit", "fixpnt"}) {
		std::cout << std::left << std::setw(20) << fam;
		for (int b : {8, 12, 16, 32}) {
			const auto key = std::make_pair(std::string(fam), b);
			auto it = best.find(key);
			if (it != best.end())
				std::cout << std::right << std::setw(14) << sdr::to_string(it->second);
			else if (any.count(key))
				std::cout << std::right << std::setw(14) << "none";
			else
				std::cout << std::right << std::setw(14) << "-";
		}
		std::cout << "\n";
	}
	std::cout << std::string(76, '-') << "\n";
	std::cout << "  \"none\" = the family was measured at that width and carried no\n"
	             "  modulation within the loss budget; \"-\" = not in the sweep.\n"
	             "  The IEEE row is float at 32 bits and double as the reference;\n"
	             "  cfloat<32,8> and cfloat<16,5> ARE binary32 and binary16, so the\n"
	             "  cfloat row continues the IEEE one below 32 bits.\n"
	             "  256-QAM is the top of this sweep, not a ceiling of the format.\n";
}

// ============================================================================
// CSV export
// ============================================================================

void write_sweep_csv(const std::string& path, const std::vector<SweepRow>& rows) {
	std::ofstream out(path);
	if (!out) throw std::runtime_error("cannot open " + path);
	out << "pipeline,config,family,bit_width,modulation,evm_quiet,evm_arith,"
	       "ebn0_op_db,evm_op,loss_db,ber_op,bit_errors,total_bits,ber_theory,"
	       "clip_fraction,usable\n";
	out << std::setprecision(15);
	for (const auto& r : rows) {
		out << "sdr_demo," << r.config << "," << r.family << "," << r.bit_width
		    << "," << sdr::to_string(r.modulation) << "," << r.evm_quiet << ","
		    << r.evm_arith << "," << r.ebn0_op_db << "," << r.evm_op << ","
		    << r.loss_db << "," << r.ber_op << "," << r.bit_errors << ","
		    << r.total_bits << "," << r.ber_theory << ","
		    << r.clip_fraction << "," << (r.usable ? 1 : 0) << "\n";
	}
}

// ============================================================================
// Dynamic range
// ============================================================================
//
// The sweep above runs every configuration at full scale, which is the case a
// fixed-point design is scaled for and the case posit's tapered precision has
// nothing to offer. Backing the input off is the other half of the picture:
// it is what a receiver sees before its AGC settles, or on a weak signal in a
// system sized for a strong one, and it separates a format that spends its
// bits uniformly across a fixed range from one that keeps relative precision
// as the signal shrinks.

struct BackoffRow {
	std::string config;
	int         bit_width = 0;
	std::vector<double> evm;      // one entry per backoff level
};

const double kBackoffDb[] = {0.0, -12.0, -24.0, -36.0, -48.0};

void print_backoff(const std::vector<BackoffRow>& rows) {
	std::cout << "\n" << std::string(104, '=') << "\n";
	std::cout << "  Input backoff — 16-QAM, quiet channel, EVM after the level is "
	             "restored\n";
	std::cout << std::string(104, '=') << "\n";
	std::cout << std::left << std::setw(20) << "Number system"
	          << std::right << std::setw(8) << "bits";
	for (double db : kBackoffDb) {
		std::ostringstream h; h << std::fixed << std::setprecision(0) << db << " dB";
		std::cout << std::right << std::setw(13) << h.str();
	}
	std::cout << "\n" << std::string(93, '-') << "\n";
	for (const auto& r : rows) {
		std::cout << std::left << std::setw(20) << r.config
		          << std::right << std::setw(8) << r.bit_width;
		for (double e : r.evm) std::cout << std::right << std::setw(13) << fmt_sci(e);
		std::cout << "\n";
	}
	std::cout << std::string(93, '-') << "\n";
	std::cout << "  The backoff attenuates signal and noise together, so Eb/N0 is "
	             "unchanged and the\n  receiver's gain is restored before the "
	             "measurement. Every column therefore reports\n  the same link, "
	             "differing only in where the waveform sits inside the number "
	             "format.\n";
}

// ============================================================================
// Findings
// ============================================================================
//
// Derived from the measurements rather than asserted, so changing the sweep
// parameters changes the conclusion instead of leaving a stale claim behind.

// Deepest backoff at which a format still held its full-scale EVM to within a
// factor of two. Reported in dB below full scale; 0 means it lost the factor
// immediately.
double usable_range_db(const BackoffRow& r) {
	if (r.evm.empty() || !(r.evm.front() > 0.0)) return 0.0;
	const double limit = 2.0 * r.evm.front();
	double deepest = 0.0;
	for (std::size_t i = 0; i < r.evm.size(); ++i)
		if (r.evm[i] <= limit) deepest = std::min(deepest, kBackoffDb[i]);
	// Negating 0.0 yields -0.0, which prints as "-0"; a format that fails at
	// the very first backoff step should read as a range of zero.
	return deepest < 0.0 ? -deepest : 0.0;
}

void print_findings(const std::vector<SweepRow>& rows,
                    const std::vector<BackoffRow>& backoff) {
	std::cout << "\n" << std::string(104, '=') << "\n";
	std::cout << "  What the sweep found\n";
	std::cout << std::string(104, '=') << "\n";

	// 1. At each bit width, which family carried the densest constellation.
	std::cout << "\n  At full scale, the densest constellation each width "
	             "carried:\n";
	for (int w : {8, 12, 16, 32}) {
		std::size_t best_bps = 0;
		std::vector<std::string> winners;
		for (const auto& r : rows) {
			if (r.bit_width != w || !r.usable) continue;
			const auto bps = sdr::bits_per_symbol(r.modulation);
			if (bps > best_bps) { best_bps = bps; winners.clear(); }
			if (bps == best_bps &&
			    std::find(winners.begin(), winners.end(), r.config) == winners.end())
				winners.push_back(r.config);
		}
		std::cout << "    " << std::setw(2) << w << "-bit: ";
		if (winners.empty()) { std::cout << "nothing within the loss budget\n"; continue; }
		std::cout << best_bps << " bits/symbol —";
		for (const auto& c : winners) std::cout << " " << c;
		std::cout << "\n";
	}
	std::cout <<
		"\n  Fixed-point holding its own here is the expected result, not an\n"
		"  anomaly. The link is amplitude-normalized end to end, so the whole\n"
		"  waveform lives in one octave and a uniform absolute step is exactly\n"
		"  what an EVM measurement rewards. Posit's tapered precision buys\n"
		"  nothing when nothing needs the dynamic range.\n";

	// 2. What that changes once the signal is not at full scale.
	std::cout << "\n  Backoff each format tolerated before its EVM doubled:\n";
	for (const auto& r : backoff)
		std::cout << "    " << std::left << std::setw(16) << r.config
		          << std::right << std::setw(6) << std::fixed << std::setprecision(0)
		          << usable_range_db(r) << " dB\n";
	std::cout <<
		"\n  That is where the families separate. A normalized full-scale link\n"
		"  is the best case for fixed-point and the worst case for posit; move\n"
		"  the signal off full scale — an un-settled AGC, a weak carrier, a\n"
		"  system sized for the strong one — and the uniform grid runs out\n"
		"  while the tapered one keeps its relative precision.\n";
}

// The Pareto view: cost in bits against delivered spectral efficiency. A
// point is on the frontier when nothing in the sweep carries at least as many
// bits per symbol for fewer bits of arithmetic.
void write_pareto_csv(const std::string& path, const std::vector<SweepRow>& rows) {
	std::ofstream out(path);
	if (!out) throw std::runtime_error("cannot open " + path);
	out << "config,family,bit_width,modulation,bits_per_symbol,evm_arith,"
	       "loss_db,on_frontier\n";
	out << std::setprecision(15);
	for (const auto& r : rows) {
		if (!r.usable) continue;
		const auto bps = sdr::bits_per_symbol(r.modulation);
		bool dominated = false;
		for (const auto& o : rows) {
			if (!o.usable) continue;
			if (o.bit_width <= r.bit_width &&
			    sdr::bits_per_symbol(o.modulation) >= bps &&
			    (o.bit_width < r.bit_width ||
			     sdr::bits_per_symbol(o.modulation) > bps)) {
				dominated = true;
				break;
			}
		}
		out << r.config << "," << r.family << "," << r.bit_width << ","
		    << sdr::to_string(r.modulation) << "," << bps << "," << r.evm_arith
		    << "," << r.loss_db << "," << (dominated ? 0 : 1) << "\n";
	}
}

// Received clouds for plotting, stacked with a tag column so one file holds
// every configuration worth looking at.
struct ConstellationDump {
	std::string config;
	Modulation  modulation;
	double      ebn0_db;
	const std::vector<std::complex<double>>* points;
};

void write_constellations_csv(const std::string& path,
                              const std::vector<ConstellationDump>& dumps) {
	std::ofstream out(path);
	if (!out) throw std::runtime_error("cannot open " + path);
	out << "config,modulation,ebn0_db,i,q\n";
	out << std::setprecision(9);
	for (const auto& d : dumps)
		for (const auto& s : *d.points)
			out << d.config << "," << sdr::to_string(d.modulation) << ","
			    << d.ebn0_db << "," << s.real() << "," << s.imag() << "\n";
}

// ============================================================================
// Argument handling
// ============================================================================

void print_usage(const char* argv0) {
	std::cout <<
		"Usage: " << argv0 << " [options]\n"
		"  --symbols=N        symbols per measurement   (default 2000)\n"
		"  --ber-symbols=N    symbols for the double-chain theory check (default 60000)\n"
		"  --sps=N            samples per symbol        (default 4)\n"
		"  --rolloff=X        RRC excess bandwidth      (default 0.35)\n"
		"  --span=N           RRC span in symbols       (default 10)\n"
		"  --quiet-ebn0=X     Eb/N0 for the arithmetic-only pass (default 80 dB)\n"
		"  --target-ber=X     BER defining the operating point   (default 1e-3)\n"
		"  --loss-budget=X    implementation loss allowed, dB    (default 1.0)\n"
		"  --seed=N           PRNG seed                 (default 1)\n"
		"  --csv-dir=PATH     where the CSV files go\n"
		"  --help\n";
}

double parse_double(const std::string& s, const char* flag) {
	try {
		std::size_t used = 0;
		const double v = std::stod(s, &used);
		if (used != s.size()) throw std::invalid_argument("trailing characters");
		return v;
	} catch (const std::exception&) {
		throw std::invalid_argument(std::string(flag) + ": not a number: " + s);
	}
}

std::size_t parse_size(const std::string& s, const char* flag) {
	try {
		std::size_t used = 0;
		const unsigned long long v = std::stoull(s, &used);
		if (used != s.size()) throw std::invalid_argument("trailing characters");
		return static_cast<std::size_t>(v);
	} catch (const std::exception&) {
		throw std::invalid_argument(std::string(flag) + ": not an integer: " + s);
	}
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
	DemoParams p;
	std::string csv_dir = sw::dsp::demo::output_dir();

	try {
		for (int i = 1; i < argc; ++i) {
			const std::string a = argv[i];
			if (a == "--help" || a == "-h") { print_usage(argv[0]); return 0; }
			else if (a.rfind("--symbols=", 0) == 0)     p.num_symbols = parse_size(a.substr(10), "--symbols");
			else if (a.rfind("--ber-symbols=", 0) == 0) p.ber_symbols = parse_size(a.substr(14), "--ber-symbols");
			else if (a.rfind("--sps=", 0) == 0)         p.samples_per_symbol = parse_size(a.substr(6), "--sps");
			else if (a.rfind("--rolloff=", 0) == 0)     p.rolloff = parse_double(a.substr(10), "--rolloff");
			else if (a.rfind("--span=", 0) == 0)        p.rrc_span = parse_size(a.substr(7), "--span");
			else if (a.rfind("--quiet-ebn0=", 0) == 0)  p.quiet_ebn0_db = parse_double(a.substr(13), "--quiet-ebn0");
			else if (a.rfind("--target-ber=", 0) == 0)  p.target_ber = parse_double(a.substr(13), "--target-ber");
			else if (a.rfind("--loss-budget=", 0) == 0) p.loss_budget_db = parse_double(a.substr(14), "--loss-budget");
			else if (a.rfind("--seed=", 0) == 0)        p.seed = static_cast<unsigned>(parse_size(a.substr(7), "--seed"));
			else if (a.rfind("--csv-dir=", 0) == 0)     csv_dir = a.substr(10);
			else { std::cerr << "Unknown argument: " << a << "\n"; print_usage(argv[0]); return 1; }
		}
		if (p.samples_per_symbol < 2)
			throw std::invalid_argument("--sps must be >= 2");
		if (!(p.rolloff > 0.0 && p.rolloff <= 1.0))
			throw std::invalid_argument("--rolloff must be in (0, 1]");
		if (p.rrc_span == 0)
			throw std::invalid_argument("--span must be >= 1");
		if (p.num_symbols <= 4 * p.rrc_span)
			throw std::invalid_argument("--symbols must exceed 4*span");
		if (p.ber_symbols <= 4 * p.rrc_span)
			throw std::invalid_argument("--ber-symbols must exceed 4*span");
		if (!(p.target_ber > 0.0 && p.target_ber < 0.5))
			throw std::invalid_argument("--target-ber must be in (0, 0.5)");
		if (!(p.loss_budget_db > 0.0))
			throw std::invalid_argument("--loss-budget must be positive");
	} catch (const std::exception& ex) {
		std::cerr << "Error: " << ex.what() << "\n";
		print_usage(argv[0]);
		return 1;
	}
	if (!csv_dir.empty() && csv_dir.back() == '/') csv_dir.pop_back();

	try {
		std::cout << std::string(104, '=') << "\n";
		std::cout << "  SDR Link — Mixed-Precision Sweep (Issue #104)\n";
		std::cout << "  TX: bits -> map -> RRC x" << p.samples_per_symbol
		          << "  |  channel: AWGN  |  RX: RRC matched -> sample -> demap\n";
		std::cout << "  RRC: span " << p.rrc_span << " symbols, rolloff "
		          << p.rolloff << ", " << (p.rrc_span * p.samples_per_symbol + 1)
		          << " taps, unit energy\n";
		std::cout << "  " << p.num_symbols << " symbols per measurement, seed "
		          << p.seed << "\n";
		std::cout << "  Coefficients, accumulators and samples all run in the "
		             "number system under test.\n";
		std::cout << std::string(104, '=') << "\n";

		// --- reference chain --------------------------------------------
		//
		// Run first and validated against the closed form. Every precision
		// claim below is a difference from these numbers, so if the double
		// chain does not track theory nothing after it means anything.
		std::map<Modulation, Reference> refs;
		DemoParams ber_p = p;
		ber_p.num_symbols = p.ber_symbols;

		std::cout << "\n  Reference (all-double) chain vs. AWGN theory  ("
		          << p.ber_symbols << " symbols)\n";
		std::cout << "  " << std::string(88, '-') << "\n";
		std::cout << "  " << std::left << std::setw(12) << "Modulation"
		          << std::right << std::setw(14) << "Eb/N0 op(dB)"
		          << std::right << std::setw(14) << "BER measured"
		          << std::right << std::setw(14) << "BER theory"
		          << std::right << std::setw(10) << "ratio"
		          << std::right << std::setw(12) << "EVM quiet"
		          << "\n";
		for (Modulation m : kModulations) {
			Reference ref;
			ref.ebn0_op_db = ebn0_for_ber(m, p.target_ber);
			ref.evm_quiet  = run_chain<double, double, double>(p, m, p.quiet_ebn0_db).evm_rms;
			ref.evm_op     = run_chain<double, double, double>(p, m, ref.ebn0_op_db).evm_rms;

			const auto check = run_chain<double, double, double>(ber_p, m, ref.ebn0_op_db);
			ref.ber_check  = check.ber;
			ref.ber_bits   = check.total_bits;
			ref.ber_errors = check.bit_errors;
			refs[m] = ref;

			const double theory = sdr::theoretical_ber_awgn(m, ref.ebn0_op_db);
			std::cout << "  " << std::left << std::setw(12) << sdr::to_string(m)
			          << std::right << std::setw(14) << std::fixed << std::setprecision(2) << ref.ebn0_op_db
			          << std::right << std::setw(14) << fmt_sci(ref.ber_check)
			          << std::right << std::setw(14) << fmt_sci(theory)
			          << std::right << std::setw(10) << std::fixed << std::setprecision(2)
			                                         << (theory > 0.0 ? ref.ber_check / theory : 0.0)
			          << std::right << std::setw(12) << fmt_sci(ref.evm_quiet)
			          << "\n";
		}
		std::cout << "  " << std::string(88, '-') << "\n";
		std::cout << "  Everything below is a difference from these numbers, so the\n"
		             "  reference is checked against the closed form first. The ratio\n"
		             "  scatters around 1 by roughly the Poisson spread of the error\n"
		             "  count itself -- a few hundred errors is a relative sigma of a\n"
		             "  few percent, so the low-order rows land on either side of 1 and\n"
		             "  mean nothing individually. The one systematic effect is at the\n"
		             "  top: the closed form assumes an untruncated matched filter, and\n"
		             "  the residual ISI of a 10-symbol RRC costs a fraction of a dB,\n"
		             "  which only 256-QAM's steep BER slope turns into a visible\n"
		             "  excess. No large or slope-shaped departure is what says the\n"
		             "  link is right.\n";
		std::cout << "  The quiet-channel EVM is that truncation ISI on its own: the\n"
		             "  floor every configuration carries, removed in power before any\n"
		             "  arithmetic contribution is reported.\n";

		// --- the sweep ---------------------------------------------------
		std::vector<SweepRow> rows;
		std::vector<ConstellationDump> dumps;
		// Clouds are kept only for the configurations worth plotting; holding
		// every one would be tens of megabytes of points nobody looks at.
		std::map<std::string, std::vector<std::complex<double>>> kept;
		const char* kPlotted[] = {"double", "posit<16,2>", "cfloat<16,5>",
		                          "fixpnt<16,13>", "posit<8,2>", "cfloat<8,4>",
		                          "fixpnt<8,5>"};

		std::size_t num_configs = 0;
		auto add = [&](auto tag, const char* name, const char* family, int bits) {
			using T = decltype(tag);
			++num_configs;
			std::cout << "  measuring " << std::left << std::setw(16) << name
			          << std::flush;
			for (Modulation m : kModulations) {
				const Reference& ref = refs.at(m);
				SweepRow r;
				r.config     = name;
				r.family     = family;
				r.bit_width  = bits;
				r.modulation = m;
				r.ebn0_op_db = ref.ebn0_op_db;
				r.ber_theory = sdr::theoretical_ber_awgn(m, ref.ebn0_op_db);

				const auto quiet = run_chain<T, T, T>(p, m, p.quiet_ebn0_db);
				r.evm_quiet     = quiet.evm_rms;
				r.evm_arith     = excess_evm(quiet.evm_rms, ref.evm_quiet);

				const auto op = run_chain<T, T, T>(p, m, ref.ebn0_op_db);
				r.evm_op        = op.evm_rms;
				r.ber_op        = op.ber;
				r.bit_errors    = op.bit_errors;
				r.total_bits    = op.total_bits;
				r.clip_fraction = op.clip_fraction;
				r.loss_db = (ref.evm_op > 0.0 && op.evm_rms > 0.0)
					? 10.0 * std::log10((op.evm_rms * op.evm_rms) /
					                    (ref.evm_op * ref.evm_op))
					: 0.0;
				r.usable = (r.loss_db <= p.loss_budget_db);

				if (m == Modulation::qam16 &&
				    std::find(std::begin(kPlotted), std::end(kPlotted),
				              std::string(name)) != std::end(kPlotted)) {
					kept[name] = op.constellation;
					dumps.push_back({name, m, ref.ebn0_op_db, nullptr});
				}
				rows.push_back(std::move(r));
			}
			std::cout << "done\n";
		};

		// The count is reported after the fact rather than written into this
		// banner, so adding an add() call below cannot leave a stale number here.
		std::cout << "\n  Sweeping the number-system configurations across "
		          << std::size(kModulations)
		          << " modulations (2 links each: quiet, then at the operating point)\n";
		add(double{},  "double",         "IEEE",   64);
		add(float{},   "float",          "IEEE",   32);
		add(cf32{},    "cfloat<32,8>",   "cfloat", 32);
		add(cf16{},    "cfloat<16,5>",   "cfloat", 16);
		add(cf12{},    "cfloat<12,4>",   "cfloat", 12);
		add(cf8{},     "cfloat<8,4>",    "cfloat",  8);
		add(p32{},     "posit<32,2>",    "posit",  32);
		add(p16{},     "posit<16,2>",    "posit",  16);
		add(p12{},     "posit<12,2>",    "posit",  12);
		add(p8{},      "posit<8,2>",     "posit",   8);
		add(p8e0{},    "posit<8,0>",     "posit",   8);
		add(q32{},     "fixpnt<32,29>",  "fixpnt", 32);
		add(q16{},     "fixpnt<16,13>",  "fixpnt", 16);
		add(q12{},     "fixpnt<12,9>",   "fixpnt", 12);
		add(q8{},      "fixpnt<8,5>",    "fixpnt",  8);

		std::cout << "  " << num_configs << " configurations swept\n";

		for (auto& d : dumps) d.points = &kept.at(d.config);

		for (Modulation m : kModulations)
			print_sweep(m, rows, refs.at(m), p.loss_budget_db);

		print_modulation_ceiling(rows);

		// --- dynamic range -----------------------------------------------
		std::vector<BackoffRow> backoff;
		auto add_backoff = [&](auto tag, const char* name, int bits) {
			using T = decltype(tag);
			BackoffRow br;
			br.config    = name;
			br.bit_width = bits;
			for (double db : kBackoffDb) {
				const double scale = std::pow(10.0, db / 20.0);
				br.evm.push_back(run_chain<T, T, T>(p, Modulation::qam16,
				                                    p.quiet_ebn0_db, scale).evm_rms);
			}
			backoff.push_back(std::move(br));
		};
		std::cout << "\n  Measuring input backoff (16-QAM, "
		          << std::size(kBackoffDb) << " levels)..." << std::flush;
		add_backoff(double{}, "double",        64);
		add_backoff(cf16{},   "cfloat<16,5>",  16);
		add_backoff(p16{},    "posit<16,2>",   16);
		add_backoff(q16{},    "fixpnt<16,13>", 16);
		add_backoff(cf8{},    "cfloat<8,4>",    8);
		add_backoff(p8{},     "posit<8,2>",     8);
		add_backoff(q8{},     "fixpnt<8,5>",    8);
		std::cout << " done\n";
		print_backoff(backoff);
		print_findings(rows, backoff);

		// --- per-block attribution ---------------------------------------
		//
		// A different question from the sweep, and it needs a different model.
		// Narrowing one block at a time with the rest at double is the only
		// definition that attributes; the sweep above deliberately narrows
		// everything, which measures the cost but hides where it went.
		std::cout << "\n" << std::string(104, '=') << "\n";
		std::cout << "  Per-block attribution — 16-QAM, one block narrowed at a "
		             "time (analysis/sdr_precision.hpp)\n";
		std::cout << std::string(104, '=') << "\n";

		analysis::SdrLinkConfig lc;
		lc.modulation         = Modulation::qam16;
		lc.samples_per_symbol = p.samples_per_symbol;
		lc.rolloff            = p.rolloff;
		lc.rrc_span           = p.rrc_span;
		lc.ebn0_db            = p.quiet_ebn0_db;
		lc.num_symbols        = p.num_symbols;
		lc.seed               = p.seed;

		std::vector<analysis::SdrLinkResult> blocks;
		auto attribute = [&](auto tag, const char* name, int bits) {
			using T = decltype(tag);
			auto part = analysis::analyze_blocks<T>(lc, name, bits);
			// The reference row is identical for every type; keep one.
			blocks.insert(blocks.end(),
			              part.begin() + (blocks.empty() ? 0 : 1), part.end());
		};
		attribute(p16{}, "posit<16,2>", 16);
		attribute(cf16{}, "cfloat<16,5>", 16);
		attribute(p8{},  "posit<8,2>",   8);
		attribute(cf8{}, "cfloat<8,4>",  8);

		std::cout << std::left  << std::setw(18) << "Type"
		          << std::left  << std::setw(18) << "Block"
		          << std::right << std::setw(14) << "EVM"
		          << std::right << std::setw(18) << "contribution"
		          << "\n" << std::string(68, '-') << "\n";
		for (const auto& b : blocks)
			std::cout << std::left  << std::setw(18) << b.scalar_type
			          << std::left  << std::setw(18) << b.block
			          << std::right << std::setw(14) << fmt_sci(b.evm_rms)
			          << std::right << std::setw(18) << fmt_sci(b.evm_contribution)
			          << "\n";
		std::cout << std::string(68, '-') << "\n";
		std::cout << "  Contributions do not sum to whole_chain, and which way "
		             "they miss depends on\n  the precision — see the header. "
		             "whole_chain is measured, never summed.\n";
		std::cout << "  A contribution of 0 is a RESOLUTION LIMIT, not an exact "
		             "result: that run measured\n  no more EVM than the double "
		             "reference did. At 16 bits the arithmetic sits an order\n"
		             "  of magnitude below the truncation floor and the power "
		             "subtraction has little\n  left to resolve; the 8-bit rows, "
		             "which clear the floor, are where this\n  breakdown has "
		             "margin.\n";

		// --- EVM budgets --------------------------------------------------
		std::cout << "\n  EVM a modulation can absorb before a noiseless symbol "
		             "sits on a decision boundary:\n";
		for (Modulation m : kModulations)
			std::cout << "    " << std::left << std::setw(10) << sdr::to_string(m)
			          << std::fixed << std::setprecision(4)
			          << analysis::evm_budget(m) << "\n";
		std::cout << "  A ceiling, not a target; the loss budget above is the "
		             "operational criterion.\n";

		// --- output --------------------------------------------------------
		const std::string sweep_csv  = csv_dir + "/sdr_demo.csv";
		const std::string pareto_csv = csv_dir + "/sdr_demo_pareto.csv";
		const std::string const_csv  = csv_dir + "/sdr_demo_constellations.csv";
		const std::string blocks_csv = csv_dir + "/sdr_demo_blocks.csv";
		const std::string back_csv   = csv_dir + "/sdr_demo_backoff.csv";
		write_sweep_csv(sweep_csv, rows);
		write_pareto_csv(pareto_csv, rows);
		write_constellations_csv(const_csv, dumps);
		analysis::write_sdr_precision_csv(blocks_csv, blocks);
		{
			std::ofstream out(back_csv);
			if (!out) throw std::runtime_error("cannot open " + back_csv);
			out << "config,bit_width,modulation,backoff_db,evm_rms\n";
			out << std::setprecision(15);
			for (const auto& r : backoff)
				for (std::size_t i = 0; i < r.evm.size(); ++i)
					out << r.config << "," << r.bit_width << ",16-QAM,"
					    << kBackoffDb[i] << "," << r.evm[i] << "\n";
		}

		std::cout << "\n" << std::string(104, '=') << "\n";
		std::cout << "  " << rows.size() << " configuration/modulation pairs measured\n";
		std::cout << "  sweep:          " << sweep_csv  << "\n";
		std::cout << "  Pareto:         " << pareto_csv << "\n";
		std::cout << "  constellations: " << const_csv  << "\n";
		std::cout << "  block breakdown:" << blocks_csv << "\n";
		std::cout << "  input backoff:  " << back_csv   << "\n";
		std::cout << std::string(104, '=') << "\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "ERROR: " << ex.what() << "\n";
		return 1;
	}
}
