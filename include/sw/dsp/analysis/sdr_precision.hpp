#pragma once
// sdr_precision.hpp: end-to-end precision analysis for an SDR link.
//
// The per-module precision sweeps elsewhere in this library answer "what does
// this block cost?". This one answers the question a system designer actually
// has: given a link, WHICH BLOCK should get the wider arithmetic, and how
// narrow can the rest go before the constellation stops closing.
//
// THE CHAIN measured here is
//
//   bits -> constellation map -> RRC pulse shaping -> AWGN -> RRC matched
//   filter -> symbol sampling -> demap -> bits
//
// deliberately WITHOUT the timing and carrier loops. Those converge
// stochastically, so their residual would land on top of the arithmetic
// residual being measured and there would be no way to tell the two apart.
// Their precision behaviour is characterized in their own tests, where the
// loop is the subject rather than the noise floor. What remains here is the
// path whose error is purely arithmetic.
//
// PER-BLOCK ATTRIBUTION works by narrowing ONE block at a time and holding
// every other block at double. The difference between that and the all-double
// reference is the block's own contribution, which is the only definition
// that composes: measuring blocks jointly gives a number that cannot be
// attributed, and narrowing everything at once gives a total that hides
// which block spent it.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/filter/fir/fir_filter.hpp>
#include <sw/dsp/filter/fir/polyphase.hpp>
#include <sw/dsp/sdr/constellation.hpp>
#include <sw/dsp/sdr/metrics.hpp>
#include <sw/dsp/sdr/rrc.hpp>

namespace sw::dsp::analysis {

using sw::dsp::sdr::Modulation;

// Which block runs at the narrow type. Everything else stays at double, so a
// result is attributable to exactly one block.
enum class SdrBlock {
	none,             // all-double reference
	constellation,    // the symbol table
	tx_shaping,       // transmit RRC coefficients and filtering
	rx_matched,       // receive matched filter
	whole_chain       // every block narrow at once
};

inline const char* to_string(SdrBlock b) {
	switch (b) {
		case SdrBlock::none:          return "reference";
		case SdrBlock::constellation: return "constellation";
		case SdrBlock::tx_shaping:    return "tx_shaping";
		case SdrBlock::rx_matched:    return "rx_matched";
		case SdrBlock::whole_chain:   return "whole_chain";
	}
	return "unknown";
}

struct SdrLinkConfig {
	Modulation  modulation        = Modulation::qam16;
	std::size_t samples_per_symbol = 4;
	double      rolloff            = 0.35;
	std::size_t rrc_span           = 10;   // symbols
	double      ebn0_db            = 30.0;
	std::size_t num_symbols        = 2000;
	unsigned    seed               = 1;
};

struct SdrLinkResult {
	std::string block;
	std::string scalar_type;
	int         bit_width = 0;
	Modulation  modulation = Modulation::qam16;
	double      ebn0_db = 0.0;

	double      evm_rms  = 0.0;    // fraction
	double      evm_db   = 0.0;
	double      mer_db   = 0.0;
	std::size_t bit_errors = 0;
	std::size_t total_bits = 0;
	double      ber = 0.0;

	// EVM this block contributes ON ITS OWN, with the all-double reference
	// removed in POWER: sqrt(max(0, evm^2 - evm_ref^2)). Filled by
	// analyze_blocks(); zero for a standalone run_link().
	//
	// Subtracting in power rather than comparing raw EVM is what makes the
	// number attributable. Every configuration carries a common floor — the
	// pulse shaping's truncation ISI and whatever channel noise the config
	// asks for — and that floor is often LARGER than the arithmetic under
	// test. At Eb/N0 = 30 dB on a 10-symbol RRC span the floor measures
	// 1.07e-02 while posit16's whole contribution sits below 1e-04, so the
	// raw EVMs agree to four digits and say nothing.
	double evm_contribution = 0.0;

	// Received symbols at the sampling instants, for constellation plots.
	std::vector<std::complex<double>> constellation;
};

namespace detail {

// Project a double tap set into T and back, so a block's coefficients carry
// exactly the precision under test while the arithmetic around it stays wide.
template <typename T>
inline mtl::vec::dense_vector<double> project(const mtl::vec::dense_vector<double>& x) {
	mtl::vec::dense_vector<double> out(x.size());
	for (std::size_t i = 0; i < x.size(); ++i)
		out[i] = static_cast<double>(static_cast<T>(x[i]));
	return out;
}

template <typename T>
inline double project_scalar(double v) {
	return static_cast<double>(static_cast<T>(v));
}

} // namespace detail

// Run one link and measure it.
//
// `Narrow` is the type applied to `block`; every other block runs at double.
// With `SdrBlock::none` the result is the all-double reference and `Narrow`
// is unused.
//
// Pre:  samples_per_symbol >= 2, num_symbols > 4*rrc_span.
// Post: EVM, MER and BER over the settled portion of the stream, plus the
//       received constellation.
template <typename Narrow>
SdrLinkResult run_link(const SdrLinkConfig& cfg, SdrBlock block) {
	if (cfg.samples_per_symbol < 2)
		throw std::invalid_argument("run_link: samples_per_symbol must be >= 2");
	if (cfg.num_symbols <= 4 * cfg.rrc_span)
		throw std::invalid_argument(
			"run_link: num_symbols must exceed 4*rrc_span so the measurement "
			"window excludes the filter transients");

	const std::size_t sps = cfg.samples_per_symbol;
	const std::size_t ntaps = cfg.rrc_span * sps + 1;
	const bool narrow_all = (block == SdrBlock::whole_chain);

	// --- constellation ------------------------------------------------
	sdr::Constellation<double> ref_map(cfg.modulation);
	const std::size_t nb = ref_map.bits_per_symbol();
	const bool narrow_map = narrow_all || block == SdrBlock::constellation;

	// --- pulse shaping ------------------------------------------------
	auto taps = sdr::rrc_filter<double>(ntaps, sps, cfg.rolloff);
	auto tx_taps = (narrow_all || block == SdrBlock::tx_shaping)
		? detail::project<Narrow>(taps) : taps;
	auto rx_taps = (narrow_all || block == SdrBlock::rx_matched)
		? detail::project<Narrow>(taps) : taps;

	// --- symbols ------------------------------------------------------
	std::mt19937 rng(cfg.seed);
	std::uniform_int_distribution<std::size_t> pick(0, ref_map.order() - 1);
	std::vector<std::size_t> idx(cfg.num_symbols);
	mtl::vec::dense_vector<double> si(cfg.num_symbols), sq(cfg.num_symbols);
	std::vector<std::complex<double>> ideal(cfg.num_symbols);
	std::vector<std::uint8_t> tx_bits, tmp(nb);
	tx_bits.reserve(cfg.num_symbols * nb);

	for (std::size_t n = 0; n < cfg.num_symbols; ++n) {
		idx[n] = pick(rng);
		const auto p = ref_map.symbol(idx[n]);
		double re = p.real(), im = p.imag();
		if (narrow_map) {
			re = detail::project_scalar<Narrow>(re);
			im = detail::project_scalar<Narrow>(im);
		}
		si[n] = re; sq[n] = im;
		// The reference an EVM is measured against is the IDEAL point, not
		// the quantized one: quantizing the table IS part of the error the
		// link delivers, so folding it into the reference would hide exactly
		// what this function is asked to attribute.
		ideal[n] = std::complex<double>(p.real(), p.imag());
		ref_map.bits_of(idx[n], tmp);
		tx_bits.insert(tx_bits.end(), tmp.begin(), tmp.end());
	}

	// --- TX shaping, channel, RX matched filter -----------------------
	const double esn0_db = sdr::esn0_db_from_ebn0_db(cfg.modulation, cfg.ebn0_db);
	const double n0 = std::pow(10.0, -esn0_db / 10.0);
	// Per-dimension noise variance is N0/2 with NO samples-per-symbol factor.
	// Both RRC filters carry unit energy, so the transmit filter puts Es = 1
	// into each symbol and the matched filter passes the noise variance
	// through unchanged — the symbol-rate SNR is therefore the sample-rate
	// SNR. Dividing by sqrt(sps) here, as if the noise had to be spread with
	// the signal, quietly raised the true Es/N0 by 10*log10(sps): 6 dB at 4
	// samples per symbol, enough to make a 16-QAM link error-free at an
	// Eb/N0 where it should be visibly breaking.
	std::normal_distribution<double> gauss(0.0, std::sqrt(n0 / 2.0));

	auto shape_and_match = [&](const mtl::vec::dense_vector<double>& sym,
	                            bool add_noise) {
		PolyphaseInterpolator<double> up(tx_taps, sps);
		auto wave = up.process_block(std::span<const double>(sym.data(), sym.size()));
		if (add_noise)
			for (std::size_t i = 0; i < wave.size(); ++i) wave[i] += gauss(rng);
		FIRFilter<double> mf(rx_taps);
		mtl::vec::dense_vector<double> out(wave.size());
		for (std::size_t i = 0; i < wave.size(); ++i) out[i] = mf.process(wave[i]);
		return out;
	};
	auto ri = shape_and_match(si, true);
	auto rq = shape_and_match(sq, true);

	// --- sample, measure ----------------------------------------------
	const std::size_t delay = ntaps - 1;                 // TX + RX group delay
	const std::size_t skip  = 2 * cfg.rrc_span;          // settle
	SdrLinkResult r;
	r.block       = to_string(block);
	r.modulation  = cfg.modulation;
	r.ebn0_db     = cfg.ebn0_db;

	std::vector<std::complex<double>> got, want;
	std::vector<std::uint8_t> rx_bits, want_bits;
	for (std::size_t k = skip; k + skip < cfg.num_symbols; ++k) {
		const std::size_t s = delay + k * sps;
		if (s >= ri.size()) break;
		const std::complex<double> y(ri[s], rq[s]);
		got.push_back(y);
		want.push_back(ideal[k]);
		ref_map.demap_hard_bits(y, tmp);
		rx_bits.insert(rx_bits.end(), tmp.begin(), tmp.end());
		want_bits.insert(want_bits.end(),
		                 tx_bits.begin() + static_cast<long>(k * nb),
		                 tx_bits.begin() + static_cast<long>((k + 1) * nb));
	}
	if (got.empty())
		throw std::runtime_error("run_link: no symbols survived the transients");

	const auto e = sdr::evm<std::complex<double>>(want, got);
	r.evm_rms = e.rms;
	r.evm_db  = e.rms_db;
	r.mer_db  = -e.rms_db;
	const auto b = sdr::ber(want_bits, rx_bits);
	r.bit_errors = b.bit_errors;
	r.total_bits = b.total_bits;
	r.ber        = b.rate;
	r.constellation = got;
	return r;
}

// ============================================================================
// Per-block breakdown
// ============================================================================

// Run the all-double reference and each block narrowed in turn, filling in
// each one's attributable contribution.
//
// USE A QUIET CHANNEL. Attribution is only meaningful when the arithmetic is
// what dominates: any channel noise lands on every configuration equally and
// the power subtraction removes it, but it also drowns the signal being
// measured in sampling variance. An Eb/N0 of 60 dB or more makes the
// arithmetic the loudest thing in the link.
//
// THE CONTRIBUTIONS DO NOT ADD, AND WHICH WAY THEY MISS DEPENDS ON THE
// PRECISION. The returned `whole_chain` entry is therefore MEASURED, never
// summed. On 16-QAM at 80 dB:
//
//     type      power sum of parts    whole_chain measured
//     posit16          2.97e-04              3.41e-04   (more than the sum)
//     posit8           1.37e-02              9.47e-03   (less than the sum)
//
// At posit16 the per-block errors are small and largely independent, so
// narrowing everything compounds slightly beyond the sum; at posit8 they are
// large enough to partly cancel. A breakdown is an attribution, not a budget:
// it says where the error comes from, not what narrowing everything will
// cost. Predict that by measuring `whole_chain`.
template <typename Narrow>
std::vector<SdrLinkResult> analyze_blocks(const SdrLinkConfig& cfg,
                                           const std::string& type_name,
                                           int bit_width) {
	const SdrBlock order[] = {SdrBlock::none, SdrBlock::constellation,
	                          SdrBlock::tx_shaping, SdrBlock::rx_matched,
	                          SdrBlock::whole_chain};
	std::vector<SdrLinkResult> rows;
	rows.reserve(5);
	for (SdrBlock b : order) {
		auto r = run_link<Narrow>(cfg, b);
		r.scalar_type = (b == SdrBlock::none) ? "double" : type_name;
		r.bit_width   = (b == SdrBlock::none) ? 64 : bit_width;
		rows.push_back(std::move(r));
	}
	const double ref2 = rows.front().evm_rms * rows.front().evm_rms;
	for (auto& r : rows) {
		const double d = r.evm_rms * r.evm_rms - ref2;
		r.evm_contribution = (d > 0.0) ? std::sqrt(d) : 0.0;
	}
	return rows;
}

// ============================================================================
// Pareto frontier
// ============================================================================

struct SdrParetoPoint {
	Modulation  modulation;
	std::string scalar_type;
	int         bit_width = 0;
	double      evm_rms = 0.0;
	double      ber = 0.0;
	bool        usable = false;   // met the EVM budget for this modulation
};

// EVM a modulation can tolerate before its own decision distance is at risk.
//
// Half the distance from a constellation point to its nearest neighbour, in
// units of the RMS symbol amplitude — the error that would put a noiseless
// symbol exactly on a decision boundary. A practical link needs a good margin
// below this; it is a ceiling, not a target.
inline double evm_budget(Modulation m) {
	sdr::Constellation<double> c(m);
	double dmin = 1e30;
	for (std::size_t i = 0; i < c.order(); ++i)
		for (std::size_t j = i + 1; j < c.order(); ++j) {
			const auto a = c.symbol(i), b = c.symbol(j);
			dmin = std::min(dmin, std::hypot(a.real() - b.real(),
			                                  a.imag() - b.imag()));
		}
	return dmin / 2.0;   // reference power is 1 by construction
}

// ============================================================================
// CSV export
// ============================================================================

// One row per measurement, schema-compatible at the identifier columns with
// precision_sweep.csv and acquisition_demo.csv so the same Python tooling
// reads all three.
inline void write_sdr_precision_csv(const std::string& path,
                                    const std::vector<SdrLinkResult>& rows) {
	std::ofstream out(path);
	if (!out) throw std::runtime_error(
		"write_sdr_precision_csv: cannot open " + path);
	out << "pipeline,block,scalar_type,bit_width,modulation,ebn0_db,"
	       "evm_rms,evm_db,mer_db,evm_contribution,bit_errors,total_bits,ber\n";
	for (const auto& r : rows) {
		out << "sdr_link," << r.block << "," << r.scalar_type << ","
		    << r.bit_width << "," << sdr::to_string(r.modulation) << ","
		    << r.ebn0_db << "," << r.evm_rms << "," << r.evm_db << ","
		    << r.mer_db << "," << r.evm_contribution << "," << r.bit_errors
		    << "," << r.total_bits << "," << r.ber << "\n";
	}
}

// Received constellation points, one row per symbol, for plotting.
inline void write_constellation_csv(const std::string& path,
                                    const SdrLinkResult& r) {
	std::ofstream out(path);
	if (!out) throw std::runtime_error(
		"write_constellation_csv: cannot open " + path);
	out << "block,scalar_type,modulation,i,q\n";
	for (const auto& s : r.constellation)
		out << r.block << "," << r.scalar_type << ","
		    << sdr::to_string(r.modulation) << ","
		    << s.real() << "," << s.imag() << "\n";
}

} // namespace sw::dsp::analysis
