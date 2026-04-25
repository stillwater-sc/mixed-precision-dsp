#pragma once
// acquisition_precision.hpp: precision-analysis primitives for high-rate
// data acquisition pipelines (NCO, CIC, half-band, polyphase FIR, DDC,
// DecimationChain).
//
// This header provides the primitives needed to characterize how the
// arithmetic precision of each stage affects output quality:
//
//   snr_db / enob_from_snr_db   - scalar quality metrics
//   measure_nco_sfdr_db          - spurious-free dynamic range of an NCO
//   check_cic_bit_growth         - verify observed vs theoretical bit growth
//   AcquisitionPrecisionRow      - schema-compatible Pareto-row record
//   write_acquisition_csv        - CSV writer matching precision_sweep schema
//
// Per-stage noise budgeting is a workflow built on top of these primitives:
// the user constructs two Chain instances that differ in exactly one
// stage's scalar type, runs the same input through both, and uses
// snr_db on the outputs to isolate that stage's contribution.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/spectral/fft.hpp>

namespace sw::dsp::analysis {

// ---------------------------------------------------------------------------
// Scalar quality metrics
// ---------------------------------------------------------------------------

// Effective number of bits from SNR in dB. Standard formula:
//   ENOB = (SNR_dB - 1.76) / 6.02
// Source: Walt Kester / Analog Devices "Define ENOB" tutorial. Assumes
// a sinusoidal full-scale input with quantization-noise dominated error.
inline double enob_from_snr_db(double snr_db) {
	return (snr_db - 1.76) / 6.02;
}

// SNR in dB of a test signal against a reference. Both spans must have
// the same length. Returns +300 (effectively infinite) when noise power
// is below the double-precision underflow boundary, signalling a
// bit-identical match.
template <class RefScalar, class TestScalar>
	requires ConvertibleToDouble<RefScalar> && ConvertibleToDouble<TestScalar>
double snr_db(std::span<const RefScalar> reference,
              std::span<const TestScalar> test) {
	if (reference.size() != test.size())
		throw std::invalid_argument(
			"snr_db: reference and test spans must have the same length");
	if (reference.empty()) return 0.0;

	double signal_power = 0.0;
	double noise_power = 0.0;
	for (std::size_t i = 0; i < reference.size(); ++i) {
		const double r = static_cast<double>(reference[i]);
		const double t = static_cast<double>(test[i]);
		signal_power += r * r;
		const double e = r - t;
		noise_power += e * e;
	}
	if (signal_power <= 0.0) return 0.0;
	if (noise_power < 1e-300) return 300.0;
	return 10.0 * std::log10(signal_power / noise_power);
}

// Convenience overload for mtl::vec::dense_vector.
template <class RefScalar, class TestScalar>
	requires ConvertibleToDouble<RefScalar> && ConvertibleToDouble<TestScalar>
double snr_db(const mtl::vec::dense_vector<RefScalar>& reference,
              const mtl::vec::dense_vector<TestScalar>& test) {
	return snr_db(std::span<const RefScalar>(reference.data(), reference.size()),
	              std::span<const TestScalar>(test.data(), test.size()));
}

// ---------------------------------------------------------------------------
// NCO spurious-free dynamic range
// ---------------------------------------------------------------------------

// Measure SFDR of an NCO by:
//   1. Generating fft_size complex samples
//   2. Forward FFT (zero-pad to next power of 2 if needed)
//   3. Locating the peak bin, then the largest spur outside a small
//      guard band around the peak
//   4. Returning 20 * log10(peak / spur) in dB
//
// The NCO is mutated (its phase advances). Caller may want to reset()
// before/after to keep the run reproducible.
template <class NCO>
double measure_nco_sfdr_db(NCO& nco, std::size_t fft_size,
                           std::size_t guard_bins = 2) {
	if (fft_size == 0)
		throw std::invalid_argument("measure_nco_sfdr_db: fft_size must be > 0");

	// Round fft_size up to next power of 2 for the FFT. Guard against the
	// shift wrapping N to 0 for pathologically large fft_size — that would
	// turn the loop infinite (since 0 < fft_size stays true and 0 << 1 = 0).
	std::size_t N = 1;
	constexpr std::size_t shift_limit = std::numeric_limits<std::size_t>::max() >> 1;
	while (N < fft_size) {
		if (N > shift_limit)
			throw std::overflow_error(
				"measure_nco_sfdr_db: fft_size too large to round up to a "
				"power of 2 within size_t");
		N <<= 1;
	}

	using complex_t = typename NCO::complex_t;
	mtl::vec::dense_vector<std::complex<double>> data(N);
	for (std::size_t n = 0; n < fft_size; ++n) {
		complex_t z = nco.generate_sample();
		data[n] = std::complex<double>(static_cast<double>(z.real()),
		                                static_cast<double>(z.imag()));
	}
	for (std::size_t n = fft_size; n < N; ++n) data[n] = std::complex<double>(0, 0);

	sw::dsp::spectral::fft_forward<double>(data);

	// Find peak (excluding bin 0 = DC).
	double peak_mag = 0.0;
	std::size_t peak_bin = 0;
	for (std::size_t k = 1; k < N; ++k) {
		const double m = std::abs(data[k]);
		if (m > peak_mag) { peak_mag = m; peak_bin = k; }
	}
	if (peak_mag <= 0.0)
		throw std::runtime_error("measure_nco_sfdr_db: signal has zero magnitude");

	// Max spur outside guard around peak. Use circular bin distance so
	// peaks near bin 1 or N-1 don't treat their wrap-adjacent neighbours
	// as far-away spurs.
	double spur_mag = 0.0;
	for (std::size_t k = 1; k < N; ++k) {
		const std::size_t diff = (k > peak_bin) ? (k - peak_bin) : (peak_bin - k);
		const std::size_t dist = std::min(diff, N - diff);
		if (dist <= guard_bins) continue;
		const double m = std::abs(data[k]);
		if (m > spur_mag) spur_mag = m;
	}
	if (spur_mag <= 0.0) return 300.0;
	return 20.0 * std::log10(peak_mag / spur_mag);
}

// ---------------------------------------------------------------------------
// CIC bit-growth verification
// ---------------------------------------------------------------------------

struct CICBitGrowthReport {
	int    theoretical_bits;     // M * ceil(log2(R*D)) — Hogenauer's formula
	int    observed_bits;        // ceil(log2(max |output|)) for the test input
	double max_abs_output;       // raw measured peak
	double headroom_bits;        // theoretical - observed, both as floats
	bool   within_theory;        // true when observed <= theoretical
};

// Process the input through a CIC, recording the absolute peak of its
// output samples, then compare against the theoretical worst-case
// growth M*ceil(log2(R*D)). For a full-scale all-ones input the
// theoretical and observed values converge.
//
// Note: this measures the OUTPUT peak (which is what the StateScalar
// must accommodate at the comb-stage end). Internal integrator
// magnitudes can be larger transiently but wrap around modulo their
// width — the standard CIC analysis assumes 2's-complement wrap is
// preserved, which `int`/`fixpnt` accumulators naturally provide.
template <class CIC, class Sample>
	requires ConvertibleToDouble<Sample> &&
	         requires(const CIC& c) { static_cast<double>(c.output()); }
CICBitGrowthReport check_cic_bit_growth(CIC& cic,
                                         std::span<const Sample> input) {
	double max_abs = 0.0;
	for (std::size_t n = 0; n < input.size(); ++n) {
		if (cic.push(input[n])) {
			const double y = std::abs(static_cast<double>(cic.output()));
			if (y > max_abs) max_abs = y;
		}
	}

	const int R = cic.decimation_ratio();
	const int M = cic.num_stages();
	const int D = cic.differential_delay();
	const int theoretical = M * static_cast<int>(std::ceil(std::log2(
		static_cast<double>(R) * static_cast<double>(D))));
	const double observed_f = (max_abs > 0.0) ? std::log2(max_abs) : 0.0;
	const int observed = static_cast<int>(std::ceil(observed_f));
	return CICBitGrowthReport{
		theoretical,
		observed,
		max_abs,
		static_cast<double>(theoretical) - observed_f,
		observed <= theoretical
	};
}

// ---------------------------------------------------------------------------
// Pareto-row record + CSV writer (schema-compatible with precision_sweep)
// ---------------------------------------------------------------------------

struct AcquisitionPrecisionRow {
	std::string pipeline;       // "ddc" | "decim_chain" | "nco" | etc.
	std::string config_name;    // human-readable configuration label
	std::string coeff_type;     // string repr of CoeffScalar
	std::string state_type;     // string repr of StateScalar
	std::string sample_type;    // string repr of SampleScalar
	int         total_bits = 0; // sum of bit-widths across the three scalars
	double      output_snr_db = 0.0;
	double      output_enob = 0.0;
	double      nco_sfdr_db = -1.0;          // -1 = N/A
	double      cic_overflow_margin_bits = -1.0;  // -1 = N/A
};

namespace detail {
inline std::string csv_quote(const std::string& s) {
	if (s.find_first_of(",\"\n") == std::string::npos) return s;
	std::string out = "\"";
	for (char c : s) {
		if (c == '"') out += "\"\"";
		else out += c;
	}
	out += '"';
	return out;
}
} // namespace detail

// Write a header + one line per row. The first five string columns and
// the SNR column align with applications/precision_sweep/precision_sweep.csv
// so the existing Python visualization can read either file's common columns.
inline void write_acquisition_csv(const std::string& path,
                                  const std::vector<AcquisitionPrecisionRow>& rows) {
	std::ofstream out(path);
	if (!out)
		throw std::runtime_error("write_acquisition_csv: cannot open " + path);
	out << "pipeline,config_name,coeff_type,state_type,sample_type,"
	    << "total_bits,output_snr_db,output_enob,nco_sfdr_db,"
	    << "cic_overflow_margin_bits\n";
	for (const auto& r : rows) {
		out << detail::csv_quote(r.pipeline) << ','
		    << detail::csv_quote(r.config_name) << ','
		    << detail::csv_quote(r.coeff_type) << ','
		    << detail::csv_quote(r.state_type) << ','
		    << detail::csv_quote(r.sample_type) << ','
		    << r.total_bits << ','
		    << r.output_snr_db << ','
		    << r.output_enob << ','
		    << r.nco_sfdr_db << ','
		    << r.cic_overflow_margin_bits << '\n';
	}
}

} // namespace sw::dsp::analysis
