#pragma once
// calibration.hpp: Front-end calibration / equalization framework for
// instrument-style data acquisition.
//
// Real instruments — both oscilloscopes and spectrum analyzers — have analog
// input paths (anti-alias filters, cable losses, attenuator networks) that
// imprint a non-flat magnitude/phase response onto the captured signal.
// This header provides:
//
//   CalibrationProfile<CoeffScalar>
//     Stores a measured frequency-response correction as tabulated
//     (frequency, gain_dB, phase_rad) triples. Linear interpolation between
//     tabulated points; clamps to endpoints outside the calibrated band.
//     Constructible from explicit vectors or from a CSV file.
//
//   EqualizerFilter<CoeffScalar, StateScalar, SampleScalar>
//     A streaming FIR whose magnitude/phase response cancels a
//     CalibrationProfile. Constructor takes the profile, desired filter
//     length, and sample rate, runs frequency-sampling design with a
//     Hamming window, and clamps the inverse to a configurable max gain
//     to avoid amplifying noise where the profile has deep nulls.
//
// Design math is performed in double and cast to CoeffScalar at the end —
// the same pattern used in the SDR demo (`acquisition_demo.cpp`) and for
// the same reason: filter-design variance from running Remez/Kaiser at
// low precision would conflate with the streaming-arithmetic precision the
// rest of the three-scalar parameterization is meant to characterize.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <fstream>
#include <numbers>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/filter/fir/fir_filter.hpp>

namespace sw::dsp::instrument {

// =============================================================================
// CalibrationProfile
//
// Tabulated frequency-response correction. Stores three parallel arrays:
// frequencies (Hz, monotonically increasing), gain_dB at each frequency,
// and phase_rad at each frequency. gain_dB() and phase_rad() return the
// linearly-interpolated correction at any query frequency; queries below
// the lowest tabulated frequency clamp to the first entry, queries above
// the highest clamp to the last.
//
// Not templated on a numeric type — calibration measurements come from
// external sweep instruments and are inherently double-precision-accurate.
// The streaming-arithmetic precision (CoeffScalar / StateScalar / SampleScalar)
// is decided at the EqualizerFilter level.
// =============================================================================
class CalibrationProfile {
public:
	// Construct from explicit (frequency, gain_dB, phase_rad) vectors.
	// All three must have the same length and `frequencies` must be
	// strictly monotonically increasing.
	CalibrationProfile(std::vector<double> frequencies,
	                   std::vector<double> gain_dB,
	                   std::vector<double> phase_rad)
		: frequencies_(std::move(frequencies)),
		  gain_dB_(std::move(gain_dB)),
		  phase_rad_(std::move(phase_rad)) {
		if (frequencies_.size() != gain_dB_.size() ||
		    frequencies_.size() != phase_rad_.size())
			throw std::invalid_argument(
				"CalibrationProfile: frequencies, gain_dB, phase_rad must "
				"have the same length");
		if (frequencies_.size() < 2)
			throw std::invalid_argument(
				"CalibrationProfile: need at least 2 tabulated points");
		for (std::size_t i = 1; i < frequencies_.size(); ++i) {
			if (!(frequencies_[i] > frequencies_[i - 1]))
				throw std::invalid_argument(
					"CalibrationProfile: frequencies must be strictly "
					"monotonically increasing");
		}
	}

	// Construct from a CSV file with one row per frequency:
	//   freq_hz,gain_dB,phase_rad
	// The first line may optionally be a header (any non-numeric first
	// token); subsequent lines are data. Lines starting with '#' are
	// treated as comments and skipped.
	static CalibrationProfile from_csv(const std::string& path) {
		std::ifstream in(path);
		if (!in)
			throw std::invalid_argument(
				"CalibrationProfile::from_csv: cannot open '" + path + "'");

		std::vector<double> f, g, p;
		std::string line;
		bool first_line_seen = false;
		while (std::getline(in, line)) {
			if (line.empty() || line.front() == '#') continue;

			// Try to parse as three comma-separated doubles. If the first
			// row fails to parse, treat it as a header and skip.
			double  freq, gain, phase;
			char    c1, c2;
			std::istringstream iss(line);
			iss >> freq >> c1 >> gain >> c2 >> phase;
			const bool ok = !iss.fail() && c1 == ',' && c2 == ',';
			if (!ok) {
				if (!first_line_seen) { first_line_seen = true; continue; }
				throw std::invalid_argument(
					"CalibrationProfile::from_csv: bad row in '" + path +
					"': '" + line + "'");
			}
			first_line_seen = true;
			f.push_back(freq);
			g.push_back(gain);
			p.push_back(phase);
		}
		return CalibrationProfile(std::move(f), std::move(g), std::move(p));
	}

	// Linearly-interpolated gain (dB) at any query frequency.
	double gain_dB(double freq_hz) const {
		return interp_(freq_hz, gain_dB_);
	}

	// Linearly-interpolated phase (rad) at any query frequency.
	double phase_rad(double freq_hz) const {
		return interp_(freq_hz, phase_rad_);
	}

	std::size_t size()         const { return frequencies_.size(); }
	double      freq_min()     const { return frequencies_.front(); }
	double      freq_max()     const { return frequencies_.back(); }

	std::span<const double> frequencies() const {
		return {frequencies_.data(), frequencies_.size()};
	}

private:
	double interp_(double f, const std::vector<double>& y) const {
		if (f <= frequencies_.front()) return y.front();
		if (f >= frequencies_.back())  return y.back();
		// Binary search for the interval [i-1, i] containing f.
		auto it = std::upper_bound(frequencies_.begin(), frequencies_.end(), f);
		const std::size_t i = static_cast<std::size_t>(it - frequencies_.begin());
		const double f0 = frequencies_[i - 1];
		const double f1 = frequencies_[i];
		const double y0 = y[i - 1];
		const double y1 = y[i];
		const double t  = (f - f0) / (f1 - f0);
		return y0 + t * (y1 - y0);
	}

	std::vector<double> frequencies_;
	std::vector<double> gain_dB_;
	std::vector<double> phase_rad_;
};

// =============================================================================
// EqualizerFilter
//
// Frequency-sampling design of an FIR that inverts a CalibrationProfile.
//
// Algorithm:
//   1. Sample the desired *inverse* response on a uniform frequency grid
//      ω_k = 2πk/num_taps for k = 0..num_taps-1, mapped to physical
//      frequency f_k = k * sample_rate / num_taps.
//   2. For each grid point, read the profile's gain/phase and form the
//      inverse: gain_inv = -gain_dB, phase_inv = -phase_rad. Clamp
//      gain_inv to max_gain_dB to avoid blowing up at deep nulls.
//   3. Enforce conjugate symmetry: H_d[num_taps - k] = conj(H_d[k])
//      so the inverse-DFT result is real.
//   4. Inverse DFT: h[n] = (1/num_taps) Σ_k H_d[k] exp(j 2π k n / num_taps).
//      Direct sum (O(N²)) is fine for typical num_taps in [16, 256].
//   5. Multiply by a Hamming window and shift by num_taps/2 to center
//      the linear-phase impulse response.
//   6. Cast to CoeffScalar and hand to FIRFilter.
//
// All design math is in double; only the final taps are cast.
// =============================================================================
template <DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class EqualizerFilter {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;

	// profile        : the calibration to invert
	// num_taps       : FIR length (longer = better in-band fidelity but
	//                  more arithmetic and longer group delay)
	// sample_rate_hz : the rate at which this filter will be run
	// max_gain_dB    : clamp on the inverse magnitude. Default +60 dB
	//                  (1000×) is generous for typical analog front-end
	//                  compensation; anywhere the profile has a null
	//                  deeper than -max_gain_dB, the equalizer will be
	//                  capped instead of amplifying noise infinitely.
	EqualizerFilter(const CalibrationProfile& profile,
	                std::size_t num_taps,
	                double sample_rate_hz,
	                double max_gain_dB = 60.0)
		: fir_(design_taps(profile, num_taps, sample_rate_hz, max_gain_dB)) {}

	// Streaming process — single sample
	SampleScalar process(SampleScalar in) { return fir_.process(in); }

	// Block process (in-place)
	void process_block(std::span<SampleScalar> samples) {
		fir_.process_block(samples);
	}

	// Block process — separate input/output spans
	void process_block(std::span<const SampleScalar> input,
	                   std::span<SampleScalar> output) {
		fir_.process_block(input, output);
	}

	std::size_t num_taps() const { return fir_.order() + 1; }

private:
	static mtl::vec::dense_vector<CoeffScalar>
	design_taps(const CalibrationProfile& profile,
	            std::size_t num_taps,
	            double sample_rate_hz,
	            double max_gain_dB) {
		if (num_taps < 3)
			throw std::invalid_argument(
				"EqualizerFilter: num_taps must be >= 3");
		if (!(sample_rate_hz > 0.0))
			throw std::invalid_argument(
				"EqualizerFilter: sample_rate_hz must be > 0");
		if (!(max_gain_dB > 0.0))
			throw std::invalid_argument(
				"EqualizerFilter: max_gain_dB must be > 0");

		const std::size_t N    = num_taps;
		const double      pi   = std::numbers::pi_v<double>;
		const double      max_lin = std::pow(10.0, max_gain_dB / 20.0);

		// 1+2: Sample the inverse response with conjugate symmetry.
		// We sample only the lower half of the spectrum (k = 0..N/2) and
		// mirror by conjugate symmetry to the upper half. This guarantees
		// a real-valued inverse-DFT result.
		std::vector<std::complex<double>> H_d(N);
		for (std::size_t k = 0; k <= N / 2; ++k) {
			const double f       = static_cast<double>(k) * sample_rate_hz / N;
			const double gain_dB = profile.gain_dB(f);
			const double phase   = profile.phase_rad(f);

			// Inverse magnitude, clamped
			double inv_mag = std::pow(10.0, -gain_dB / 20.0);
			if (inv_mag > max_lin) inv_mag = max_lin;

			// Inverse phase
			const double inv_phase = -phase;

			H_d[k] = std::complex<double>(
				inv_mag * std::cos(inv_phase),
				inv_mag * std::sin(inv_phase));
		}
		// Conjugate symmetry for the upper half:
		//   H_d[N - k] = conj(H_d[k]) for k = 1..floor((N-1)/2)
		for (std::size_t k = 1; k < (N + 1) / 2; ++k) {
			H_d[N - k] = std::conj(H_d[k]);
		}
		// Nyquist bin (k = N/2) for even N must be real; the loop above
		// already wrote a complex value but its imag should be zero (or
		// negligible) since cos/sin of inv_phase + conjugate. For safety
		// when N is even, force the imaginary part to zero:
		if (N % 2 == 0) {
			H_d[N / 2] = std::complex<double>(H_d[N / 2].real(), 0.0);
		}

		// 4: Inverse DFT (direct sum). h[n] = (1/N) Σ H_d[k] exp(j 2πkn/N).
		// We then linear-phase-shift by N/2 samples (delay = (N-1)/2 for
		// odd N or N/2 for even N) to center the impulse response.
		std::vector<double> h_centered(N);
		const double         delay = static_cast<double>(N - 1) / 2.0;
		for (std::size_t n = 0; n < N; ++n) {
			std::complex<double> acc{0.0, 0.0};
			for (std::size_t k = 0; k < N; ++k) {
				const double angle = 2.0 * pi * static_cast<double>(k) *
				                     (static_cast<double>(n) - delay) / N;
				acc += H_d[k] * std::complex<double>(std::cos(angle),
				                                     std::sin(angle));
			}
			h_centered[n] = acc.real() / static_cast<double>(N);
		}

		// 5: Hamming window
		mtl::vec::dense_vector<CoeffScalar> taps(N);
		for (std::size_t n = 0; n < N; ++n) {
			const double w = 0.54 - 0.46 * std::cos(
				2.0 * pi * static_cast<double>(n) / static_cast<double>(N - 1));
			taps[n] = static_cast<CoeffScalar>(h_centered[n] * w);
		}
		return taps;
	}

	FIRFilter<CoeffScalar, StateScalar, SampleScalar> fir_;
};

} // namespace sw::dsp::instrument
