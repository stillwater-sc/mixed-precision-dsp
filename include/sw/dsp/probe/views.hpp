#pragma once
// views.hpp: Domain-view helpers for probe-captured streams.
//
// Free functions that turn a SignalProbe's captured samples into
// structured, CSV-dumpable data in the domain the analyst wants:
//   * time_view          - raw samples + sample rate
//   * magnitude_spectrum - windowed FFT magnitudes in dB
//   * phase_spectrum     - windowed FFT phases in rad (unwrapped optional)
//   * iq_constellation   - I/Q pairs for complex probes
//
// Each returned struct exposes dump_csv(path) so the mp-dsp-python
// peer repo can pick up the data for rendering.
//
// FFT-based views zero-pad to the next power of two (library FFT is
// power-of-two only). For real probes the magnitude/phase output is
// the one-sided spectrum [0, fs/2]; for complex probes it is the
// full two-sided spectrum in fftshift order [-fs/2, +fs/2).
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/probe/signal_probe.hpp>
#include <sw/dsp/spectral/fft.hpp>
#include <sw/dsp/windows/blackman.hpp>
#include <sw/dsp/windows/hamming.hpp>
#include <sw/dsp/windows/hanning.hpp>
#include <sw/dsp/windows/kaiser.hpp>
#include <sw/dsp/windows/rectangular.hpp>

namespace sw::dsp::probe {

// ============================================================================
// Common: window choice + FFT sizing helpers
// ============================================================================

enum class WindowType {
	Rectangular,  // no window (best frequency resolution, worst leakage)
	Hamming,      // ~-42 dB peak sidelobe (default, general purpose)
	Hann,         // ~-32 dB peak sidelobe
	Blackman,     // ~-58 dB peak sidelobe
	Kaiser,       // parameterized by beta (best flexibility)
};

namespace detail {

inline std::vector<double> build_window(std::size_t n, WindowType w,
                                          double kaiser_beta = 8.6) {
	auto to_vec = [&](const auto& v) {
		std::vector<double> out(v.size());
		for (std::size_t i = 0; i < v.size(); ++i) out[i] = v[i];
		return out;
	};
	switch (w) {
		case WindowType::Rectangular: return to_vec(rectangular_window<double>(n));
		case WindowType::Hamming:     return to_vec(hamming_window<double>(n));
		case WindowType::Hann:        return to_vec(hanning_window<double>(n));
		case WindowType::Blackman:    return to_vec(blackman_window<double>(n));
		case WindowType::Kaiser:      return to_vec(kaiser_window<double>(n, kaiser_beta));
	}
	// Unreachable; silence compiler warning.
	return to_vec(hamming_window<double>(n));
}

inline std::size_t next_pow2(std::size_t n) {
	std::size_t p = 1;
	while (p < n) p <<= 1;
	return p;
}

// Unwrap a phase array in place: adjust each element to be within
// pi of the previous one by adding/subtracting multiples of 2*pi.
inline void unwrap_phase(std::vector<double>& phi) {
	if (phi.size() < 2) return;
	const double two_pi = 2.0 * std::numbers::pi_v<double>;
	for (std::size_t i = 1; i < phi.size(); ++i) {
		double d = phi[i] - phi[i - 1];
		while (d >  std::numbers::pi_v<double>) { phi[i] -= two_pi; d -= two_pi; }
		while (d < -std::numbers::pi_v<double>) { phi[i] += two_pi; d += two_pi; }
	}
}

} // namespace detail

// ============================================================================
// Time view - raw samples, cast to double for the CSV
// ============================================================================

template <class T>
struct TimeView {
	std::vector<double> samples;
	double              sample_rate_hz;
	std::string         label;

	void dump_csv(const std::string& path) const {
		std::ofstream out(path);
		if (!out) throw std::runtime_error("TimeView::dump_csv: cannot open " + path);
		out << "sample_index,time_s,sample_value\n";
		out << std::setprecision(17);
		for (std::size_t i = 0; i < samples.size(); ++i) {
			out << i << "," << (static_cast<double>(i) / sample_rate_hz)
			    << "," << samples[i] << "\n";
		}
	}
};

// Real-valued probe view.
template <DspScalar T>
TimeView<T> time_view(const SignalProbe<T>& probe) {
	TimeView<T> v;
	v.sample_rate_hz = probe.sample_rate();
	v.label          = probe.label();
	auto s = probe.samples();
	v.samples.resize(s.size());
	for (std::size_t i = 0; i < s.size(); ++i)
		v.samples[i] = static_cast<double>(s[i]);
	return v;
}

// ============================================================================
// Magnitude spectrum
//
// For real T: one-sided spectrum, N/2+1 bins from 0 to fs/2.
// For std::complex<T> (via SignalProbe<std::complex<T>>): two-sided
// spectrum in fftshift order, N bins from -fs/2 to +fs/2.
// ============================================================================

template <class T>
struct MagnitudeSpectrum {
	std::vector<double> freqs_hz;
	std::vector<double> magnitudes_dB;
	std::string         label;
	std::size_t         fft_size = 0;

	void dump_csv(const std::string& path) const {
		std::ofstream out(path);
		if (!out) throw std::runtime_error(
			"MagnitudeSpectrum::dump_csv: cannot open " + path);
		out << "freq_hz,magnitude_dB\n";
		out << std::setprecision(17);
		for (std::size_t i = 0; i < freqs_hz.size(); ++i) {
			out << freqs_hz[i] << "," << magnitudes_dB[i] << "\n";
		}
	}
};

// Real probe -> one-sided magnitude spectrum.
template <DspScalar T>
MagnitudeSpectrum<T> magnitude_spectrum(const SignalProbe<T>& probe,
                                          WindowType window = WindowType::Hamming,
                                          double kaiser_beta = 8.6) {
	MagnitudeSpectrum<T> spec;
	spec.label = probe.label();
	auto s = probe.samples();
	if (s.empty()) return spec;

	const std::size_t n = s.size();
	const std::size_t fft_n = detail::next_pow2(n);
	spec.fft_size = fft_n;
	const auto win = detail::build_window(n, window, kaiser_beta);

	mtl::vec::dense_vector<std::complex<double>> buf(fft_n,
	                                                    std::complex<double>{});
	for (std::size_t i = 0; i < n; ++i) {
		buf[i] = std::complex<double>(static_cast<double>(s[i]) * win[i], 0.0);
	}
	sw::dsp::spectral::fft_forward<double>(buf);

	// Normalization: coherent gain divides out so a full-scale tone
	// at exact bin frequency reads magnitude 1.0 (0 dBFS-ish).
	double win_sum = 0.0;
	for (double w : win) win_sum += w;
	if (win_sum <= 0.0) win_sum = 1.0;

	const std::size_t half = fft_n / 2 + 1;
	spec.freqs_hz.resize(half);
	spec.magnitudes_dB.resize(half);
	const double fs = probe.sample_rate();
	for (std::size_t k = 0; k < half; ++k) {
		spec.freqs_hz[k] = static_cast<double>(k) * fs
		                    / static_cast<double>(fft_n);
		double m = std::abs(buf[k]) / win_sum;
		spec.magnitudes_dB[k] = (m > 1e-300) ? 20.0 * std::log10(m) : -300.0;
	}
	return spec;
}

// Complex probe -> two-sided magnitude spectrum in fftshift order.
template <DspScalar T>
MagnitudeSpectrum<T> magnitude_spectrum(
        const SignalProbe<std::complex<T>>& probe,
        WindowType window = WindowType::Hamming,
        double kaiser_beta = 8.6) {
	MagnitudeSpectrum<T> spec;
	spec.label = probe.label();
	auto s = probe.samples();
	if (s.empty()) return spec;

	const std::size_t n = s.size();
	const std::size_t fft_n = detail::next_pow2(n);
	spec.fft_size = fft_n;
	const auto win = detail::build_window(n, window, kaiser_beta);

	mtl::vec::dense_vector<std::complex<double>> buf(fft_n,
	                                                    std::complex<double>{});
	for (std::size_t i = 0; i < n; ++i) {
		const auto& z = s[i];
		buf[i] = std::complex<double>(static_cast<double>(z.real()) * win[i],
		                                static_cast<double>(z.imag()) * win[i]);
	}
	sw::dsp::spectral::fft_forward<double>(buf);

	double win_sum = 0.0;
	for (double w : win) win_sum += w;
	if (win_sum <= 0.0) win_sum = 1.0;

	spec.freqs_hz.resize(fft_n);
	spec.magnitudes_dB.resize(fft_n);
	const double fs = probe.sample_rate();
	// fftshift: bins [N/2..N-1] represent negative freqs -fs/2..0,
	// bins [0..N/2-1] represent positive freqs 0..fs/2-.
	for (std::size_t k = 0; k < fft_n; ++k) {
		std::size_t out_k = (k + fft_n / 2) % fft_n;
		double f = (static_cast<double>(k) - static_cast<double>(fft_n) / 2.0)
		            * fs / static_cast<double>(fft_n);
		spec.freqs_hz[k] = f;
		double m = std::abs(buf[out_k]) / win_sum;
		spec.magnitudes_dB[k] = (m > 1e-300) ? 20.0 * std::log10(m) : -300.0;
	}
	return spec;
}

// ============================================================================
// Phase spectrum
// ============================================================================

template <class T>
struct PhaseSpectrum {
	std::vector<double> freqs_hz;
	std::vector<double> phases_rad;
	std::string         label;
	bool                unwrapped = false;

	void dump_csv(const std::string& path) const {
		std::ofstream out(path);
		if (!out) throw std::runtime_error(
			"PhaseSpectrum::dump_csv: cannot open " + path);
		out << "freq_hz,phase_rad\n";
		out << std::setprecision(17);
		for (std::size_t i = 0; i < freqs_hz.size(); ++i) {
			out << freqs_hz[i] << "," << phases_rad[i] << "\n";
		}
	}
};

template <DspScalar T>
PhaseSpectrum<T> phase_spectrum(const SignalProbe<T>& probe,
                                  bool unwrap = true,
                                  WindowType window = WindowType::Hamming,
                                  double kaiser_beta = 8.6) {
	PhaseSpectrum<T> spec;
	spec.label = probe.label();
	spec.unwrapped = unwrap;
	auto s = probe.samples();
	if (s.empty()) return spec;

	const std::size_t n = s.size();
	const std::size_t fft_n = detail::next_pow2(n);
	const auto win = detail::build_window(n, window, kaiser_beta);

	mtl::vec::dense_vector<std::complex<double>> buf(fft_n,
	                                                    std::complex<double>{});
	for (std::size_t i = 0; i < n; ++i) {
		buf[i] = std::complex<double>(static_cast<double>(s[i]) * win[i], 0.0);
	}
	sw::dsp::spectral::fft_forward<double>(buf);

	const std::size_t half = fft_n / 2 + 1;
	spec.freqs_hz.resize(half);
	spec.phases_rad.resize(half);
	const double fs = probe.sample_rate();
	for (std::size_t k = 0; k < half; ++k) {
		spec.freqs_hz[k] = static_cast<double>(k) * fs
		                    / static_cast<double>(fft_n);
		spec.phases_rad[k] = std::arg(buf[k]);
	}
	if (unwrap) detail::unwrap_phase(spec.phases_rad);
	return spec;
}

// Complex-probe phase spectrum: two-sided, fftshift order.
template <DspScalar T>
PhaseSpectrum<T> phase_spectrum(const SignalProbe<std::complex<T>>& probe,
                                  bool unwrap = true,
                                  WindowType window = WindowType::Hamming,
                                  double kaiser_beta = 8.6) {
	PhaseSpectrum<T> spec;
	spec.label = probe.label();
	spec.unwrapped = unwrap;
	auto s = probe.samples();
	if (s.empty()) return spec;

	const std::size_t n = s.size();
	const std::size_t fft_n = detail::next_pow2(n);
	const auto win = detail::build_window(n, window, kaiser_beta);

	mtl::vec::dense_vector<std::complex<double>> buf(fft_n,
	                                                    std::complex<double>{});
	for (std::size_t i = 0; i < n; ++i) {
		const auto& z = s[i];
		buf[i] = std::complex<double>(static_cast<double>(z.real()) * win[i],
		                                static_cast<double>(z.imag()) * win[i]);
	}
	sw::dsp::spectral::fft_forward<double>(buf);

	spec.freqs_hz.resize(fft_n);
	spec.phases_rad.resize(fft_n);
	const double fs = probe.sample_rate();
	for (std::size_t k = 0; k < fft_n; ++k) {
		std::size_t out_k = (k + fft_n / 2) % fft_n;
		spec.freqs_hz[k] = (static_cast<double>(k)
		                     - static_cast<double>(fft_n) / 2.0)
		                    * fs / static_cast<double>(fft_n);
		spec.phases_rad[k] = std::arg(buf[out_k]);
	}
	if (unwrap) detail::unwrap_phase(spec.phases_rad);
	return spec;
}

// ============================================================================
// I/Q constellation (complex probes only)
// ============================================================================

template <class T>
struct IQConstellation {
	std::vector<double> i_values;
	std::vector<double> q_values;
	std::string         label;

	void dump_csv(const std::string& path) const {
		std::ofstream out(path);
		if (!out) throw std::runtime_error(
			"IQConstellation::dump_csv: cannot open " + path);
		out << "sample_index,i,q\n";
		out << std::setprecision(17);
		for (std::size_t i = 0; i < i_values.size(); ++i) {
			out << i << "," << i_values[i] << "," << q_values[i] << "\n";
		}
	}
};

template <DspScalar T>
IQConstellation<T> iq_constellation(
        const SignalProbe<std::complex<T>>& probe) {
	IQConstellation<T> c;
	c.label = probe.label();
	auto s = probe.samples();
	c.i_values.resize(s.size());
	c.q_values.resize(s.size());
	for (std::size_t i = 0; i < s.size(); ++i) {
		c.i_values[i] = static_cast<double>(s[i].real());
		c.q_values[i] = static_cast<double>(s[i].imag());
	}
	return c;
}

} // namespace sw::dsp::probe
