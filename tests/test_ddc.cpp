// test_ddc.cpp: test Digital Down-Converter (DDC)
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/acquisition/ddc.hpp>
#include <sw/dsp/acquisition/cic.hpp>
#include <sw/dsp/acquisition/halfband.hpp>
#include <sw/dsp/filter/fir/polyphase.hpp>
#include <sw/dsp/filter/fir/fir_design.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/spectral/fft.hpp>
#include <sw/dsp/windows/hamming.hpp>

#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <universal/number/posit/posit.hpp>

using namespace sw::dsp;
using sw::dsp::spectral::fft_forward;

bool near(double a, double b, double eps = 1e-6) {
	return std::abs(a - b) < eps;
}

// Design a lowpass polyphase decimator for use in DDC tests.
// Cutoff = 0.1 (normalized to input fs) gives a 4800 Hz passband at fs=48000.
static PolyphaseDecimator<double> make_polyphase_decimator(std::size_t factor) {
	std::size_t num_taps = 64 * factor + 1;
	auto window = hamming_window<double>(num_taps);
	auto taps   = design_fir_lowpass<double>(num_taps, 0.45 / static_cast<double>(factor), window);
	return PolyphaseDecimator<double>(taps, factor);
}

// ============================================================================
// Construction and accessors
// ============================================================================

void test_construction() {
	auto decim = make_polyphase_decimator(4);
	DDC<double> ddc(6000.0, 48000.0, decim);

	if (!near(ddc.center_frequency(), 6000.0, 1e-12))
		throw std::runtime_error("test failed: center_frequency = " +
			std::to_string(ddc.center_frequency()));
	if (!near(ddc.sample_rate(), 48000.0, 1e-12))
		throw std::runtime_error("test failed: sample_rate = " +
			std::to_string(ddc.sample_rate()));

	std::cout << "  construction: passed\n";
}

// ============================================================================
// Parameter validation
// ============================================================================

void test_validation() {
	auto decim = make_polyphase_decimator(4);

	bool caught = false;
	try { DDC<double> ddc(6000.0, 0.0, decim); }
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: zero sample_rate should throw");

	caught = false;
	try { DDC<double> ddc(6000.0, -48000.0, decim); }
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: negative sample_rate should throw");

	std::cout << "  validation: passed\n";
}

// ============================================================================
// Core acceptance: tone at IF translates to DC
// ============================================================================

void test_tone_to_dc() {
	double fs = 48000.0;
	double f_if = 6000.0;
	std::size_t N = 4096;
	std::size_t R = 4;

	auto decim = make_polyphase_decimator(R);
	DDC<double> ddc(f_if, fs, decim);

	// Generate real tone at IF
	mtl::vec::dense_vector<double> input(N);
	for (std::size_t n = 0; n < N; ++n) {
		input[n] = std::cos(2.0 * pi * f_if * static_cast<double>(n) / fs);
	}

	auto out = ddc.process_block(input);

	// Expected output count: N / R (plus possibly one from phase 0 at start)
	if (out.size() < N / R - 1 || out.size() > N / R + 1)
		throw std::runtime_error("test failed: unexpected output size = " +
			std::to_string(out.size()));

	// Skip filter transient: group delay / R samples
	std::size_t skip = 32;
	if (out.size() <= skip + 64)
		throw std::runtime_error("test failed: not enough post-transient samples");

	// Measure mean magnitude in steady state; should be near 0.5 (filter DC gain near 1)
	double sum_mag = 0.0;
	std::size_t count = 0;
	for (std::size_t i = skip; i < out.size(); ++i) {
		double mag = std::abs(static_cast<std::complex<double>>(out[i]));
		sum_mag += mag;
		++count;
	}
	double mean_mag = sum_mag / static_cast<double>(count);

	if (std::abs(mean_mag - 0.5) > 0.05)
		throw std::runtime_error("test failed: mean |DDC out| = " +
			std::to_string(mean_mag) + ", expected ~0.5");

	// Also FFT the steady-state output and verify DC is the peak
	std::size_t M = 1;
	while (M * 2 <= out.size() - skip) M *= 2;
	mtl::vec::dense_vector<std::complex<double>> fft_buf(M);
	for (std::size_t i = 0; i < M; ++i) {
		fft_buf[i] = static_cast<std::complex<double>>(out[skip + i]);
	}
	fft_forward<double>(fft_buf);

	double peak_mag = 0.0;
	std::size_t peak_bin = 0;
	for (std::size_t k = 0; k < M; ++k) {
		double mag = std::abs(fft_buf[k]);
		if (mag > peak_mag) { peak_mag = mag; peak_bin = k; }
	}

	if (peak_bin != 0)
		throw std::runtime_error("test failed: spectrum peak at bin " +
			std::to_string(peak_bin) + ", expected bin 0 (DC)");

	std::cout << "  tone_to_dc: mean |out| = " << mean_mag
	          << ", spectral peak at DC, passed\n";
}

// ============================================================================
// Frequency offset: tone at IF+df translates to df
// ============================================================================

void test_frequency_offset() {
	double fs = 48000.0;
	double f_if = 6000.0;
	double df = 1000.0;
	std::size_t N = 4096;
	std::size_t R = 4;

	auto decim = make_polyphase_decimator(R);
	DDC<double> ddc(f_if, fs, decim);

	mtl::vec::dense_vector<double> input(N);
	for (std::size_t n = 0; n < N; ++n) {
		input[n] = std::cos(2.0 * pi * (f_if + df) * static_cast<double>(n) / fs);
	}

	auto out = ddc.process_block(input);

	std::size_t skip = 32;
	std::size_t M = 1;
	while (M * 2 <= out.size() - skip) M *= 2;

	mtl::vec::dense_vector<std::complex<double>> fft_buf(M);
	for (std::size_t i = 0; i < M; ++i) {
		fft_buf[i] = static_cast<std::complex<double>>(out[skip + i]);
	}
	fft_forward<double>(fft_buf);

	double peak_mag = 0.0;
	std::size_t peak_bin = 0;
	for (std::size_t k = 0; k < M; ++k) {
		double mag = std::abs(fft_buf[k]);
		if (mag > peak_mag) { peak_mag = mag; peak_bin = k; }
	}

	// Expected bin: df / (fs / R / M) = df * M * R / fs
	double fs_out = fs / static_cast<double>(R);
	double expected_bin = df * static_cast<double>(M) / fs_out;
	double measured_freq = static_cast<double>(peak_bin) * fs_out / static_cast<double>(M);

	if (std::abs(measured_freq - df) > fs_out / static_cast<double>(M))
		throw std::runtime_error("test failed: baseband tone at " +
			std::to_string(measured_freq) + " Hz, expected " + std::to_string(df));

	std::cout << "  frequency_offset: peak at bin " << peak_bin
	          << " (" << measured_freq << " Hz), expected ~" << expected_bin
	          << ", passed\n";
}

// ============================================================================
// Image rejection: tone in the filter stopband is attenuated
// ============================================================================

void test_image_rejection() {
	double fs = 48000.0;
	double f_if = 6000.0;
	std::size_t N = 4096;
	std::size_t R = 4;

	// In-band reference: tone at f_if
	{
		auto decim = make_polyphase_decimator(R);
		DDC<double> ddc(f_if, fs, decim);
		mtl::vec::dense_vector<double> input(N);
		for (std::size_t n = 0; n < N; ++n) {
			input[n] = std::cos(2.0 * pi * f_if * static_cast<double>(n) / fs);
		}
		auto out = ddc.process_block(input);
		std::size_t skip = 32;
		double ref_mag = 0.0;
		std::size_t cnt = 0;
		for (std::size_t i = skip; i < out.size(); ++i) {
			ref_mag += std::abs(static_cast<std::complex<double>>(out[i]));
			++cnt;
		}
		ref_mag /= static_cast<double>(cnt);

		// Out-of-band interferer at f_if + 18000 = 24000 (Nyquist of input)
		auto decim2 = make_polyphase_decimator(R);
		DDC<double> ddc2(f_if, fs, decim2);
		mtl::vec::dense_vector<double> input2(N);
		double f_interferer = f_if + 15000.0;  // 21000 Hz, well into stopband after mix
		for (std::size_t n = 0; n < N; ++n) {
			input2[n] = std::cos(2.0 * pi * f_interferer * static_cast<double>(n) / fs);
		}
		auto out2 = ddc2.process_block(input2);
		double int_mag = 0.0;
		cnt = 0;
		for (std::size_t i = skip; i < out2.size(); ++i) {
			int_mag += std::abs(static_cast<std::complex<double>>(out2[i]));
			++cnt;
		}
		int_mag /= static_cast<double>(cnt);

		double rejection_db = 20.0 * std::log10(ref_mag / (int_mag + 1e-30));
		if (rejection_db < 30.0)
			throw std::runtime_error("test failed: image rejection only " +
				std::to_string(rejection_db) + " dB (need >= 30)");

		std::cout << "  image_rejection: " << rejection_db << " dB, passed\n";
	}
}

// ============================================================================
// Streaming vs block processing equivalence
// ============================================================================

void test_stream_vs_block() {
	double fs = 48000.0;
	double f_if = 6000.0;
	std::size_t N = 512;
	std::size_t R = 4;

	mtl::vec::dense_vector<double> input(N);
	for (std::size_t n = 0; n < N; ++n) {
		input[n] = std::cos(2.0 * pi * (f_if + 500.0) * static_cast<double>(n) / fs);
	}

	auto decim_a = make_polyphase_decimator(R);
	DDC<double> ddc_a(f_if, fs, decim_a);
	auto out_block = ddc_a.process_block(input);

	auto decim_b = make_polyphase_decimator(R);
	DDC<double> ddc_b(f_if, fs, decim_b);
	std::vector<std::complex<double>> stream_out;
	for (std::size_t n = 0; n < N; ++n) {
		auto [ready, z] = ddc_b.process(input[n]);
		if (ready) stream_out.push_back(static_cast<std::complex<double>>(z));
	}

	if (stream_out.size() != out_block.size())
		throw std::runtime_error("test failed: stream/block size mismatch " +
			std::to_string(stream_out.size()) + " vs " + std::to_string(out_block.size()));

	for (std::size_t i = 0; i < stream_out.size(); ++i) {
		auto zb = static_cast<std::complex<double>>(out_block[i]);
		double ie = std::abs(stream_out[i].real() - zb.real());
		double qe = std::abs(stream_out[i].imag() - zb.imag());
		if (ie > 1e-12 || qe > 1e-12)
			throw std::runtime_error("test failed: stream[" + std::to_string(i) +
				"] differs from block (ie=" + std::to_string(ie) + " qe=" + std::to_string(qe) + ")");
	}

	std::cout << "  stream_vs_block: " << stream_out.size() << " samples match, passed\n";
}

// ============================================================================
// DDC with half-band decimator (2:1)
// ============================================================================

void test_halfband_decimator() {
	double fs = 48000.0;
	double f_if = 8000.0;
	std::size_t N = 2048;

	auto taps = design_halfband<double>(31, 0.1);
	HalfBandFilter<double> hb(taps);
	DDC<double, double, double, HalfBandFilter<double>> ddc(f_if, fs, hb);

	mtl::vec::dense_vector<double> input(N);
	for (std::size_t n = 0; n < N; ++n) {
		input[n] = std::cos(2.0 * pi * f_if * static_cast<double>(n) / fs);
	}

	auto out = ddc.process_block(input);

	// Half-band decimates by 2, so we expect N/2 outputs
	if (out.size() < N / 2 - 1 || out.size() > N / 2 + 1)
		throw std::runtime_error("test failed: half-band output size = " +
			std::to_string(out.size()));

	std::size_t skip = 32;
	double sum_mag = 0.0;
	std::size_t cnt = 0;
	for (std::size_t i = skip; i < out.size(); ++i) {
		sum_mag += std::abs(static_cast<std::complex<double>>(out[i]));
		++cnt;
	}
	double mean_mag = sum_mag / static_cast<double>(cnt);

	if (std::abs(mean_mag - 0.5) > 0.05)
		throw std::runtime_error("test failed: half-band DDC mean |out| = " +
			std::to_string(mean_mag) + ", expected ~0.5");

	std::cout << "  halfband_decimator: mean |out| = " << mean_mag << ", passed\n";
}

// ============================================================================
// DDC with CIC decimator
// ============================================================================

void test_cic_decimator() {
	double fs = 48000.0;
	// CIC is a boxcar-like filter: its first null is at fs/R. With R=4 and
	// f_if=6000, the complex LO shifts the IF tone to DC, and CIC's lowpass
	// response near DC gives gain (R*D)^M = 64 (for R=4, D=1, M=3).
	double f_if = 3000.0;
	std::size_t N = 2048;
	int R = 4;
	int M = 3;

	CICDecimator<double> cic(R, M);
	DDC<double, double, double, CICDecimator<double>> ddc(f_if, fs, cic);

	mtl::vec::dense_vector<double> input(N);
	for (std::size_t n = 0; n < N; ++n) {
		input[n] = std::cos(2.0 * pi * f_if * static_cast<double>(n) / fs);
	}

	auto out = ddc.process_block(input);

	// Expected CIC gain: (R*D)^M
	double cic_gain = std::pow(static_cast<double>(R), static_cast<double>(M));
	// After DDC, DC amplitude should be 0.5 * cic_gain
	double expected = 0.5 * cic_gain;

	std::size_t skip = 16;
	double sum_mag = 0.0;
	std::size_t cnt = 0;
	for (std::size_t i = skip; i < out.size(); ++i) {
		sum_mag += std::abs(static_cast<std::complex<double>>(out[i]));
		++cnt;
	}
	double mean_mag = sum_mag / static_cast<double>(cnt);

	// Tolerate 5% — CIC has non-flat passband near DC
	if (std::abs(mean_mag - expected) > 0.05 * expected)
		throw std::runtime_error("test failed: CIC DDC mean |out| = " +
			std::to_string(mean_mag) + ", expected ~" + std::to_string(expected));

	std::cout << "  cic_decimator: mean |out| = " << mean_mag
	          << " (expected " << expected << "), passed\n";
}

// ============================================================================
// Reset
// ============================================================================

void test_reset() {
	double fs = 48000.0;
	double f_if = 6000.0;
	std::size_t N = 256;
	std::size_t R = 4;

	auto decim1 = make_polyphase_decimator(R);
	DDC<double> ddc(f_if, fs, decim1);

	mtl::vec::dense_vector<double> input(N);
	for (std::size_t n = 0; n < N; ++n) {
		input[n] = std::cos(2.0 * pi * f_if * static_cast<double>(n) / fs);
	}

	auto out1 = ddc.process_block(input);
	ddc.reset();
	auto out2 = ddc.process_block(input);

	if (out1.size() != out2.size())
		throw std::runtime_error("test failed: reset changed output size");

	for (std::size_t i = 0; i < out1.size(); ++i) {
		auto z1 = static_cast<std::complex<double>>(out1[i]);
		auto z2 = static_cast<std::complex<double>>(out2[i]);
		if (std::abs(z1.real() - z2.real()) > 1e-12 ||
		    std::abs(z1.imag() - z2.imag()) > 1e-12)
			throw std::runtime_error("test failed: reset did not reproduce output");
	}

	std::cout << "  reset: passed\n";
}

// ============================================================================
// Set center frequency (retune)
// ============================================================================

void test_set_center_frequency() {
	double fs = 48000.0;
	double f1 = 6000.0;
	double f2 = 10000.0;
	std::size_t N = 2048;
	std::size_t R = 4;

	auto decim = make_polyphase_decimator(R);
	DDC<double> ddc(f1, fs, decim);

	// Retune to f2
	ddc.set_center_frequency(f2);
	if (!near(ddc.center_frequency(), f2, 1e-12))
		throw std::runtime_error("test failed: center_frequency after retune");

	// Feed tone at f2 -- should come out at DC
	mtl::vec::dense_vector<double> input(N);
	for (std::size_t n = 0; n < N; ++n) {
		input[n] = std::cos(2.0 * pi * f2 * static_cast<double>(n) / fs);
	}
	auto out = ddc.process_block(input);

	std::size_t skip = 32;
	double sum_mag = 0.0;
	std::size_t cnt = 0;
	for (std::size_t i = skip; i < out.size(); ++i) {
		sum_mag += std::abs(static_cast<std::complex<double>>(out[i]));
		++cnt;
	}
	double mean_mag = sum_mag / static_cast<double>(cnt);

	if (std::abs(mean_mag - 0.5) > 0.05)
		throw std::runtime_error("test failed: retuned DDC mean |out| = " +
			std::to_string(mean_mag));

	std::cout << "  set_center_frequency: mean |out| = " << mean_mag << ", passed\n";
}

// ============================================================================
// Mixed-precision: posit<32,2> for state and samples
// ============================================================================

void test_mixed_precision_posit() {
	using posit_t = sw::universal::posit<32, 2>;

	posit_t fs(48000.0);
	posit_t f_if(6000.0);
	std::size_t N = 1024;
	std::size_t R = 4;

	// Design taps in double then cast
	std::size_t num_taps = 64 * R + 1;
	auto window = hamming_window<double>(num_taps);
	auto taps_d = design_fir_lowpass<double>(num_taps, 0.45 / static_cast<double>(R), window);
	mtl::vec::dense_vector<posit_t> taps(num_taps);
	for (std::size_t i = 0; i < num_taps; ++i) taps[i] = posit_t(taps_d[i]);

	PolyphaseDecimator<posit_t> decim(taps, R);
	DDC<posit_t> ddc(f_if, fs, decim);

	mtl::vec::dense_vector<posit_t> input(N);
	for (std::size_t n = 0; n < N; ++n) {
		input[n] = posit_t(std::cos(2.0 * pi * 6000.0 * static_cast<double>(n) / 48000.0));
	}

	auto out = ddc.process_block(input);

	std::size_t skip = 32;
	if (out.size() <= skip + 32)
		throw std::runtime_error("test failed: posit DDC not enough samples");

	double sum_mag = 0.0;
	std::size_t cnt = 0;
	for (std::size_t i = skip; i < out.size(); ++i) {
		double r = static_cast<double>(out[i].real());
		double q = static_cast<double>(out[i].imag());
		sum_mag += std::sqrt(r*r + q*q);
		++cnt;
	}
	double mean_mag = sum_mag / static_cast<double>(cnt);

	if (std::abs(mean_mag - 0.5) > 0.05)
		throw std::runtime_error("test failed: posit DDC mean |out| = " +
			std::to_string(mean_mag));

	std::cout << "  mixed_precision_posit: mean |out| = " << mean_mag << ", passed\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
	try {
		std::cout << "DDC tests\n";
		test_construction();
		test_validation();
		test_tone_to_dc();
		test_frequency_offset();
		test_image_rejection();
		test_stream_vs_block();
		test_halfband_decimator();
		test_cic_decimator();
		test_reset();
		test_set_center_frequency();
		test_mixed_precision_posit();
		std::cout << "All DDC tests passed.\n";
	} catch (const std::exception& e) {
		std::cerr << "FAIL: " << e.what() << "\n";
		return 1;
	}
	return 0;
}
