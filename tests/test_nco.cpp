// test_nco.cpp: test Numerically Controlled Oscillator
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/acquisition/nco.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/spectral/fft.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <universal/number/posit/posit.hpp>

using namespace sw::dsp;
using sw::dsp::spectral::fft;

bool near(double a, double b, double eps = 1e-6) {
	return std::abs(a - b) < eps;
}

// ============================================================================
// Basic construction and accessors
// ============================================================================

void test_construction() {
	NCO<double> nco(1000.0, 48000.0);

	double expected_inc = 1000.0 / 48000.0;
	if (!near(nco.phase_increment(), expected_inc, 1e-12))
		throw std::runtime_error("test failed: phase_increment = " +
			std::to_string(nco.phase_increment()));

	if (!near(nco.phase(), 0.0, 1e-15))
		throw std::runtime_error("test failed: initial phase not zero");

	std::cout << "  construction: passed\n";
}

// ============================================================================
// Parameter validation
// ============================================================================

void test_validation() {
	bool caught = false;

	try { NCO<double>(100.0, 0.0); }
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: zero sample_rate should throw");

	caught = false;
	try { NCO<double>(100.0, -1000.0); }
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: negative sample_rate should throw");

	caught = false;
	try {
		NCO<double> nco(100.0, 1000.0);
		nco.set_frequency(200.0, 0.0);
	}
	catch (const std::invalid_argument&) { caught = true; }
	if (!caught) throw std::runtime_error("test failed: set_frequency with zero rate should throw");

	std::cout << "  validation: passed\n";
}

// ============================================================================
// Output frequency matches requested frequency
// ============================================================================

void test_frequency_accuracy() {
	double fs = 8192.0;
	double f0 = 1000.0;
	std::size_t N = 8192;

	NCO<double> nco(f0, fs);
	auto signal = nco.generate_block_real(N);

	auto spectrum = fft(signal);

	double max_mag = 0.0;
	std::size_t max_bin = 0;
	for (std::size_t k = 1; k < N / 2; ++k) {
		double mag = std::abs(static_cast<std::complex<double>>(spectrum[k]));
		if (mag > max_mag) {
			max_mag = mag;
			max_bin = k;
		}
	}

	double measured_freq = static_cast<double>(max_bin) * fs / static_cast<double>(N);
	if (!near(measured_freq, f0, fs / static_cast<double>(N)))
		throw std::runtime_error("test failed: measured_freq = " +
			std::to_string(measured_freq) + ", expected " + std::to_string(f0));

	std::cout << "  frequency_accuracy: peak at bin " << max_bin
	          << " (" << measured_freq << " Hz), passed\n";
}

// ============================================================================
// I/Q output: cos + j*sin, unit magnitude
// ============================================================================

void test_iq_output() {
	NCO<double> nco(1000.0, 48000.0);

	double max_mag_err = 0.0;
	for (int i = 0; i < 1000; ++i) {
		auto iq = nco.generate_sample();
		double mag = std::sqrt(iq.real() * iq.real() + iq.imag() * iq.imag());
		double err = std::abs(mag - 1.0);
		if (err > max_mag_err) max_mag_err = err;
	}

	if (max_mag_err > 1e-7)
		throw std::runtime_error("test failed: max I/Q magnitude error = " +
			std::to_string(max_mag_err));

	std::cout << "  iq_output: max magnitude error = " << max_mag_err << ", passed\n";
}

// ============================================================================
// Phase continuity across block boundaries
// ============================================================================

void test_phase_continuity() {
	double fs = 48000.0;
	double f0 = 1000.0;
	NCO<double> nco(f0, fs);

	auto block1 = nco.generate_block(100);

	auto block2 = nco.generate_block(100);

	NCO<double> nco2(f0, fs);
	auto full_block = nco2.generate_block(200);

	double max_i_err = 0.0;
	double max_q_err = 0.0;
	for (std::size_t i = 0; i < 100; ++i) {
		double ie = std::abs(block1[i].real() - full_block[i].real());
		double qe = std::abs(block1[i].imag() - full_block[i].imag());
		if (ie > max_i_err) max_i_err = ie;
		if (qe > max_q_err) max_q_err = qe;
	}
	for (std::size_t i = 0; i < 100; ++i) {
		double ie = std::abs(block2[i].real() - full_block[100 + i].real());
		double qe = std::abs(block2[i].imag() - full_block[100 + i].imag());
		if (ie > max_i_err) max_i_err = ie;
		if (qe > max_q_err) max_q_err = qe;
	}

	if (max_i_err > 1e-7 || max_q_err > 1e-7)
		throw std::runtime_error("test failed: phase discontinuity, max_i_err=" +
			std::to_string(max_i_err) + " max_q_err=" + std::to_string(max_q_err));

	std::cout << "  phase_continuity: passed\n";
}

// ============================================================================
// Phase offset
// ============================================================================

void test_phase_offset() {
	double fs = 48000.0;
	double f0 = 0.0;
	NCO<double> nco(f0, fs);

	// Zero frequency, zero offset -> cos(0) = 1, sin(0) = 0
	auto iq0 = nco.generate_sample();
	if (!near(iq0.real(), 1.0, 1e-7) || !near(iq0.imag(), 0.0, 1e-7))
		throw std::runtime_error("test failed: zero freq/offset should give (1,0)");

	nco.reset();

	// Set phase offset to 0.25 (90 degrees) -> cos(pi/2) = 0, sin(pi/2) = 1
	nco.set_phase_offset(0.25);
	auto iq90 = nco.generate_sample();
	if (!near(iq90.real(), 0.0, 1e-7) || !near(iq90.imag(), 1.0, 1e-7))
		throw std::runtime_error("test failed: 90-degree offset should give (0,1), got (" +
			std::to_string(iq90.real()) + "," + std::to_string(iq90.imag()) + ")");

	std::cout << "  phase_offset: passed\n";
}

// ============================================================================
// Reset clears state
// ============================================================================

void test_reset() {
	NCO<double> nco(1000.0, 48000.0);

	nco.generate_block(100);
	if (near(nco.phase(), 0.0, 1e-12))
		throw std::runtime_error("test failed: phase should have advanced");

	nco.reset();
	if (!near(nco.phase(), 0.0, 1e-15))
		throw std::runtime_error("test failed: phase not zero after reset");

	auto block1 = nco.generate_block(50);
	nco.reset();
	auto block2 = nco.generate_block(50);

	for (std::size_t i = 0; i < 50; ++i) {
		if (!near(block1[i].real(), block2[i].real(), 1e-15) ||
		    !near(block1[i].imag(), block2[i].imag(), 1e-15))
			throw std::runtime_error("test failed: reset did not reproduce output");
	}

	std::cout << "  reset: passed\n";
}

// ============================================================================
// Negative frequency (clockwise rotation)
// ============================================================================

void test_negative_frequency() {
	double fs = 48000.0;
	double f0 = 1000.0;

	NCO<double> nco_pos(f0, fs);
	NCO<double> nco_neg(-f0, fs);

	for (int i = 0; i < 100; ++i) {
		auto pos = nco_pos.generate_sample();
		auto neg = nco_neg.generate_sample();

		// Negative frequency should be complex conjugate of positive
		if (!near(pos.real(), neg.real(), 1e-7))
			throw std::runtime_error("test failed: neg freq real mismatch at sample " +
				std::to_string(i));
		if (!near(pos.imag(), -neg.imag(), 1e-7))
			throw std::runtime_error("test failed: neg freq imag mismatch at sample " +
				std::to_string(i));
	}

	std::cout << "  negative_frequency: passed\n";
}

// ============================================================================
// Block generation: span overload
// ============================================================================

void test_block_span() {
	NCO<double> nco1(1000.0, 48000.0);
	NCO<double> nco2(1000.0, 48000.0);

	auto vec_out = nco1.generate_block(64);

	std::array<std::complex<double>, 64> span_out{};
	nco2.generate_block(std::span<std::complex<double>>(span_out));

	for (std::size_t i = 0; i < 64; ++i) {
		if (!near(vec_out[i].real(), span_out[i].real(), 1e-15) ||
		    !near(vec_out[i].imag(), span_out[i].imag(), 1e-15))
			throw std::runtime_error("test failed: span vs dense_vector mismatch at " +
				std::to_string(i));
	}

	std::cout << "  block_span: passed\n";
}

// ============================================================================
// Real-only block generation
// ============================================================================

void test_block_real() {
	double fs = 48000.0;
	double f0 = 1000.0;
	NCO<double> nco1(f0, fs);
	NCO<double> nco2(f0, fs);

	auto real_out = nco1.generate_block_real(100);
	auto complex_out = nco2.generate_block(100);

	for (std::size_t i = 0; i < 100; ++i) {
		if (!near(real_out[i], complex_out[i].real(), 1e-7))
			throw std::runtime_error("test failed: real block mismatch at " +
				std::to_string(i));
	}

	std::cout << "  block_real: passed\n";
}

// ============================================================================
// Mix-down: real signal * conj(NCO) produces baseband
// ============================================================================

void test_mix_down() {
	double fs = 8192.0;
	double f0 = 1000.0;
	std::size_t N = 8192;

	// Generate a test signal at f0
	NCO<double> sig_gen(f0, fs);
	auto sig = sig_gen.generate_block_real(N);

	// Mix down with NCO at same frequency -> baseband (DC)
	NCO<double> lo(f0, fs);
	auto mixed = lo.mix_down(sig);

	// After mixing, the signal should be at DC. Check that the real part
	// is roughly constant (cos^2 component) and imaginary is near zero
	// after averaging (sin*cos component averages to zero).
	double sum_real = 0.0;
	double sum_imag = 0.0;
	for (std::size_t i = 0; i < N; ++i) {
		sum_real += mixed[i].real();
		sum_imag += mixed[i].imag();
	}
	double avg_real = sum_real / static_cast<double>(N);
	double avg_imag = sum_imag / static_cast<double>(N);

	// cos(x)*cos(x) averages to 0.5
	if (!near(avg_real, 0.5, 0.01))
		throw std::runtime_error("test failed: mix_down avg_real = " +
			std::to_string(avg_real) + ", expected ~0.5");

	// sin(x)*cos(x) averages to 0
	if (!near(avg_imag, 0.0, 0.01))
		throw std::runtime_error("test failed: mix_down avg_imag = " +
			std::to_string(avg_imag) + ", expected ~0.0");

	std::cout << "  mix_down: avg_real=" << avg_real
	          << " avg_imag=" << avg_imag << ", passed\n";
}

// ============================================================================
// Frequency change mid-stream
// ============================================================================

void test_frequency_change() {
	double fs = 48000.0;
	NCO<double> nco(1000.0, fs);

	nco.generate_block(100);
	double phase_before = nco.phase();

	nco.set_frequency(2000.0, fs);
	double new_inc = 2000.0 / 48000.0;
	if (!near(nco.phase_increment(), new_inc, 1e-12))
		throw std::runtime_error("test failed: phase_increment after set_frequency");

	// Phase should not have been reset
	if (!near(nco.phase(), phase_before, 1e-12))
		throw std::runtime_error("test failed: phase changed by set_frequency");

	std::cout << "  frequency_change: passed\n";
}

// ============================================================================
// SFDR measurement: verify spur levels scale with precision
// ============================================================================

void test_sfdr_double() {
	double fs = 8192.0;
	double f0 = 1000.0;
	std::size_t N = 8192;

	NCO<double> nco(f0, fs);
	auto signal = nco.generate_block_real(N);

	auto spectrum = fft(signal);

	// Find the fundamental bin and its magnitude
	std::size_t fund_bin = static_cast<std::size_t>(
		std::round(f0 * static_cast<double>(N) / fs));
	double fund_mag = std::abs(static_cast<std::complex<double>>(spectrum[fund_bin]));

	// Find the largest spur (excluding DC and fundamental +/- 1 bin)
	double max_spur = 0.0;
	for (std::size_t k = 1; k < N / 2; ++k) {
		if (k >= fund_bin - 1 && k <= fund_bin + 1) continue;
		double mag = std::abs(static_cast<std::complex<double>>(spectrum[k]));
		if (mag > max_spur) max_spur = mag;
	}

	double sfdr_db = 20.0 * std::log10(fund_mag / max_spur);

	// Double precision should give very high SFDR (> 200 dB)
	std::cout << "  sfdr_double: SFDR = " << sfdr_db << " dB\n";
	if (sfdr_db < 100.0)
		throw std::runtime_error("test failed: double SFDR too low: " +
			std::to_string(sfdr_db) + " dB");

	std::cout << "  sfdr_double: passed\n";
}

void test_sfdr_float() {
	float fs = 8192.0f;
	float f0 = 1000.0f;
	std::size_t N = 8192;

	NCO<float> nco(f0, fs);
	auto signal = nco.generate_block_real(N);

	// Convert to double for FFT analysis
	mtl::vec::dense_vector<double> signal_d(N);
	for (std::size_t i = 0; i < N; ++i)
		signal_d[i] = static_cast<double>(signal[i]);

	auto spectrum = fft(signal_d);

	std::size_t fund_bin = static_cast<std::size_t>(
		std::round(static_cast<double>(f0) * static_cast<double>(N) / static_cast<double>(fs)));
	double fund_mag = std::abs(spectrum[fund_bin]);
	double max_spur = 0.0;
	for (std::size_t k = 1; k < N / 2; ++k) {
		if (k >= fund_bin - 1 && k <= fund_bin + 1) continue;
		double mag = std::abs(spectrum[k]);
		if (mag > max_spur) max_spur = mag;
	}

	double sfdr_db = 20.0 * std::log10(fund_mag / max_spur);

	// Float precision (~24 bits mantissa): SFDR should be > 100 dB
	std::cout << "  sfdr_float: SFDR = " << sfdr_db << " dB\n";
	if (sfdr_db < 80.0)
		throw std::runtime_error("test failed: float SFDR too low: " +
			std::to_string(sfdr_db) + " dB");

	std::cout << "  sfdr_float: passed\n";
}

// ============================================================================
// Mixed precision: float state, double output
// ============================================================================

void test_mixed_precision() {
	double fs = 48000.0;
	double f0 = 1000.0;

	NCO<double, double> nco_dd(f0, fs);
	NCO<float, double> nco_fd(static_cast<float>(f0), static_cast<float>(fs));

	double max_err = 0.0;
	for (int i = 0; i < 1000; ++i) {
		auto dd = nco_dd.generate_sample();
		auto fd = nco_fd.generate_sample();
		double err_i = std::abs(dd.real() - fd.real());
		double err_q = std::abs(dd.imag() - fd.imag());
		double err = std::max(err_i, err_q);
		if (err > max_err) max_err = err;
	}

	// Float state introduces phase quantization error; should be small but measurable
	std::cout << "  mixed_precision: max error = " << max_err << "\n";
	if (max_err > 1e-4)
		throw std::runtime_error("test failed: mixed precision error too large: " +
			std::to_string(max_err));

	std::cout << "  mixed_precision: passed\n";
}

// ============================================================================
// Posit type: posit<32,2> as state scalar
// ============================================================================

void test_posit_type() {
	using p32 = sw::universal::posit<32, 2>;

	p32 fs(48000.0);
	p32 f0(1000.0);

	NCO<p32, p32> nco(f0, fs);

	// Reference with double
	NCO<double, double> nco_ref(1000.0, 48000.0);

	double max_err = 0.0;
	for (int i = 0; i < 200; ++i) {
		auto p_iq = nco.generate_sample();
		auto d_iq = nco_ref.generate_sample();

		double err_i = std::abs(static_cast<double>(p_iq.real()) - d_iq.real());
		double err_q = std::abs(static_cast<double>(p_iq.imag()) - d_iq.imag());
		double err = std::max(err_i, err_q);
		if (err > max_err) max_err = err;
	}

	std::cout << "  posit_type: max error vs double = " << max_err << "\n";
	if (max_err > 1e-5)
		throw std::runtime_error("test failed: posit error too large: " +
			std::to_string(max_err));

	std::cout << "  posit_type: passed\n";
}

// ============================================================================
// Phase wrapping: accumulator stays in [0, 1)
// ============================================================================

void test_phase_wrapping() {
	NCO<double> nco(1000.0, 48000.0);

	for (int i = 0; i < 100000; ++i) {
		nco.generate_sample();
		double p = nco.phase();
		if (p < 0.0 || p >= 1.0)
			throw std::runtime_error("test failed: phase out of range [0,1) at sample " +
				std::to_string(i) + ": phase = " + std::to_string(p));
	}

	std::cout << "  phase_wrapping: passed\n";
}

// ============================================================================
// DC (zero frequency): constant output
// ============================================================================

void test_dc_output() {
	NCO<double> nco(0.0, 48000.0);

	for (int i = 0; i < 100; ++i) {
		auto iq = nco.generate_sample();
		if (!near(iq.real(), 1.0, 1e-7) || !near(iq.imag(), 0.0, 1e-7))
			throw std::runtime_error("test failed: DC output not (1,0) at sample " +
				std::to_string(i));
	}

	std::cout << "  dc_output: passed\n";
}

// ============================================================================
// Nyquist frequency (fs/2): alternating +1/-1
// ============================================================================

void test_nyquist_frequency() {
	double fs = 48000.0;
	NCO<double> nco(fs / 2.0, fs);

	// At fs/2, phase increments by 0.5 each sample
	// cos(0) = 1, cos(pi) = -1, cos(2*pi) = 1, ...
	std::array<double, 4> expected = {1.0, -1.0, 1.0, -1.0};
	for (std::size_t i = 0; i < 4; ++i) {
		double y = nco.generate_real();
		if (!near(y, expected[i], 1e-7))
			throw std::runtime_error("test failed: nyquist sample " +
				std::to_string(i) + " = " + std::to_string(y) +
				", expected " + std::to_string(expected[i]));
	}

	std::cout << "  nyquist_frequency: passed\n";
}

// ============================================================================

int main() {
	try {
		std::cout << "NCO tests\n";
		test_construction();
		test_validation();
		test_frequency_accuracy();
		test_iq_output();
		test_phase_continuity();
		test_phase_offset();
		test_reset();
		test_negative_frequency();
		test_block_span();
		test_block_real();
		test_mix_down();
		test_frequency_change();
		test_sfdr_double();
		test_sfdr_float();
		test_mixed_precision();
		test_posit_type();
		test_phase_wrapping();
		test_dc_output();
		test_nyquist_frequency();
		std::cout << "All NCO tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAIL: " << e.what() << '\n';
		return 1;
	}
}
