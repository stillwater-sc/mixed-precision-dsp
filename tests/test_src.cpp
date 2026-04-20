// test_src.cpp: tests for rational sample-rate conversion
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/conditioning/src.hpp>
#include <sw/dsp/signals/generators.hpp>
#include <sw/dsp/quantization/sqnr.hpp>

#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>

#include <universal/number/posit/posit.hpp>
#include <universal/number/fixpnt/fixpnt.hpp>

using namespace sw::dsp;

void test_ratio_and_gcd() {
	// 44100 -> 48000: L=160, M=147 reduced by GCD
	RationalResampler<> r1(160, 147);
	if (r1.interp_factor() != 160 || r1.decim_factor() != 147)
		throw std::runtime_error("test failed: 160/147 should be coprime");

	// 48000 -> 16000: L=1, M=3
	RationalResampler<> r2(1, 3);
	if (r2.interp_factor() != 1 || r2.decim_factor() != 3)
		throw std::runtime_error("test failed: 1/3 ratio");

	// GCD reduction: 2/4 -> 1/2
	RationalResampler<> r3(2, 4);
	if (r3.interp_factor() != 1 || r3.decim_factor() != 2)
		throw std::runtime_error("test failed: 2/4 should reduce to 1/2");

	double expected = 160.0 / 147.0;
	if (std::abs(r1.ratio() - expected) > 1e-10)
		throw std::runtime_error("test failed: ratio() incorrect");

	std::cout << "  ratio_and_gcd: passed\n";
}

void test_identity_resampling() {
	// L=1, M=1 should pass through unchanged (minus filter delay)
	RationalResampler<> resampler(1, 1);
	auto sig = sine<double>(256, 100.0, 44100.0);
	auto out = resampler.process(sig);

	if (out.size() != sig.size())
		throw std::runtime_error("test failed: 1:1 output size mismatch, got " +
		                         std::to_string(out.size()));

	// After filter settles, output should match input (with delay)
	// Find peak correlation to identify the delay
	double best_corr = 0;
	std::size_t best_d = 0;
	for (std::size_t d = 0; d < 50; ++d) {
		double corr = 0;
		for (std::size_t i = d; i < sig.size(); ++i) {
			corr += static_cast<double>(sig[i - d]) * static_cast<double>(out[i]);
		}
		if (corr > best_corr) { best_corr = corr; best_d = d; }
	}

	// Verify the delayed output matches
	double max_err = 0;
	std::size_t skip = best_d + 20; // skip transient
	for (std::size_t i = skip; i < sig.size() - 10; ++i) {
		double err = std::abs(static_cast<double>(sig[i - best_d]) -
		                      static_cast<double>(out[i]));
		if (err > max_err) max_err = err;
	}
	// L=M=1 filter has cutoff at Nyquist with finite taps, so some ripple expected
	if (!(max_err < 0.1))
		throw std::runtime_error("test failed: 1:1 resampling max error " +
		                         std::to_string(max_err) + " (delay=" +
		                         std::to_string(best_d) + ")");

	std::cout << "  identity_resampling: passed (delay=" << best_d
	          << ", max_err=" << max_err << ")\n";
}

void test_44100_to_48000() {
	// 44100 → 48000 Hz: L=160, M=147
	double f_tone = 1000.0;
	double fs_in = 44100.0;
	double fs_out = 48000.0;
	std::size_t N_in = 4410; // 100ms at 44100

	mtl::vec::dense_vector<double> sig(N_in);
	for (std::size_t i = 0; i < N_in; ++i) {
		sig[i] = std::sin(2.0 * std::numbers::pi * f_tone *
		                  static_cast<double>(i) / fs_in);
	}

	RationalResampler<> resampler(160, 147, 10, 5.0);
	auto out = resampler.process(sig);

	// Expected output length: ~N_in * 160/147 = ~4800
	std::size_t expected_len = static_cast<std::size_t>(
		std::round(static_cast<double>(N_in) * 160.0 / 147.0));
	if (out.size() < expected_len - 10 || out.size() > expected_len + 10)
		throw std::runtime_error("test failed: 44100->48000 output length " +
		                         std::to_string(out.size()) + " expected ~" +
		                         std::to_string(expected_len));

	// Verify the 1 kHz tone is preserved in the output by measuring
	// the dominant frequency via zero-crossing rate
	std::size_t skip = 500; // skip filter transient
	std::size_t crossings = 0;
	for (std::size_t i = skip + 1; i < out.size(); ++i) {
		if ((static_cast<double>(out[i - 1]) < 0 && static_cast<double>(out[i]) >= 0) ||
		    (static_cast<double>(out[i - 1]) >= 0 && static_cast<double>(out[i]) < 0))
			++crossings;
	}
	double measured_freq = static_cast<double>(crossings) * fs_out /
	                       (2.0 * static_cast<double>(out.size() - skip));
	if (std::abs(measured_freq - f_tone) > 50.0)
		throw std::runtime_error("test failed: frequency shifted to " +
		                         std::to_string(measured_freq) + " Hz");

	std::cout << "  44100_to_48000: passed (out_len=" << out.size()
	          << ", measured_freq=" << measured_freq << " Hz)\n";
}

void test_48000_to_16000() {
	// 48000 → 16000 Hz: L=1, M=3 (pure decimation)
	double fs_in = 48000.0;
	std::size_t N_in = 4800; // 100ms

	// Signal: 1 kHz (should be preserved) + 10 kHz (should be attenuated)
	mtl::vec::dense_vector<double> sig(N_in);
	for (std::size_t i = 0; i < N_in; ++i) {
		double t = static_cast<double>(i) / fs_in;
		sig[i] = std::sin(2.0 * std::numbers::pi * 1000.0 * t)
		       + std::sin(2.0 * std::numbers::pi * 10000.0 * t);
	}

	RationalResampler<> resampler(1, 3, 10, 5.0);
	auto out = resampler.process(sig);

	// Expected ~1600 samples
	if (out.size() < 1500 || out.size() > 1700)
		throw std::runtime_error("test failed: 48000->16000 output length " +
		                         std::to_string(out.size()));

	// The 10 kHz tone is above Nyquist (8 kHz) of the output rate and
	// should be heavily attenuated. Measure RMS of output vs a pure
	// 1 kHz reference at the output rate.
	std::size_t skip = 200;
	double rms_out = 0;
	for (std::size_t i = skip; i < out.size(); ++i) {
		double v = static_cast<double>(out[i]);
		rms_out += v * v;
	}
	rms_out = std::sqrt(rms_out / static_cast<double>(out.size() - skip));

	// Pure 1 kHz at unit amplitude has RMS = 1/sqrt(2) ≈ 0.707
	// With the 10 kHz removed, output RMS should be close to this
	if (!(rms_out > 0.4 && rms_out < 1.0))
		throw std::runtime_error("test failed: output RMS " +
		                         std::to_string(rms_out) + " not in expected range");

	std::cout << "  48000_to_16000: passed (out_len=" << out.size()
	          << ", rms=" << rms_out << ")\n";
}

void test_roundtrip_snr() {
	// Resample 44100 → 48000 → 44100, measure SNR vs original.
	// Use short filter (half_length=3) to keep delay manageable, and
	// a long signal (1 second) so we have plenty of valid data.
	double fs1 = 44100.0;
	std::size_t N = 44100; // 1 second

	mtl::vec::dense_vector<double> sig(N);
	for (std::size_t i = 0; i < N; ++i) {
		sig[i] = std::sin(2.0 * std::numbers::pi * 440.0 *
		                  static_cast<double>(i) / fs1);
	}

	RationalResampler<> up(160, 147, 3, 5.0);
	RationalResampler<> down(147, 160, 3, 5.0);

	auto upsampled = up.process(sig);
	auto roundtrip = down.process(upsampled);

	// Search for best delay alignment. roundtrip[d] ≈ sig[0].
	// Each filter: ~2*3*160+1 = 961 taps, delay ~480 at 48kHz.
	// Total roundtrip delay at 44.1kHz ≈ 2 * 480 * (147/160) ≈ 882.
	double best_snr = -999;
	std::size_t best_d = 0;
	std::size_t skip = 1000; // skip transient at start of reference
	std::size_t compare_len = 20000;
	for (std::size_t d = 200; d < 2000 &&
	     d + skip + compare_len < roundtrip.size(); d += 5) {
		if (skip + compare_len > N) break;
		mtl::vec::dense_vector<double> r(compare_len);
		mtl::vec::dense_vector<double> a(compare_len);
		for (std::size_t i = 0; i < compare_len; ++i) {
			r[i] = sig[i + skip];
			a[i] = static_cast<double>(roundtrip[i + d + skip]);
		}
		double snr = sqnr_db(r, a);
		if (snr > best_snr) { best_snr = snr; best_d = d; }
	}

	// Refine around best_d
	for (std::size_t d = (best_d > 5 ? best_d - 5 : 0);
	     d < best_d + 5 && d + skip + compare_len < roundtrip.size(); ++d) {
		mtl::vec::dense_vector<double> r(compare_len);
		mtl::vec::dense_vector<double> a(compare_len);
		for (std::size_t i = 0; i < compare_len; ++i) {
			r[i] = sig[i + skip];
			a[i] = static_cast<double>(roundtrip[i + d + skip]);
		}
		double snr = sqnr_db(r, a);
		if (snr > best_snr) { best_snr = snr; best_d = d; }
	}

	if (!(best_snr > 40.0))
		throw std::runtime_error("test failed: roundtrip SNR " +
		                         std::to_string(best_snr) + " dB too low (delay=" +
		                         std::to_string(best_d) + ")");

	std::cout << "  roundtrip_snr: passed (" << best_snr << " dB, delay="
	          << best_d << ")\n";
}

void test_integer_upsample() {
	// L=4, M=1 should produce 4x as many samples
	RationalResampler<> resampler(4, 1);
	auto sig = sine<double>(100, 100.0, 1000.0);
	auto out = resampler.process(sig);

	if (out.size() != 400)
		throw std::runtime_error("test failed: 4x upsample length " +
		                         std::to_string(out.size()) + " expected 400");

	std::cout << "  integer_upsample: passed (out_len=" << out.size() << ")\n";
}

void test_integer_downsample() {
	// L=1, M=4 should produce 1/4 as many samples
	RationalResampler<> resampler(1, 4);
	auto sig = sine<double>(400, 10.0, 1000.0);
	auto out = resampler.process(sig);

	if (out.size() != 100)
		throw std::runtime_error("test failed: 4x downsample length " +
		                         std::to_string(out.size()) + " expected 100");

	std::cout << "  integer_downsample: passed (out_len=" << out.size() << ")\n";
}

void test_reset() {
	RationalResampler<> resampler(3, 2);
	auto sig = sine<double>(100, 440.0, 44100.0);

	auto out1 = resampler.process(sig);
	resampler.reset();
	auto out2 = resampler.process(sig);

	if (out1.size() != out2.size())
		throw std::runtime_error("test failed: reset output size mismatch");

	for (std::size_t i = 0; i < out1.size(); ++i) {
		if (out1[i] != out2[i])
			throw std::runtime_error("test failed: reset output mismatch at " +
			                         std::to_string(i));
	}

	std::cout << "  reset: passed\n";
}

void test_posit_types() {
	using p32 = sw::universal::posit<32, 2>;
	RationalResampler<double, double, p32> resampler(3, 2);

	std::size_t N = 100;
	mtl::vec::dense_vector<p32> sig(N);
	for (std::size_t i = 0; i < N; ++i) {
		sig[i] = p32(std::sin(2.0 * std::numbers::pi * 440.0 *
		                      static_cast<double>(i) / 44100.0));
	}

	auto out = resampler.process(sig);
	if (out.size() < 140 || out.size() > 160)
		throw std::runtime_error("test failed: posit output size " +
		                         std::to_string(out.size()));

	for (std::size_t i = 0; i < out.size(); ++i) {
		if (!std::isfinite(static_cast<double>(out[i])))
			throw std::runtime_error("test failed: posit output not finite at " +
			                         std::to_string(i));
	}

	std::cout << "  posit_types: passed (posit<32,2>, out_len=" << out.size() << ")\n";
}

void test_fixpnt_types() {
	using fxp = sw::universal::fixpnt<16, 8, sw::universal::Saturate, uint16_t>;
	RationalResampler<double, double, fxp> resampler(2, 3);

	std::size_t N = 150;
	mtl::vec::dense_vector<fxp> sig(N);
	for (std::size_t i = 0; i < N; ++i) {
		sig[i] = fxp(0.5 * std::sin(2.0 * std::numbers::pi * 440.0 *
		                             static_cast<double>(i) / 44100.0));
	}

	auto out = resampler.process(sig);
	if (out.size() < 90 || out.size() > 110)
		throw std::runtime_error("test failed: fixpnt output size " +
		                         std::to_string(out.size()));

	std::cout << "  fixpnt_types: passed (fixpnt<16,8>, out_len=" << out.size() << ")\n";
}

void test_invalid_args() {
	bool caught = false;
	try { RationalResampler<>(0, 1); } catch (const std::invalid_argument&) { caught = true; }
	if (!caught)
		throw std::runtime_error("test failed: should reject L=0");

	caught = false;
	try { RationalResampler<>(1, 0); } catch (const std::invalid_argument&) { caught = true; }
	if (!caught)
		throw std::runtime_error("test failed: should reject M=0");

	std::cout << "  invalid_args: passed\n";
}

int main() {
	try {
		std::cout << "Sample Rate Conversion Tests\n";

		test_ratio_and_gcd();
		test_identity_resampling();
		test_44100_to_48000();
		test_48000_to_16000();
		test_roundtrip_snr();
		test_integer_upsample();
		test_integer_downsample();
		test_reset();
		test_posit_types();
		test_fixpnt_types();
		test_invalid_args();

		std::cout << "All sample rate conversion tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << '\n';
		return 1;
	}
}
