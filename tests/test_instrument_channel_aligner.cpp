// test_instrument_channel_aligner.cpp: tests for the multi-channel
// time-alignment wrapper.
//
// Coverage:
//   - Constructor validation: empty skews, nonzero skew[0] throw
//   - num_channels() reports the right count
//   - Single-channel passthrough (delays = {0.0})
//   - Two-channel zero-skew: outputs match inputs after the FIR transient
//   - **Skew-correction headline test**: synthesize two channels of the
//     same tone with a known 0.3-sample skew, run through ChannelAligner,
//     verify the output channels are correlated to within ~99%
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)`.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <sw/dsp/instrument/channel_aligner.hpp>

using namespace sw::dsp::instrument;

#define REQUIRE(cond) \
	do { if (!(cond)) throw std::runtime_error( \
		std::string("test failed: ") + #cond + \
		" at " __FILE__ ":" + std::to_string(__LINE__)); } while (0)

// ============================================================================
// Constructor validation
// ============================================================================

void test_ctor_empty_skews_throws() {
	bool threw = false;
	try {
		std::span<const double> empty;
		ChannelAligner<double> ca(empty);
	} catch (const std::invalid_argument&) { threw = true; }
	REQUIRE(threw);
	std::cout << "  ctor_empty_skews_throws: passed\n";
}

void test_ctor_nonzero_reference_skew_throws() {
	std::array<double, 2> skews = {0.1, 0.3};   // skew[0] != 0 — bug guard
	bool threw = false;
	try { ChannelAligner<double> ca(std::span<const double>{skews}); }
	catch (const std::invalid_argument&) { threw = true; }
	REQUIRE(threw);
	std::cout << "  ctor_nonzero_reference_skew_throws: passed\n";
}

void test_num_channels() {
	std::array<double, 4> skews = {0.0, 0.25, 0.5, 0.75};
	ChannelAligner<double> ca(std::span<const double>{skews});
	REQUIRE(ca.num_channels() == 4);
	std::cout << "  num_channels: passed\n";
}

// ============================================================================
// Single-channel + zero-skew passthrough
// ============================================================================

void test_single_channel() {
	std::array<double, 1> skews = {0.0};
	ChannelAligner<double> ca(std::span<const double>{skews});
	std::array<double, 1> in_buf;

	// Push a few samples and verify the channel-0 output is what came out
	// of the underlying FractionalDelay(0, 31). With delay=0, an impulse
	// at sample 0 reappears at sample 15 (group delay = (N-1)/2).
	std::vector<double> out_at_15;
	for (std::size_t n = 0; n < 32; ++n) {
		in_buf[0] = (n == 0 ? 1.0 : 0.0);
		auto out = ca.process(std::span<const double>{in_buf});
		REQUIRE(out.size() == 1);
		if (n == 15) out_at_15.push_back(out[0]);
	}
	REQUIRE(!out_at_15.empty());
	REQUIRE(std::abs(out_at_15[0] - 1.0) < 0.05);   // close to unity
	std::cout << "  single_channel: passed (impulse at sample 15 = "
	          << out_at_15[0] << ")\n";
}

void test_two_channels_zero_skew() {
	std::array<double, 2> skews = {0.0, 0.0};
	ChannelAligner<double> ca(std::span<const double>{skews});
	std::array<double, 2> in_buf;
	// Both channels get the same impulse; outputs should match.
	std::vector<double> ch0, ch1;
	for (std::size_t n = 0; n < 32; ++n) {
		in_buf[0] = (n == 0 ? 1.0 : 0.0);
		in_buf[1] = (n == 0 ? 1.0 : 0.0);
		auto out = ca.process(std::span<const double>{in_buf});
		ch0.push_back(out[0]);
		ch1.push_back(out[1]);
	}
	for (std::size_t n = 0; n < 32; ++n) {
		REQUIRE(std::abs(ch0[n] - ch1[n]) < 1e-12);
	}
	std::cout << "  two_channels_zero_skew: passed\n";
}

void test_process_wrong_size_throws() {
	std::array<double, 2> skews = {0.0, 0.3};
	ChannelAligner<double> ca(std::span<const double>{skews});
	std::array<double, 1> in_buf = {0.0};   // wrong length
	bool threw = false;
	try {
		(void)ca.process(std::span<const double>{in_buf});
	} catch (const std::invalid_argument&) { threw = true; }
	REQUIRE(threw);
	std::cout << "  process_wrong_size_throws: passed\n";
}

// ============================================================================
// Headline test: skew correction
// ============================================================================

// Compute Pearson correlation coefficient between two equal-length streams,
// skipping the first `skip` samples to avoid the FIR transient.
double pearson_correlation(const std::vector<double>& a,
                           const std::vector<double>& b,
                           std::size_t skip) {
	const std::size_t N = a.size();
	double sum_a = 0.0, sum_b = 0.0;
	for (std::size_t i = skip; i < N; ++i) {
		sum_a += a[i];
		sum_b += b[i];
	}
	const double mean_a = sum_a / static_cast<double>(N - skip);
	const double mean_b = sum_b / static_cast<double>(N - skip);
	double num = 0.0, den_a = 0.0, den_b = 0.0;
	for (std::size_t i = skip; i < N; ++i) {
		const double da = a[i] - mean_a;
		const double db = b[i] - mean_b;
		num   += da * db;
		den_a += da * da;
		den_b += db * db;
	}
	return num / std::sqrt(den_a * den_b);
}

void test_skew_correction_two_channels() {
	// Convention: skews[c] is the magnitude of the per-channel delay.
	// Since the wrapper only does causal delays in [0, 1), it can only
	// align channels that are sampled LATER than the reference (delay
	// pulls their effective time BACK toward the reference). The
	// reference (skews[0]=0) must therefore be the EARLIEST-sampling
	// channel; other channels are sampled LATER by skews[c] samples and
	// get delayed by exactly that amount to align with the reference.
	//
	// Setup:
	//   ground truth signal: s(t) = sin(2πf t/T)
	//   channel 0 samples at t = n*T:       ch0_in[n] = sin(2πf*n)
	//   channel 1 samples 0.3T LATER:       ch1_in[n] = sin(2πf*(n + 0.3))
	//
	// Channel 1's sample n already represents the signal value at time
	// (n + 0.3)*T — 0.3 sample periods AHEAD of channel 0's grid. The
	// FractionalDelay(0.3) on channel 1 produces an output sample that
	// represents the input from 0.3 samples earlier, i.e., the signal
	// at time ((n + 0.3) - 0.3)*T = n*T — back in line with channel 0.
	//
	// After both channels go through the shared 15-sample FIR group
	// delay, the two outputs both represent s((n - 15)*T) and should be
	// highly correlated.

	const double pi = std::numbers::pi_v<double>;
	const double f  = 0.20;
	const std::size_t N_in = 256;

	std::vector<double> ch0_in(N_in), ch1_in(N_in);
	for (std::size_t n = 0; n < N_in; ++n) {
		ch0_in[n] = std::sin(2.0 * pi * f * static_cast<double>(n));
		ch1_in[n] = std::sin(2.0 * pi * f *
		                      (static_cast<double>(n) + 0.3));   // 0.3 late
	}

	std::array<double, 2> skews = {0.0, 0.3};
	ChannelAligner<double> ca(std::span<const double>{skews});

	std::vector<double> ch0_out(N_in), ch1_out(N_in);
	std::array<double, 2> in_buf;
	for (std::size_t n = 0; n < N_in; ++n) {
		in_buf[0] = ch0_in[n];
		in_buf[1] = ch1_in[n];
		auto out = ca.process(std::span<const double>{in_buf});
		ch0_out[n] = out[0];
		ch1_out[n] = out[1];
	}
	// Skip transient (first 31 = num_taps samples).
	const double rho = pearson_correlation(ch0_out, ch1_out, /*skip=*/31);
	if (!(rho > 0.99))
		throw std::runtime_error(
			"skew correction: correlation = " + std::to_string(rho) +
			" (expected > 0.99)");
	std::cout << "  skew_correction_two_channels: passed (correlation="
	          << rho << ")\n";
}

void test_uncorrected_skew_has_lower_correlation() {
	// Sanity check on the test setup: WITHOUT alignment, two channels
	// with a 0.3-sample skew should be visibly less correlated. If the
	// uncorrected correlation is already 1.0, the headline test isn't
	// proving anything.
	const double pi = std::numbers::pi_v<double>;
	const double f  = 0.20;
	const std::size_t N = 256;

	std::vector<double> ch0(N), ch1(N);
	for (std::size_t n = 0; n < N; ++n) {
		ch0[n] = std::sin(2.0 * pi * f * static_cast<double>(n));
		ch1[n] = std::sin(2.0 * pi * f *
		                   (static_cast<double>(n) + 0.3));   // 0.3 late
	}
	const double rho_uncorrected = pearson_correlation(ch0, ch1, /*skip=*/0);
	REQUIRE(rho_uncorrected < 0.99);   // there really is a meaningful skew
	std::cout << "  uncorrected_skew_has_lower_correlation: passed "
	             "(uncorrected rho=" << rho_uncorrected << ")\n";
}

// ============================================================================
// main
// ============================================================================

int main() {
	try {
		std::cout << "test_instrument_channel_aligner\n";

		test_ctor_empty_skews_throws();
		test_ctor_nonzero_reference_skew_throws();
		test_num_channels();

		test_single_channel();
		test_two_channels_zero_skew();
		test_process_wrong_size_throws();

		test_uncorrected_skew_has_lower_correlation();
		test_skew_correction_two_channels();

		std::cout << "all tests passed\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "FAILED: " << ex.what() << "\n";
		return 1;
	}
}
