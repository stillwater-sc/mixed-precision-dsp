// butterworth_lowpass.cpp: Butterworth lowpass filter demonstration
//
// Demonstrates mixed-precision filter processing, filter design,
// coefficient inspection, frequency response, pole-zero placement,
// impulse response, and signal filtering.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/filter/iir/butterworth.hpp>
#include <sw/dsp/filter/filter.hpp>
#include <sw/dsp/signals/generators.hpp>
#include <sw/dsp/viz/viz.hpp>

#include <sw/universal/number/fixpnt/fixpnt.hpp>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace sw::dsp;

void print_separator(const std::string& title) {
	std::cout << "\n" << std::string(78, '=') << "\n";
	std::cout << "  " << title << "\n";
	std::cout << std::string(78, '=') << "\n\n";
}

// Show biquad coefficients for a cascade
template <typename CascadeType>
void print_coefficients(const CascadeType& cascade) {
	for (int i = 0; i < cascade.num_stages(); ++i) {
		const auto& c = cascade.stage(i);
		std::cout << "  Stage " << i << ":\n";
		std::cout << "    b0=" << std::setw(12) << std::setprecision(8) << std::fixed
		          << static_cast<double>(c.b0)
		          << "  b1=" << std::setw(12) << static_cast<double>(c.b1)
		          << "  b2=" << std::setw(12) << static_cast<double>(c.b2) << "\n";
		std::cout << "    a1=" << std::setw(12) << static_cast<double>(c.a1)
		          << "  a2=" << std::setw(12) << static_cast<double>(c.a2) << "\n";
	}
}

// Run impulse response comparison between double reference and a mixed-precision config
template <typename MixedFilter>
void compare_impulse_response(const std::string& label, MixedFilter& filter_mixed,
                               double sample_rate, double cutoff_freq) {
	using sample_t = typename MixedFilter::sample_scalar;

	// Reference: pure double
	SimpleFilter<iir::ButterworthLowPass<4, double, double, double>> filter_ref;
	filter_ref.setup(4, sample_rate, cutoff_freq);
	filter_mixed.reset();

	constexpr int N = 100;
	double max_err = 0.0;
	double max_ref = 0.0;
	for (int n = 0; n < N; ++n) {
		double x_d = (n == 0) ? 1.0 : 0.0;
		sample_t x = static_cast<sample_t>(x_d);
		sample_t y_mixed = filter_mixed.process(x);
		double y_ref = filter_ref.process(x_d);
		double err = std::abs(static_cast<double>(y_mixed) - y_ref);
		max_err = std::max(max_err, err);
		max_ref = std::max(max_ref, std::abs(y_ref));
	}

	std::cout << "  " << label << ":\n";
	std::cout << "    Max absolute error: " << std::scientific << std::setprecision(6) << max_err;
	if (max_ref > 0) {
		std::cout << "  (relative: " << std::scientific << max_err / max_ref << ")";
	}
	std::cout << "\n";
}

int main() {
	constexpr double sample_rate = 44100.0;
	constexpr double cutoff_freq = 2000.0;

	std::cout << "Butterworth Lowpass Filter Demonstration\n";
	std::cout << "Sample rate: " << sample_rate << " Hz\n";
	std::cout << "Cutoff frequency: " << cutoff_freq << " Hz\n";

	// ================================================================
	// 1. Mixed-precision demonstration
	// ================================================================
	print_separator("Mixed-Precision Impulse Response Comparison");

	std::cout << "  4th order Butterworth LP at " << cutoff_freq << " Hz\n";
	std::cout << "  Coefficients designed in double, processing in target type.\n";
	std::cout << "  Reference: double coefficients, double state, double samples.\n\n";

	{
		// Native float: design in double, process in float
		SimpleFilter<iir::ButterworthLowPass<4, double, float, float>> filter_float;
		filter_float.setup(4, sample_rate, cutoff_freq);
		compare_impulse_response("float state + float samples (IEEE 754 binary32)",
		                          filter_float, sample_rate, cutoff_freq);
	}

	{
		// Universal fixpnt<16,8>: 16-bit fixed point with 8 fractional bits
		// Range: [-128, 127.996] with resolution 1/256 = 0.00390625
		using fp16_8 = sw::universal::fixpnt<16, 8>;
		SimpleFilter<iir::ButterworthLowPass<4, double, fp16_8, fp16_8>> filter_fixpnt;
		filter_fixpnt.setup(4, sample_rate, cutoff_freq);
		compare_impulse_response("fixpnt<16,8> state + samples (16-bit fixed, 8 frac bits)",
		                          filter_fixpnt, sample_rate, cutoff_freq);
	}

	// ================================================================
	// 2. Design filters of orders 2, 4, 6, 8 and show coefficients
	// ================================================================
	print_separator("Biquad Coefficients by Order");

	for (int order : {2, 4, 6, 8}) {
		iir::ButterworthLowPass<8> filter;
		filter.setup(order, sample_rate, cutoff_freq);
		std::cout << "Order " << order << " (" << filter.cascade().num_stages()
		          << " biquad stages):\n";
		print_coefficients(filter.cascade());
		std::cout << "\n";
	}

	// ================================================================
	// 3. Magnitude response
	// ================================================================
	print_separator("Magnitude Response (4th order Butterworth LP at 2 kHz)");

	{
		iir::ButterworthLowPass<4> filter;
		filter.setup(4, sample_rate, cutoff_freq);

		viz::PlotConfig cfg;
		cfg.width = 72;
		cfg.height = 20;
		cfg.y_min = -80.0;
		cfg.y_max = 6.0;
		cfg.title = "4th Order Butterworth LP - Magnitude Response";
		cfg.x_label = "0 Hz -> " + std::to_string(static_cast<int>(sample_rate / 2)) + " Hz";
		viz::plot_magnitude_response(std::cout, filter.cascade(), sample_rate, cfg);
	}

	// Show multiple orders overlaid (text: one per line with key)
	print_separator("Magnitude at Key Frequencies");

	std::cout << std::setw(8) << "Order"
	          << std::setw(12) << "DC (dB)"
	          << std::setw(14) << "Cutoff (dB)"
	          << std::setw(14) << "2xCutoff"
	          << std::setw(14) << "5xCutoff"
	          << std::setw(14) << "10xCutoff" << "\n";
	std::cout << std::string(76, '-') << "\n";

	for (int order : {1, 2, 3, 4, 5, 6, 7, 8}) {
		iir::ButterworthLowPass<8> filter;
		filter.setup(order, sample_rate, cutoff_freq);

		double fc = cutoff_freq / sample_rate;
		double db_dc     = 20.0 * std::log10(std::abs(filter.cascade().response(0.0)));
		double db_cutoff = 20.0 * std::log10(std::abs(filter.cascade().response(fc)));
		double db_2x     = 20.0 * std::log10(std::max(std::abs(filter.cascade().response(2 * fc)), 1e-10));
		double db_5x     = 20.0 * std::log10(std::max(std::abs(filter.cascade().response(5 * fc)), 1e-10));
		double db_10x    = 20.0 * std::log10(std::max(std::abs(filter.cascade().response(std::min(10 * fc, 0.499))), 1e-10));

		std::cout << std::setw(8) << order
		          << std::setw(12) << std::setprecision(2) << std::fixed << db_dc
		          << std::setw(14) << db_cutoff
		          << std::setw(14) << db_2x
		          << std::setw(14) << db_5x
		          << std::setw(14) << db_10x << "\n";
	}

	// ================================================================
	// 4. Phase response
	// ================================================================
	print_separator("Phase Response (4th order)");

	{
		iir::ButterworthLowPass<4> filter;
		filter.setup(4, sample_rate, cutoff_freq);

		viz::PlotConfig cfg;
		cfg.width = 72;
		cfg.height = 15;
		cfg.title = "4th Order Butterworth LP - Phase Response";
		cfg.x_label = "0 Hz -> " + std::to_string(static_cast<int>(sample_rate / 2)) + " Hz";
		viz::plot_phase_response(std::cout, filter.cascade(), sample_rate, cfg);
	}

	// ================================================================
	// 5. Pole-zero plot
	// ================================================================
	print_separator("Pole-Zero Plot (4th order)");

	{
		iir::ButterworthLowPass<4> filter;
		filter.setup(4, sample_rate, cutoff_freq);
		viz::plot_pole_zero(std::cout, filter.cascade());
	}

	// ================================================================
	// 6. Impulse response
	// ================================================================
	print_separator("Impulse Response (4th order)");

	{
		SimpleFilter<iir::ButterworthLowPass<4>> filter;
		filter.setup(4, sample_rate, cutoff_freq);

		constexpr int N = 200;
		std::vector<double> h(N);
		for (int n = 0; n < N; ++n) {
			h[n] = filter.process((n == 0) ? 1.0 : 0.0);
		}

		viz::PlotConfig cfg;
		cfg.width = 72;
		cfg.height = 15;
		cfg.title = "Impulse Response (first 200 samples)";
		cfg.x_label = "Sample index";
		viz::plot_line(std::cout, std::span<const double>(h), cfg);
	}

	// ================================================================
	// 7. Signal filtering demonstration
	// ================================================================
	print_separator("Signal Filtering: 500 Hz + 5000 Hz -> LP at 2000 Hz");

	{
		constexpr int N = 512;

		// Generate a signal with one tone below cutoff and one above
		auto tone_low  = sine<double>(N, 500.0, sample_rate);
		auto tone_high = sine<double>(N, 5000.0, sample_rate);

		std::vector<double> input(N);
		for (int n = 0; n < N; ++n) {
			input[n] = 0.5 * static_cast<double>(tone_low[n])
			         + 0.5 * static_cast<double>(tone_high[n]);
		}

		// Filter
		SimpleFilter<iir::ButterworthLowPass<4>> filter;
		filter.setup(4, sample_rate, cutoff_freq);

		std::vector<double> output(N);
		for (int n = 0; n < N; ++n) {
			output[n] = filter.process(input[n]);
		}

		// Plot first 200 samples of input and output
		constexpr int show = 200;

		{
			viz::PlotConfig cfg;
			cfg.width = 72;
			cfg.height = 10;
			cfg.title = "Input: 500 Hz + 5000 Hz";
			cfg.y_min = -1.1;
			cfg.y_max = 1.1;
			std::span<const double> s(input.data(), show);
			viz::plot_line(std::cout, s, cfg);
		}

		std::cout << "\n";

		{
			viz::PlotConfig cfg;
			cfg.width = 72;
			cfg.height = 10;
			cfg.title = "Output: after 4th order Butterworth LP at 2 kHz";
			cfg.y_min = -1.1;
			cfg.y_max = 1.1;
			std::span<const double> s(output.data(), show);
			viz::plot_line(std::cout, s, cfg);
		}
	}

	std::cout << "\n";
	return 0;
}
