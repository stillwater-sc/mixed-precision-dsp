// test_spectrum_front_end_corrector.cpp: tests that the spectrum/
// FrontEndCorrector alias correctly forwards to instrument::
// EqualizerFilter.
//
// The alias is a single line of code; the comprehensive testing of
// the underlying equalizer is in test_instrument_calibration. This
// file is a deliberate-minimum sanity check:
//   - Type identity: spectrum::FrontEndCorrector<...> resolves to
//     the same type as instrument::EqualizerFilter<...>.
//   - CalibrationProfile alias resolves to the underlying type.
//   - A FrontEndCorrector instance produces the same streaming
//     output as a directly-constructed EqualizerFilter on the same
//     profile + input.
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)`.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <sw/dsp/spectrum/front_end_corrector.hpp>

#define REQUIRE(cond) \
	do { if (!(cond)) throw std::runtime_error( \
		std::string("test failed: ") + #cond + \
		" at " __FILE__ ":" + std::to_string(__LINE__)); } while (0)

// ============================================================================
// Type identity
// ============================================================================

void test_type_identity() {
	using FE = sw::dsp::spectrum::FrontEndCorrector<>;
	using EQ = sw::dsp::instrument::EqualizerFilter<>;
	static_assert(std::is_same_v<FE, EQ>,
		"FrontEndCorrector must alias EqualizerFilter exactly");

	using FEf = sw::dsp::spectrum::FrontEndCorrector<float>;
	using EQf = sw::dsp::instrument::EqualizerFilter<float>;
	static_assert(std::is_same_v<FEf, EQf>,
		"FrontEndCorrector<float> must alias EqualizerFilter<float>");

	using PF = sw::dsp::spectrum::CalibrationProfile;
	using IP = sw::dsp::instrument::CalibrationProfile;
	static_assert(std::is_same_v<PF, IP>,
		"spectrum::CalibrationProfile must alias instrument::CalibrationProfile");

	std::cout << "  type_identity: passed\n";
}

// ============================================================================
// Behavioral equivalence: alias produces identical output
// ============================================================================

void test_behavioral_equivalence() {
	using namespace sw::dsp;

	// Synthetic mild-rolloff profile, same shape used by the scope demo.
	std::vector<double> freqs    = {0.0,   50e6, 100e6, 250e6, 500e6};
	std::vector<double> gains_dB = {0.0,  -0.5,  -2.0,  -3.0,  -3.0};
	std::vector<double> phases   = {0.0,  -0.10, -0.20, -0.30, -0.30};
	instrument::CalibrationProfile profile(freqs, gains_dB, phases);

	const double      fs       = 1.0e9;
	const std::size_t num_taps = 31;

	// Two instantiations on the same profile, same fs, same num_taps:
	// one via the alias, one directly. They should produce
	// bit-identical streaming output for the same input.
	spectrum::FrontEndCorrector<double> via_alias(profile, num_taps, fs);
	instrument::EqualizerFilter<double>  direct(profile, num_taps, fs);

	std::vector<double> input(256);
	for (std::size_t n = 0; n < input.size(); ++n)
		input[n] = static_cast<double>(n % 13) - 6.0;   // arbitrary repeating pattern

	for (std::size_t n = 0; n < input.size(); ++n) {
		const double y_alias  = via_alias.process(input[n]);
		const double y_direct = direct.process(input[n]);
		if (y_alias != y_direct)
			throw std::runtime_error(
				"alias output diverged from direct at sample "
				+ std::to_string(n) + ": alias=" + std::to_string(y_alias)
				+ " direct=" + std::to_string(y_direct));
	}
	std::cout << "  behavioral_equivalence: passed (256 samples bit-identical)\n";
}

// ============================================================================
// main
// ============================================================================

int main() {
	try {
		std::cout << "test_spectrum_front_end_corrector\n";

		test_type_identity();
		test_behavioral_equivalence();

		std::cout << "all tests passed\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "FAILED: " << ex.what() << "\n";
		return 1;
	}
}
