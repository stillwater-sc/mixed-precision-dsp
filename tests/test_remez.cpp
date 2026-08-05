// test_remez.cpp: Parks-McClellan (Remez exchange) equiripple FIR design tests
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/filter/fir/remez.hpp>
#include <sw/dsp/filter/fir/fir_filter.hpp>
#include <sw/dsp/math/constants.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sw::dsp;

constexpr double tolerance = 1e-4;

bool near(double a, double b, double eps = tolerance) {
	return std::abs(a - b) < eps;
}

void check(bool condition, const std::string& msg) {
	if (!condition) throw std::runtime_error("test failed: " + msg);
}

// Test 1: Basic equiripple lowpass design — verify filter produces taps
// and has the right number of them
void test_basic_lowpass() {
	std::size_t N = 31;
	std::vector<double> bands    = {0.0, 0.2, 0.3, 0.5};
	std::vector<double> desired  = {1.0, 1.0, 0.0, 0.0};
	std::vector<double> weights  = {1.0, 1.0};

	auto taps = remez<double>(N, bands, desired, weights);

	check(taps.size() == N, "tap count is " + std::to_string(taps.size()) + ", expected " + std::to_string(N));

	// All taps should be finite
	for (std::size_t i = 0; i < N; ++i) {
		check(std::isfinite(taps[i]),
		      "tap[" + std::to_string(i) + "] = " + std::to_string(taps[i]) + " is not finite");
	}

	std::cout << "  basic_lowpass: passed (N=" << N << ")\n";
}

// Test 2: Symmetric impulse response (linear phase)
// Type I (odd taps) should have h[n] = h[N-1-n]
void test_linear_phase_symmetry() {
	std::size_t N = 31;
	std::vector<double> bands    = {0.0, 0.2, 0.3, 0.5};
	std::vector<double> desired  = {1.0, 1.0, 0.0, 0.0};
	std::vector<double> weights  = {1.0, 1.0};

	auto taps = remez<double>(N, bands, desired, weights);

	for (std::size_t i = 0; i < N / 2; ++i) {
		check(near(taps[i], taps[N - 1 - i], 1e-10),
		      "symmetry: h[" + std::to_string(i) + "]=" + std::to_string(taps[i]) +
		      " != h[" + std::to_string(N-1-i) + "]=" + std::to_string(taps[N-1-i]));
	}

	std::cout << "  linear_phase_symmetry: passed\n";
}

// Test 3: DC gain should be near 1.0 for a lowpass
void test_dc_gain() {
	std::size_t N = 31;
	std::vector<double> bands    = {0.0, 0.2, 0.3, 0.5};
	std::vector<double> desired  = {1.0, 1.0, 0.0, 0.0};
	std::vector<double> weights  = {1.0, 1.0};

	auto taps = remez<double>(N, bands, desired, weights);

	double dc_sum = 0.0;
	for (std::size_t i = 0; i < N; ++i)
		dc_sum += taps[i];

	double dc_db = 20.0 * std::log10(std::abs(dc_sum));

	check(near(dc_db, 0.0, 1.0),
	      "DC gain = " + std::to_string(dc_db) + " dB, expected near 0 dB");

	std::cout << "  dc_gain: passed (" << dc_db << " dB)\n";
}

// Test 4: Stopband rejection — response should be small above stopband edge
void test_stopband_rejection() {
	std::size_t N = 31;
	std::vector<double> bands    = {0.0, 0.2, 0.3, 0.5};
	std::vector<double> desired  = {1.0, 1.0, 0.0, 0.0};
	std::vector<double> weights  = {1.0, 1.0};

	auto taps = remez<double>(N, bands, desired, weights);

	// Evaluate frequency response at several stopband frequencies
	for (double f = 0.35; f <= 0.49; f += 0.05) {
		double re = 0.0, im = 0.0;
		for (std::size_t n = 0; n < N; ++n) {
			double w = two_pi * f * static_cast<double>(n);
			re += taps[n] * std::cos(w);
			im -= taps[n] * std::sin(w);
		}
		double mag = std::sqrt(re * re + im * im);
		double db = 20.0 * std::log10(mag + 1e-30);

		check(db < -10.0,
		      "stopband at f=" + std::to_string(f) + ": " + std::to_string(db) +
		      " dB (expected < -10 dB)");
	}

	std::cout << "  stopband_rejection: passed\n";
}

// Test 5: Passband flatness — response should be near 1.0 in passband
void test_passband_flatness() {
	std::size_t N = 31;
	std::vector<double> bands    = {0.0, 0.2, 0.3, 0.5};
	std::vector<double> desired  = {1.0, 1.0, 0.0, 0.0};
	std::vector<double> weights  = {1.0, 1.0};

	auto taps = remez<double>(N, bands, desired, weights);

	for (double f = 0.01; f <= 0.19; f += 0.03) {
		double re = 0.0, im = 0.0;
		for (std::size_t n = 0; n < N; ++n) {
			double w = two_pi * f * static_cast<double>(n);
			re += taps[n] * std::cos(w);
			im -= taps[n] * std::sin(w);
		}
		double mag = std::sqrt(re * re + im * im);
		double db = 20.0 * std::log10(mag);

		check(std::abs(db) < 3.0,
		      "passband at f=" + std::to_string(f) + ": " + std::to_string(db) +
		      " dB (expected within 3 dB of 0)");
	}

	std::cout << "  passband_flatness: passed\n";
}

// Test 6: Even tap count (Type II)
void test_even_taps() {
	std::size_t N = 32;
	std::vector<double> bands    = {0.0, 0.2, 0.3, 0.5};
	std::vector<double> desired  = {1.0, 1.0, 0.0, 0.0};
	std::vector<double> weights  = {1.0, 1.0};

	auto taps = remez<double>(N, bands, desired, weights);
	check(taps.size() == N, "even tap count");

	// Should still have symmetry
	for (std::size_t i = 0; i < N / 2; ++i) {
		check(near(taps[i], taps[N - 1 - i], 1e-10),
		      "even symmetry at " + std::to_string(i));
	}

	// DC gain should be near 1.0
	double dc = 0.0;
	for (std::size_t i = 0; i < N; ++i) dc += taps[i];
	check(std::abs(dc - 1.0) < 0.5, "even DC gain = " + std::to_string(dc));

	std::cout << "  even_taps: passed (N=" << N << ")\n";
}

// Test 7: Convenience wrapper — equiripple lowpass
void test_convenience_lowpass() {
	auto taps = design_fir_equiripple_lowpass<double>(31, 0.2, 0.3);
	check(taps.size() == 31, "convenience lowpass tap count");

	double dc = 0.0;
	for (std::size_t i = 0; i < taps.size(); ++i) dc += taps[i];
	check(std::abs(dc - 1.0) < 0.5, "convenience lowpass DC gain = " + std::to_string(dc));

	std::cout << "  convenience_lowpass: passed\n";
}

// Test 8: Convenience wrapper — equiripple bandpass
void test_convenience_bandpass() {
	auto taps = design_fir_equiripple_bandpass<double>(51, 0.1, 0.2, 0.3, 0.4);
	check(taps.size() == 51, "convenience bandpass tap count");

	// DC gain should be near 0 (bandpass)
	double dc = 0.0;
	for (std::size_t i = 0; i < taps.size(); ++i) dc += taps[i];
	check(std::abs(dc) < 0.3, "convenience bandpass DC gain = " + std::to_string(dc));

	std::cout << "  convenience_bandpass: passed\n";
}

// Test 9: Taps are usable with FIRFilter
void test_fir_integration() {
	auto taps = design_fir_equiripple_lowpass<double>(31, 0.2, 0.3);

	FIRFilter<double> f(taps);
	check(f.num_taps() == 31, "FIRFilter tap count");

	// Process impulse
	double y0 = f.process(1.0);
	check(std::isfinite(y0), "FIR impulse response finite");

	for (int i = 0; i < 50; ++i) {
		double y = f.process(0.0);
		check(std::isfinite(y), "FIR zero-input response finite");
	}

	std::cout << "  fir_integration: passed\n";
}

// Test 10: Input validation
void test_validation() {
	bool caught = false;

	// Too few taps
	try {
		remez<double>(2, {0.0, 0.2, 0.3, 0.5}, {1.0, 1.0, 0.0, 0.0}, {1.0, 1.0});
	} catch (const std::invalid_argument&) {
		caught = true;
	}
	check(caught, "num_taps < 3 should throw");

	// Odd number of band edges
	caught = false;
	try {
		remez<double>(31, {0.0, 0.2, 0.3}, {1.0, 1.0, 0.0}, {1.0});
	} catch (const std::invalid_argument&) {
		caught = true;
	}
	check(caught, "odd band edges should throw");

	// Mismatched desired/bands
	caught = false;
	try {
		remez<double>(31, {0.0, 0.2, 0.3, 0.5}, {1.0, 1.0}, {1.0, 1.0});
	} catch (const std::invalid_argument&) {
		caught = true;
	}
	check(caught, "mismatched desired size should throw");

	std::cout << "  validation: passed\n";
}

// Test 11: Hilbert transformer — antisymmetric taps, DC gain = 0
void test_hilbert() {
	std::size_t N = 31;
	std::vector<double> bands    = {0.05, 0.45};
	std::vector<double> desired  = {1.0, 1.0};
	std::vector<double> weights  = {1.0};

	auto taps = remez<double>(N, bands, desired, weights, RemezBandType::hilbert);
	check(taps.size() == N, "hilbert tap count");

	// Antisymmetric: h[n] = -h[N-1-n]
	std::size_t L = (N - 1) / 2;
	for (std::size_t i = 0; i < L; ++i) {
		check(near(taps[i], -taps[N - 1 - i], 1e-8),
		      "hilbert antisymmetry at " + std::to_string(i));
	}
	// Center tap should be 0 for odd-length Type III
	check(near(taps[L], 0.0, 1e-10), "hilbert center tap = " + std::to_string(taps[L]));

	// DC gain should be 0 (antisymmetric taps cancel)
	double dc = 0.0;
	for (std::size_t i = 0; i < N; ++i) dc += taps[i];
	check(near(dc, 0.0, 1e-8), "hilbert DC gain = " + std::to_string(dc));

	std::cout << "  hilbert: passed (antisymmetric, DC=" << dc << ")\n";
}

// Test 12: Differentiator — antisymmetric taps, DC gain = 0
void test_differentiator() {
	std::size_t N = 31;
	std::vector<double> bands    = {0.01, 0.45};
	std::vector<double> desired  = {0.01, 0.45};  // linear ramp
	std::vector<double> weights  = {1.0};

	auto taps = remez<double>(N, bands, desired, weights, RemezBandType::differentiator);
	check(taps.size() == N, "differentiator tap count");

	// Antisymmetric: h[n] = -h[N-1-n]
	std::size_t L = (N - 1) / 2;
	for (std::size_t i = 0; i < L; ++i) {
		check(near(taps[i], -taps[N - 1 - i], 1e-8),
		      "differentiator antisymmetry at " + std::to_string(i));
	}

	// DC gain should be 0
	double dc = 0.0;
	for (std::size_t i = 0; i < N; ++i) dc += taps[i];
	check(near(dc, 0.0, 1e-8), "differentiator DC gain = " + std::to_string(dc));

	std::cout << "  differentiator: passed (antisymmetric, DC=" << dc << ")\n";
}

// Test 13: Extended validation
void test_extended_validation() {
	bool caught = false;

	// Negative weight
	try {
		remez<double>(31, {0.0, 0.2, 0.3, 0.5}, {1.0, 1.0, 0.0, 0.0}, {-1.0, 1.0});
	} catch (const std::invalid_argument&) {
		caught = true;
	}
	check(caught, "negative weight should throw");

	// Band edge out of range
	caught = false;
	try {
		remez<double>(31, {0.0, 0.6, 0.7, 0.9}, {1.0, 1.0, 0.0, 0.0}, {1.0, 1.0});
	} catch (const std::invalid_argument&) {
		caught = true;
	}
	check(caught, "band edge > 0.5 should throw");

	// Non-monotonic bands
	caught = false;
	try {
		remez<double>(31, {0.3, 0.2, 0.1, 0.5}, {1.0, 1.0, 0.0, 0.0}, {1.0, 1.0});
	} catch (const std::invalid_argument&) {
		caught = true;
	}
	check(caught, "non-monotonic bands should throw");

	std::cout << "  extended_validation: passed\n";
}

// ---------------------------------------------------------------------------
// Issue #203 regression tests.
//
// The Remez designers returned filters that were not equiripple: a fixed
// ~2.4 dB passband ripple with 1.25 DC gain regardless of length, and
// stopband attenuation that plateaued near 20 dB and then collapsed as taps
// were added. The tests below pin the properties that were violated. The
// reference figures are Parks-McClellan values (cross-checked against
// scipy.signal.remez for the same specifications).
// ---------------------------------------------------------------------------

// Zero-phase amplitude response of a Type I (odd-length, symmetric) filter.
double type1_amplitude(const mtl::vec::dense_vector<double>& h, double f) {
	std::size_t L = (h.size() - 1) / 2;
	double v = h[L];
	for (std::size_t k = 1; k <= L; ++k)
		v += 2.0 * h[L + k] * std::cos(two_pi * f * static_cast<double>(k));
	return v;
}

// Worst |H| over [f0, f1].
double peak_magnitude(const mtl::vec::dense_vector<double>& h, double f0, double f1) {
	double worst = 0.0;
	for (int i = 0; i <= 2000; ++i)
		worst = std::max(worst, std::abs(type1_amplitude(h, f0 + (f1 - f0) * i / 2000.0)));
	return worst;
}

// DC gain must be 1, not 1.25, and the passband must actually ripple about
// 1 rather than peaking at DC. Ripple must shrink as taps are added.
void test_equiripple_passband() {
	const std::vector<double> bands   = {0.0, 0.20, 0.25, 0.5};
	const std::vector<double> desired = {1.0, 1.0, 0.0, 0.0};
	const std::vector<double> weights = {1.0, 1.0};

	double prev_ripple_db = 1e9;
	for (std::size_t N : {63u, 95u, 127u}) {
		auto h = remez<double>(N, bands, desired, weights);

		double lo = 1e30, hi = -1e30;
		for (int i = 0; i <= 1000; ++i) {
			double m = type1_amplitude(h, 0.20 * i / 1000.0);
			lo = std::min(lo, m);
			hi = std::max(hi, m);
		}
		double ripple_db = 20.0 * std::log10(hi / lo);

		// Was 1.2473 for every length; must now be unity to well under 1%.
		double dc = type1_amplitude(h, 0.0);
		check(std::abs(dc - 1.0) < 5e-3,
		      "N=" + std::to_string(N) + " DC gain " + std::to_string(dc) + ", expected ~1.0");

		// Was pinned at ~2.4 dB for every length.
		check(ripple_db < 0.05,
		      "N=" + std::to_string(N) + " passband ripple " + std::to_string(ripple_db) +
		      " dB, expected < 0.05");

		// The defining property: more taps must buy less ripple.
		check(ripple_db < prev_ripple_db,
		      "N=" + std::to_string(N) + " ripple " + std::to_string(ripple_db) +
		      " dB did not improve on " + std::to_string(prev_ripple_db));
		prev_ripple_db = ripple_db;

		// Equiripple means the passband ripple and the stopband floor are the
		// same deviation when the band weights are equal. Before the fix these
		// disagreed by ~7x even once the passband looked reasonable.
		double delta_p = (std::pow(10.0, ripple_db / 20.0) - 1.0) /
		                 (std::pow(10.0, ripple_db / 20.0) + 1.0);
		double delta_s = peak_magnitude(h, 0.25, 0.5);
		check(delta_s < delta_p * 1.5 && delta_p < delta_s * 1.5,
		      "N=" + std::to_string(N) + " passband deviation " + std::to_string(delta_p) +
		      " and stopband deviation " + std::to_string(delta_s) + " are not equal");
	}

	std::cout << "  equiripple_passband: passed\n";
}

// Attenuation must improve monotonically with length. It previously
// plateaued near 20 dB and then went negative (stopband above passband).
void test_stopband_grows_with_taps() {
	const std::vector<double> bands   = {0.0, 0.20, 0.25, 0.5};
	const std::vector<double> desired = {1.0, 1.0, 0.0, 0.0};
	const std::vector<double> weights = {1.0, 1.0};

	// Parks-McClellan reference attenuations for this specification.
	const struct { std::size_t taps; double min_db; } expect[] = {
		{ 63,  50.0},   // reference 56.5 dB
		{ 95,  74.0},   // reference 80.2 dB
		{127,  97.0},   // reference 103.9 dB
	};

	double prev_db = 0.0;
	for (const auto& e : expect) {
		auto h = remez<double>(e.taps, bands, desired, weights);
		double atten_db = -20.0 * std::log10(peak_magnitude(h, 0.25, 0.5));

		check(atten_db > e.min_db,
		      "N=" + std::to_string(e.taps) + " stopband " + std::to_string(atten_db) +
		      " dB, expected > " + std::to_string(e.min_db));
		check(atten_db > prev_db,
		      "N=" + std::to_string(e.taps) + " stopband " + std::to_string(atten_db) +
		      " dB did not improve on " + std::to_string(prev_db));
		prev_db = atten_db;
	}

	std::cout << "  stopband_grows_with_taps: passed (127 taps -> " +
	             std::to_string(prev_db) + " dB)\n";
}

// A bandpass with equal weights on both stopbands must reject equally on
// both sides. Before the fix the two stopbands differed by ~37 dB.
void test_bandpass_symmetric_rejection() {
	auto h = remez<double>(95, {0.0, 0.15, 0.20, 0.30, 0.35, 0.5},
	                            {0.0, 0.0, 1.0, 1.0, 0.0, 0.0},
	                            {1.0, 1.0, 1.0});

	double lower_db = -20.0 * std::log10(peak_magnitude(h, 0.0, 0.15));
	double upper_db = -20.0 * std::log10(peak_magnitude(h, 0.35, 0.5));

	check(lower_db > 70.0, "lower stopband " + std::to_string(lower_db) + " dB, expected > 70");
	check(upper_db > 70.0, "upper stopband " + std::to_string(upper_db) + " dB, expected > 70");
	check(std::abs(lower_db - upper_db) < 3.0,
	      "stopbands differ: " + std::to_string(lower_db) + " vs " + std::to_string(upper_db) + " dB");

	std::cout << "  bandpass_symmetric_rejection: passed (" +
	             std::to_string(lower_db) + " / " + std::to_string(upper_db) + " dB)\n";
}

// A specification the exchange cannot solve must be reported, not returned
// as a filter whose stopband sits above its passband.
void test_nonconvergence_throws() {
	bool caught = false;
	try {
		// 127 taps with a 0.20-wide transition about 0.25 needs >300 dB of
		// attenuation — unreachable in double precision. This previously
		// returned an all-zero filter.
		remez<double>(127, {0.0, 0.15, 0.35, 0.5}, {1.0, 1.0, 0.0, 0.0}, {1.0, 1.0});
	} catch (const std::runtime_error&) {
		caught = true;
	}
	check(caught, "unsolvable specification should throw");

	std::cout << "  nonconvergence_throws: passed\n";
}

int main() {
	try {
		std::cout << "Parks-McClellan (Remez) equiripple FIR design tests\n";

		test_basic_lowpass();
		test_linear_phase_symmetry();
		test_dc_gain();
		test_stopband_rejection();
		test_passband_flatness();
		test_even_taps();
		test_convenience_lowpass();
		test_convenience_bandpass();
		test_fir_integration();
		test_validation();
		test_hilbert();
		test_differentiator();
		test_extended_validation();
		test_equiripple_passband();
		test_stopband_grows_with_taps();
		test_bandpass_symmetric_rejection();
		test_nonconvergence_throws();

		std::cout << "All Remez tests passed.\n";
		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << "\n";
		return 1;
	}
}
