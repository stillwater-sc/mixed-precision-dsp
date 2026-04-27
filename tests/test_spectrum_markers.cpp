// test_spectrum_markers.cpp: tests for the spectrum-analyzer marker /
// peak-find utilities (find_peaks, harmonic_markers, make_delta_marker).
//
// Coverage:
//   - find_peaks: synthetic 3-tone trace returns the 3 expected peaks
//     in descending amplitude order
//   - find_peaks: min_separation_bins suppresses adjacent local-max
//     chatter (broad peak across multiple bins counts once)
//   - find_peaks: sub-bin parabolic interpolation accurate to within
//     bin_step / 4 on a synthetic peak with known sub-bin offset
//   - find_peaks edge cases: empty trace, top_n=0, single-sample trace,
//     peak at first bin, peak at last bin, all-equal trace
//   - harmonic_markers: returns markers at k * fundamental for k=2..N,
//     bin_index = round(k*f / step), amplitude = trace[that bin]
//   - harmonic_markers: out-of-range harmonics silently truncated
//   - make_delta_marker: b minus a in frequency and amplitude
//   - Validation: invalid bin_freq_step_hz / fundamental_hz throw
//   - Mixed-precision sanity: float trace gives reasonable peaks
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)`.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <sw/dsp/spectrum/markers.hpp>

using namespace sw::dsp::spectrum;

#define REQUIRE(cond) \
	do { if (!(cond)) throw std::runtime_error( \
		std::string("test failed: ") + #cond + \
		" at " __FILE__ ":" + std::to_string(__LINE__)); } while (0)

static bool approx(double a, double b, double tol) {
	return std::abs(a - b) <= tol;
}

// ============================================================================
// find_peaks: synthetic 3-tone trace
// ============================================================================

void test_find_peaks_three_tones() {
	// 64-bin trace with three peaks at bins 10, 25, 50 with amplitudes
	// 30, 20, 25 dB. Background noise floor at -100 dB.
	std::array<double, 64> trace;
	trace.fill(-100.0);
	trace[10] = 30.0;
	trace[25] = 20.0;
	trace[50] = 25.0;
	const double bin_step = 1000.0;   // 1 kHz/bin

	auto peaks = find_peaks(std::span<const double>{trace}, bin_step, /*top_n=*/3);
	REQUIRE(peaks.size() == 3);
	// Returned in descending amplitude order: 30, 25, 20.
	REQUIRE(peaks[0].bin_index == 10);
	REQUIRE(peaks[1].bin_index == 50);
	REQUIRE(peaks[2].bin_index == 25);
	REQUIRE(approx(peaks[0].amplitude, 30.0, 1e-12));
	REQUIRE(approx(peaks[1].amplitude, 25.0, 1e-12));
	REQUIRE(approx(peaks[2].amplitude, 20.0, 1e-12));
	// Frequency = bin_index * step (no sub-bin interp because the
	// neighbors are at the noise floor — parabolic vertex collapses to
	// the integer bin).
	REQUIRE(approx(peaks[0].frequency_hz, 10.0 * bin_step, 1e-9));
	std::cout << "  find_peaks_three_tones: passed\n";
}

void test_find_peaks_top_n_smaller_than_available() {
	// Same trace, ask for only 2 peaks — should get the strongest two.
	std::array<double, 64> trace;
	trace.fill(-100.0);
	trace[10] = 30.0;
	trace[25] = 20.0;
	trace[50] = 25.0;
	auto peaks = find_peaks(std::span<const double>{trace}, 1000.0, /*top_n=*/2);
	REQUIRE(peaks.size() == 2);
	REQUIRE(peaks[0].bin_index == 10);
	REQUIRE(peaks[1].bin_index == 50);
	std::cout << "  find_peaks_top_n_smaller_than_available: passed\n";
}

// ============================================================================
// find_peaks: min_separation suppresses adjacent local-max chatter
// ============================================================================

void test_find_peaks_min_separation() {
	// Two distinct local maxima within min_separation_bins of each
	// other. is_local_max requires strict > on both neighbors, so the
	// previous version of this test (which had only one local max in
	// the 10..12 region) didn't actually exercise the suppression
	// logic. New layout:
	//
	//   bin 9  = 4 (local max: > -50 floor on left, > 0 on right)
	//   bin 10 = 0 (valley, not a peak)
	//   bin 11 = 5 (local max AND stronger than bin 9)
	//   bin 12 = 0 (valley)
	//
	// Both 9 and 11 are local maxes. Greedy selects bin 11 first
	// (stronger amplitude); bin 9 is then 2 bins away — within the
	// min_separation_bins=3 window — and gets suppressed.
	std::array<double, 32> trace;
	trace.fill(-50.0);
	trace[9]  = 4.0;
	trace[10] = 0.0;
	trace[11] = 5.0;
	trace[12] = 0.0;
	// Well-separated second peak.
	trace[25] = 3.0;

	auto peaks = find_peaks(std::span<const double>{trace}, 100.0,
	                         /*top_n=*/4, /*min_separation_bins=*/3);
	// Result: bin 11 (strongest, picked first) suppresses bin 9; bin 25
	// is well clear and also returned. Total = 2 markers.
	REQUIRE(peaks.size() == 2);
	REQUIRE(peaks[0].bin_index == 11);
	REQUIRE(peaks[1].bin_index == 25);
	std::cout << "  find_peaks_min_separation: passed\n";
}

// ============================================================================
// find_peaks: sub-bin parabolic interpolation
// ============================================================================

void test_find_peaks_sub_bin_interpolation() {
	// A symmetric parabolic peak with vertex between bins. y[i-1]=8,
	// y[i]=10, y[i+1]=9. Vertex offset:
	//     delta = 0.5 * (8 - 9) / (8 - 20 + 9) = 0.5 * -1 / -3 = +0.1667
	// So the recovered freq should be (i + 0.1667) * bin_step.
	std::array<double, 32> trace;
	trace.fill(-50.0);
	const std::size_t i = 15;
	trace[i - 1] = 8.0;
	trace[i]     = 10.0;
	trace[i + 1] = 9.0;
	const double bin_step = 1000.0;

	auto peaks = find_peaks(std::span<const double>{trace}, bin_step, /*top_n=*/1);
	REQUIRE(peaks.size() == 1);
	REQUIRE(peaks[0].bin_index == i);
	const double expected_freq = (static_cast<double>(i) + 1.0/6.0) * bin_step;
	REQUIRE(approx(peaks[0].frequency_hz, expected_freq, bin_step / 4.0));
	std::cout << "  find_peaks_sub_bin_interpolation: passed (freq="
	          << peaks[0].frequency_hz << " expected " << expected_freq << ")\n";
}

// ============================================================================
// find_peaks: edge cases
// ============================================================================

void test_find_peaks_edge_cases() {
	// Empty trace -> empty result.
	std::span<const double> empty{};
	REQUIRE(find_peaks(empty, 1000.0, 5).empty());

	// top_n = 0 -> empty result.
	std::array<double, 5> trace = {1.0, 2.0, 3.0, 2.0, 1.0};
	REQUIRE(find_peaks(std::span<const double>{trace}, 1000.0, 0).empty());

	// Single-sample trace -> the one bin is a peak.
	std::array<double, 1> single = {7.5};
	auto p1 = find_peaks(std::span<const double>{single}, 1000.0, 5);
	REQUIRE(p1.size() == 1);
	REQUIRE(p1[0].bin_index == 0);
	REQUIRE(approx(p1[0].amplitude, 7.5, 1e-12));

	// All-equal trace -> bin 0 is technically a peak (>= neighbor)?
	// Our is_local_max uses strict >. With equal neighbors, no bin is
	// strictly greater than its neighbor, so no peaks.
	std::array<double, 4> flat;
	flat.fill(2.0);
	REQUIRE(find_peaks(std::span<const double>{flat}, 1000.0, 5).empty());

	// Peak at first bin (must be > neighbor).
	std::array<double, 5> first_peak = {10.0, 5.0, 0.0, 0.0, 0.0};
	auto pf = find_peaks(std::span<const double>{first_peak}, 1000.0, 1);
	REQUIRE(pf.size() == 1);
	REQUIRE(pf[0].bin_index == 0);
	// Edge bin: no parabolic interp -> frequency_hz == 0 * step.
	REQUIRE(approx(pf[0].frequency_hz, 0.0, 1e-12));

	// Peak at last bin.
	std::array<double, 5> last_peak = {0.0, 0.0, 0.0, 5.0, 10.0};
	auto pl = find_peaks(std::span<const double>{last_peak}, 1000.0, 1);
	REQUIRE(pl.size() == 1);
	REQUIRE(pl[0].bin_index == 4);
	REQUIRE(approx(pl[0].frequency_hz, 4.0 * 1000.0, 1e-12));

	std::cout << "  find_peaks_edge_cases: passed\n";
}

// ============================================================================
// harmonic_markers
// ============================================================================

void test_harmonic_markers_basic() {
	// 100-bin trace, 1 kHz/bin, fundamental at 5 kHz (bin 5).
	// Harmonics at 10/15/20 kHz -> bins 10/15/20.
	std::array<double, 100> trace;
	for (std::size_t i = 0; i < trace.size(); ++i)
		trace[i] = static_cast<double>(i);   // synthetic ramp for distinct values
	const double step = 1000.0;
	const double f0 = 5000.0;

	auto h = harmonic_markers(std::span<const double>{trace}, step, f0, /*harmonics=*/3);
	REQUIRE(h.size() == 3);
	REQUIRE(h[0].bin_index == 10);   // 2nd harmonic
	REQUIRE(h[1].bin_index == 15);   // 3rd
	REQUIRE(h[2].bin_index == 20);   // 4th
	REQUIRE(approx(h[0].amplitude, 10.0, 1e-12));
	REQUIRE(approx(h[1].amplitude, 15.0, 1e-12));
	REQUIRE(approx(h[2].amplitude, 20.0, 1e-12));
	REQUIRE(approx(h[0].frequency_hz, 10000.0, 1e-9));
	std::cout << "  harmonic_markers_basic: passed\n";
}

void test_harmonic_markers_out_of_range_truncates() {
	// Fundamental at 30 kHz, trace is 100 bins at 1 kHz/bin = 100 kHz max.
	// 2nd harmonic at 60 kHz (bin 60) -> in range.
	// 3rd at 90 kHz (bin 90) -> in range.
	// 4th at 120 kHz -> out of range; truncated.
	// 5th at 150 kHz -> out of range.
	std::array<double, 100> trace;
	trace.fill(-100.0);
	trace[60] = 5.0;
	trace[90] = 3.0;
	auto h = harmonic_markers(std::span<const double>{trace}, 1000.0, 30000.0, 4);
	REQUIRE(h.size() == 2);
	REQUIRE(h[0].bin_index == 60);
	REQUIRE(h[1].bin_index == 90);
	std::cout << "  harmonic_markers_out_of_range_truncates: passed\n";
}

void test_harmonic_markers_rounding() {
	// Fundamental that doesn't sit exactly on a bin: 5333 Hz, 1 kHz step.
	// 2nd harmonic at 10666 Hz -> round to bin 11.
	// 3rd at 15999 Hz -> round to bin 16.
	std::array<double, 100> trace;
	for (std::size_t i = 0; i < trace.size(); ++i) trace[i] = static_cast<double>(i);
	auto h = harmonic_markers(std::span<const double>{trace}, 1000.0, 5333.0, 2);
	REQUIRE(h.size() == 2);
	REQUIRE(h[0].bin_index == 11);
	REQUIRE(h[1].bin_index == 16);
	std::cout << "  harmonic_markers_rounding: passed\n";
}

void test_harmonic_markers_overflow_safe() {
	// Pathological inputs that would, without the pre-cast range guard,
	// invoke implementation-defined behavior when casting an enormous
	// double to std::size_t. fundamental_hz near double's upper edge
	// drives target_bin_d to +inf at k=2; the loop must break cleanly
	// without a wrapped or trap-representation cast.
	std::array<double, 8> trace;
	trace.fill(0.0);
	auto h_inf = harmonic_markers(std::span<const double>{trace},
	                               /*step=*/1.0,
	                               /*fundamental=*/1e308,
	                               /*harmonics=*/5);
	REQUIRE(h_inf.empty());

	// Huge but finite k. With harmonics = 1000 and a fundamental that
	// puts the 2nd harmonic well past trace.size(), the loop should
	// break on the very first iteration. No crash, no garbage markers.
	auto h_big = harmonic_markers(std::span<const double>{trace},
	                               /*step=*/1.0,
	                               /*fundamental=*/100.0,    // 2nd at bin 200
	                               /*harmonics=*/1000);
	REQUIRE(h_big.empty());
	std::cout << "  harmonic_markers_overflow_safe: passed\n";
}

// ============================================================================
// make_delta_marker
// ============================================================================

void test_make_delta_marker() {
	Marker a; a.bin_index = 10; a.frequency_hz = 100.0; a.amplitude = -10.0;
	Marker b; b.bin_index = 50; b.frequency_hz = 500.0; b.amplitude = -25.0;
	auto d = make_delta_marker(a, b);
	REQUIRE(approx(d.delta_freq_hz, 400.0, 1e-12));      // 500 - 100
	REQUIRE(approx(d.delta_amplitude, -15.0, 1e-12));    // -25 - (-10)
	REQUIRE(d.a.bin_index == 10);
	REQUIRE(d.b.bin_index == 50);
	std::cout << "  make_delta_marker: passed\n";
}

// ============================================================================
// Validation
// ============================================================================

void test_validation() {
	std::array<double, 4> trace = {1.0, 2.0, 3.0, 0.0};
	bool t1 = false, t2 = false, t3 = false, t4 = false;

	try { (void)find_peaks(std::span<const double>{trace}, 0.0, 1); }
	catch (const std::invalid_argument&) { t1 = true; }
	REQUIRE(t1);

	try { (void)find_peaks(std::span<const double>{trace}, -100.0, 1); }
	catch (const std::invalid_argument&) { t2 = true; }
	REQUIRE(t2);

	try { (void)harmonic_markers(std::span<const double>{trace}, 0.0, 1000.0, 3); }
	catch (const std::invalid_argument&) { t3 = true; }
	REQUIRE(t3);

	try { (void)harmonic_markers(std::span<const double>{trace}, 1000.0, 0.0, 3); }
	catch (const std::invalid_argument&) { t4 = true; }
	REQUIRE(t4);

	std::cout << "  validation: passed\n";
}

// ============================================================================
// Mixed-precision sanity: float trace
// ============================================================================

void test_float_trace() {
	std::array<float, 32> trace;
	trace.fill(-100.0f);
	trace[8] = 5.0f;
	trace[20] = 8.0f;
	auto peaks = find_peaks(std::span<const float>{trace}, 100.0, 2);
	REQUIRE(peaks.size() == 2);
	REQUIRE(peaks[0].bin_index == 20);   // higher amplitude wins
	REQUIRE(peaks[1].bin_index == 8);
	REQUIRE(approx(peaks[0].amplitude, 8.0, 1e-6));
	std::cout << "  float_trace: passed\n";
}

// ============================================================================
// main
// ============================================================================

int main() {
	try {
		std::cout << "test_spectrum_markers\n";

		test_find_peaks_three_tones();
		test_find_peaks_top_n_smaller_than_available();
		test_find_peaks_min_separation();
		test_find_peaks_sub_bin_interpolation();
		test_find_peaks_edge_cases();

		test_harmonic_markers_basic();
		test_harmonic_markers_out_of_range_truncates();
		test_harmonic_markers_rounding();
		test_harmonic_markers_overflow_safe();

		test_make_delta_marker();
		test_validation();
		test_float_trace();

		std::cout << "all tests passed\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "FAILED: " << ex.what() << "\n";
		return 1;
	}
}
