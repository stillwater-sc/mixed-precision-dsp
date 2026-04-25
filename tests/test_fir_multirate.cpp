// test_fir_multirate.cpp: tests for polyphase and overlap-add/save convolution
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/filter/fir/fir.hpp>
#include <sw/dsp/acquisition/polyphase_decimator.hpp>
#include <sw/dsp/signals/sampling.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <stdexcept>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

using namespace sw::dsp;

namespace {

double max_abs_diff(const mtl::vec::dense_vector<double>& a,
                    const mtl::vec::dense_vector<double>& b) {
	std::size_t n = std::min(a.size(), b.size());
	double m = 0.0;
	for (std::size_t i = 0; i < n; ++i) {
		double d = std::abs(a[i] - b[i]);
		if (d > m) m = d;
	}
	// Count any length mismatch as failure too.
	if (a.size() != b.size()) m = std::max(m, 1e30);
	return m;
}

mtl::vec::dense_vector<double> random_signal(std::size_t n, unsigned seed = 42) {
	std::mt19937 rng(seed);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);
	mtl::vec::dense_vector<double> x(n);
	for (std::size_t i = 0; i < n; ++i) x[i] = dist(rng);
	return x;
}

// Simple symmetric lowpass: windowed sinc-ish. Just a small hand-rolled kernel
// for deterministic testing — exact coefficients don't matter, only that the
// filter has length M.
mtl::vec::dense_vector<double> lowpass_taps(std::size_t M) {
	mtl::vec::dense_vector<double> h(M);
	double center = static_cast<double>(M - 1) / 2.0;
	double sum = 0.0;
	for (std::size_t i = 0; i < M; ++i) {
		double x = (static_cast<double>(i) - center) / 2.0;
		double w = std::exp(-x * x * 0.5);  // Gaussian-shaped
		h[i] = w;
		sum += w;
	}
	for (std::size_t i = 0; i < M; ++i) h[i] /= sum;  // normalise DC to 1
	return h;
}

// Direct linear convolution for reference.
mtl::vec::dense_vector<double>
direct_convolve(const mtl::vec::dense_vector<double>& x,
                const mtl::vec::dense_vector<double>& h) {
	std::size_t N = x.size(), M = h.size();
	mtl::vec::dense_vector<double> y(N + M - 1, 0.0);
	for (std::size_t n = 0; n < N; ++n) {
		for (std::size_t k = 0; k < M; ++k) {
			y[n + k] += x[n] * h[k];
		}
	}
	return y;
}

// Reference for polyphase interpolation: upsample (zero-insert) then FIR.
mtl::vec::dense_vector<double>
reference_interpolate(const mtl::vec::dense_vector<double>& x,
                      const mtl::vec::dense_vector<double>& h,
                      std::size_t L) {
	auto x_up = upsample(x, L);
	// Direct FIR on the full-rate signal, then truncate to the polyphase length.
	mtl::vec::dense_vector<double> y(x_up.size(), 0.0);
	FIRFilter<double> f(h);
	for (std::size_t i = 0; i < x_up.size(); ++i) y[i] = f.process(x_up[i]);
	return y;
}

// Reference for polyphase decimation: FIR then downsample by M.
mtl::vec::dense_vector<double>
reference_decimate(const mtl::vec::dense_vector<double>& x,
                   const mtl::vec::dense_vector<double>& h,
                   std::size_t M) {
	FIRFilter<double> f(h);
	mtl::vec::dense_vector<double> y_full(x.size(), 0.0);
	for (std::size_t i = 0; i < x.size(); ++i) y_full[i] = f.process(x[i]);
	return downsample(y_full, M);
}

} // anonymous namespace

void test_polyphase_interpolator_impulse() {
	// With x = [1, 0, 0, ...], polyphase interpolation by L with taps h
	// should produce y[n] = h[n] (just the filter impulse response at the
	// full rate). Covers the simplest possible sanity check.
	mtl::vec::dense_vector<double> h({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
	std::size_t L = 3;
	PolyphaseInterpolator<double> interp(h, L);

	mtl::vec::dense_vector<double> x({1.0, 0.0, 0.0});  // 3 input → 9 output
	auto y = interp.process_block(std::span<const double>(x.data(), x.size()));
	if (y.size() != x.size() * L)
		throw std::runtime_error("test failed: interpolator output length");
	for (std::size_t i = 0; i < h.size(); ++i) {
		if (std::abs(y[i] - h[i]) > 1e-12)
			throw std::runtime_error("test failed: interpolator impulse response");
	}
	for (std::size_t i = h.size(); i < y.size(); ++i) {
		if (std::abs(y[i]) > 1e-12)
			throw std::runtime_error("test failed: interpolator impulse tail nonzero");
	}
	std::cout << "  polyphase_interpolator_impulse: passed\n";
}

void test_polyphase_interpolator_equivalence() {
	// Polyphase interpolator must match upsample-then-filter for any input.
	for (std::size_t L : {2u, 3u, 4u, 5u}) {
		auto h = lowpass_taps(31);  // not a multiple of L for at least some L
		auto x = random_signal(200, 7);

		PolyphaseInterpolator<double> interp(h, L);
		auto y_poly = interp.process_block(std::span<const double>(x.data(), x.size()));
		auto y_ref  = reference_interpolate(x, h, L);

		double err = max_abs_diff(y_poly, y_ref);
		if (err > 1e-12) {
			throw std::runtime_error(
				"test failed: interpolator L=" + std::to_string(L)
				+ " max error " + std::to_string(err));
		}
	}
	std::cout << "  polyphase_interpolator_equivalence: passed\n";
}

void test_polyphase_decimator_equivalence() {
	// Polyphase decimator must match filter-then-downsample.
	for (std::size_t M : {2u, 3u, 4u, 5u}) {
		auto h = lowpass_taps(31);
		auto x = random_signal(300, 11);
		// Trim x to a multiple of M so reference and polyphase agree on length.
		std::size_t trimmed = (x.size() / M) * M;
		mtl::vec::dense_vector<double> x_t(trimmed);
		for (std::size_t i = 0; i < trimmed; ++i) x_t[i] = x[i];

		PolyphaseDecimator<double> dec(h, M);
		auto y_poly = dec.process_block(std::span<const double>(x_t.data(), x_t.size()));
		auto y_ref  = reference_decimate(x_t, h, M);

		double err = max_abs_diff(y_poly, y_ref);
		if (err > 1e-12) {
			throw std::runtime_error(
				"test failed: decimator M=" + std::to_string(M)
				+ " max error " + std::to_string(err));
		}
	}
	std::cout << "  polyphase_decimator_equivalence: passed\n";
}

void test_polyphase_reset() {
	// After reset, running the same input twice must produce the same output.
	auto h = lowpass_taps(21);
	auto x = random_signal(64, 23);

	PolyphaseInterpolator<double> interp(h, 4);
	auto y1 = interp.process_block(std::span<const double>(x.data(), x.size()));
	interp.reset();
	auto y2 = interp.process_block(std::span<const double>(x.data(), x.size()));
	if (max_abs_diff(y1, y2) > 1e-15)
		throw std::runtime_error("test failed: interpolator reset not idempotent");

	PolyphaseDecimator<double> dec(h, 4);
	auto z1 = dec.process_block(std::span<const double>(x.data(), x.size()));
	dec.reset();
	auto z2 = dec.process_block(std::span<const double>(x.data(), x.size()));
	if (max_abs_diff(z1, z2) > 1e-15)
		throw std::runtime_error("test failed: decimator reset not idempotent");

	std::cout << "  polyphase_reset: passed\n";
}

void test_overlap_add_equivalence() {
	// One-shot overlap-add must match direct convolution for various block sizes.
	auto h = lowpass_taps(25);
	auto x = random_signal(500, 13);
	auto y_ref = direct_convolve(x, h);

	for (std::size_t L : {8u, 16u, 32u, 64u, 100u}) {
		auto y_oa = overlap_add_convolve(x, h, L);
		double err = max_abs_diff(y_oa, y_ref);
		if (err > 1e-9) {
			throw std::runtime_error(
				"test failed: overlap-add L=" + std::to_string(L)
				+ " max error " + std::to_string(err));
		}
	}
	std::cout << "  overlap_add_equivalence: passed\n";
}

void test_overlap_save_equivalence() {
	auto h = lowpass_taps(25);
	auto x = random_signal(500, 17);
	auto y_ref = direct_convolve(x, h);

	for (std::size_t L : {8u, 16u, 32u, 64u, 100u}) {
		auto y_os = overlap_save_convolve(x, h, L);
		double err = max_abs_diff(y_os, y_ref);
		if (err > 1e-9) {
			throw std::runtime_error(
				"test failed: overlap-save L=" + std::to_string(L)
				+ " max error " + std::to_string(err));
		}
	}
	std::cout << "  overlap_save_equivalence: passed\n";
}

void test_overlap_add_default_block() {
	// Default block-size selection should still produce correct output.
	auto h = lowpass_taps(17);
	auto x = random_signal(300, 19);
	auto y_ref = direct_convolve(x, h);
	auto y_def = overlap_add_convolve(x, h);  // default block_size
	double err = max_abs_diff(y_def, y_ref);
	if (err > 1e-9)
		throw std::runtime_error(
			"test failed: overlap-add default block_size error " + std::to_string(err));
	std::cout << "  overlap_add_default_block: passed\n";
}

void test_overlap_save_default_block() {
	auto h = lowpass_taps(17);
	auto x = random_signal(300, 23);
	auto y_ref = direct_convolve(x, h);
	auto y_def = overlap_save_convolve(x, h);
	double err = max_abs_diff(y_def, y_ref);
	if (err > 1e-9)
		throw std::runtime_error(
			"test failed: overlap-save default block_size error " + std::to_string(err));
	std::cout << "  overlap_save_default_block: passed\n";
}

void test_overlap_add_streaming() {
	// Feeding the signal through a stateful OverlapAddConvolver one block at a
	// time (plus flush) must produce the same output as direct convolution.
	std::size_t L = 16;
	auto h = lowpass_taps(13);
	// Use a signal length that is a multiple of L for simplicity.
	auto x = random_signal(64, 29);
	auto y_ref = direct_convolve(x, h);

	OverlapAddConvolver<double> oa(h, L);
	mtl::vec::dense_vector<double> y(x.size() + h.size() - 1, 0.0);
	std::size_t out_idx = 0;
	for (std::size_t pos = 0; pos + L <= x.size(); pos += L) {
		auto blk = oa.process_block(std::span<const double>(x.data() + pos, L));
		for (std::size_t i = 0; i < L; ++i) y[out_idx++] = blk[i];
	}
	auto tail = oa.flush();
	for (std::size_t i = 0; i < tail.size() && out_idx < y.size(); ++i) {
		y[out_idx++] = tail[i];
	}
	double err = max_abs_diff(y, y_ref);
	if (err > 1e-9)
		throw std::runtime_error(
			"test failed: overlap-add streaming error " + std::to_string(err));
	std::cout << "  overlap_add_streaming: passed\n";
}

void test_overlap_convolver_state_persistence() {
	// Two successive process_block() calls on the stateful convolver must
	// produce the same result as one big call (modulo block boundaries).
	std::size_t L = 16;
	auto h = lowpass_taps(9);
	auto x = random_signal(L * 2, 31);

	OverlapSaveConvolver<double> os(h, L);
	auto out1 = os.process_block(std::span<const double>(x.data(), L));
	auto out2 = os.process_block(std::span<const double>(x.data() + L, L));

	// Combine out1 and out2, compare to direct convolution prefix.
	mtl::vec::dense_vector<double> combined(L * 2);
	for (std::size_t i = 0; i < L; ++i) combined[i] = out1[i];
	for (std::size_t i = 0; i < L; ++i) combined[L + i] = out2[i];
	auto y_ref = direct_convolve(x, h);

	// Overlap-save emits the first x.size() samples of the linear convolution.
	double m = 0.0;
	for (std::size_t i = 0; i < combined.size(); ++i) {
		double d = std::abs(combined[i] - y_ref[i]);
		if (d > m) m = d;
	}
	if (m > 1e-9)
		throw std::runtime_error(
			"test failed: overlap-save state persistence error " + std::to_string(m));
	std::cout << "  overlap_convolver_state_persistence: passed\n";
}

// ============================================================================
// Issue #91: Polyphase decimator with non-native CoeffScalar
//
// Acceptance criterion calls for unit tests at posit and cfloat. Each test
// designs taps in double, casts to T, runs the decimator at T, and compares
// against an at-T reference (FIR-then-downsample). Tolerance is set to the
// type's ULP scale; structurally these are identical computations, so any
// drift signals an actual bug rather than precision loss.
// ============================================================================

namespace {

template <typename T>
mtl::vec::dense_vector<T> cast_signal(const mtl::vec::dense_vector<double>& x) {
	mtl::vec::dense_vector<T> y(x.size());
	for (std::size_t i = 0; i < x.size(); ++i) y[i] = T(x[i]);
	return y;
}

template <typename T>
mtl::vec::dense_vector<T> reference_decimate_T(const mtl::vec::dense_vector<T>& x,
                                                 const mtl::vec::dense_vector<T>& h,
                                                 std::size_t M) {
	FIRFilter<T> f(h);
	mtl::vec::dense_vector<T> y_full(x.size(), T{});
	for (std::size_t i = 0; i < x.size(); ++i) y_full[i] = f.process(x[i]);
	return downsample(y_full, M);
}

template <typename T>
double max_diff_to_double(const mtl::vec::dense_vector<T>& a,
                          const mtl::vec::dense_vector<T>& b) {
	if (a.size() != b.size()) return 1e30;
	double m = 0.0;
	for (std::size_t i = 0; i < a.size(); ++i) {
		double d = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
		if (d > m) m = d;
	}
	return m;
}

} // anonymous namespace

void test_polyphase_decimator_in_posit() {
	using posit_t = sw::universal::posit<32, 2>;

	for (std::size_t M : {2u, 3u, 4u, 5u}) {
		auto h_d = lowpass_taps(31);
		auto x_d = random_signal(300, 11);
		std::size_t trimmed = (x_d.size() / M) * M;
		mtl::vec::dense_vector<double> x_t(trimmed);
		for (std::size_t i = 0; i < trimmed; ++i) x_t[i] = x_d[i];

		auto h_p = cast_signal<posit_t>(h_d);
		auto x_p = cast_signal<posit_t>(x_t);

		PolyphaseDecimator<posit_t> dec(h_p, M);
		auto y_poly = dec.process_block(std::span<const posit_t>(x_p.data(), x_p.size()));
		auto y_ref  = reference_decimate_T(x_p, h_p, M);

		double err = max_diff_to_double(y_poly, y_ref);
		// posit<32,2> has ~28 mantissa bits near unity. With 31 taps, accumulated
		// rounding stays within ~1e-7 — same as the structural double bound but
		// scaled to posit ULP.
		if (err > 1e-7)
			throw std::runtime_error("test failed: posit decimator M=" +
				std::to_string(M) + " max error " + std::to_string(err));
	}
	std::cout << "  polyphase_decimator_in_posit: passed\n";
}

void test_polyphase_decimator_in_cfloat() {
	// cfloat<32,8>: same width as IEEE float, useful as a sanity check that
	// the decimator works with cfloat's representation even at native widths.
	using cfloat32 = sw::universal::cfloat<32, 8, std::uint32_t, true, false, false>;

	for (std::size_t M : {2u, 3u, 4u}) {
		auto h_d = lowpass_taps(21);
		auto x_d = random_signal(200, 17);
		std::size_t trimmed = (x_d.size() / M) * M;
		mtl::vec::dense_vector<double> x_t(trimmed);
		for (std::size_t i = 0; i < trimmed; ++i) x_t[i] = x_d[i];

		auto h_c = cast_signal<cfloat32>(h_d);
		auto x_c = cast_signal<cfloat32>(x_t);

		PolyphaseDecimator<cfloat32> dec(h_c, M);
		auto y_poly = dec.process_block(std::span<const cfloat32>(x_c.data(), x_c.size()));
		auto y_ref  = reference_decimate_T(x_c, h_c, M);

		double err = max_diff_to_double(y_poly, y_ref);
		// cfloat<32,8> ULP near unity is ~1e-7 (same as IEEE float)
		if (err > 1e-6)
			throw std::runtime_error("test failed: cfloat decimator M=" +
				std::to_string(M) + " max error " + std::to_string(err));
	}
	std::cout << "  polyphase_decimator_in_cfloat: passed\n";
}

void test_polyphase_decompose_helper() {
	// The public polyphase_decompose helper should match the per-element
	// formula sub_taps[q][p] = h[p*M + q] with zero padding at the tail.
	mtl::vec::dense_vector<double> h({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0});
	auto sub = polyphase_decompose(h, 3);

	if (sub.size() != 3)
		throw std::runtime_error("test failed: decompose factor count");
	const std::size_t expected_len = (h.size() + 3 - 1) / 3;  // ceil(7/3) = 3
	for (std::size_t q = 0; q < 3; ++q) {
		if (sub[q].size() != expected_len)
			throw std::runtime_error("test failed: decompose sub-length");
		for (std::size_t p = 0; p < expected_len; ++p) {
			std::size_t src = p * 3 + q;
			double expected = (src < h.size()) ? h[src] : 0.0;
			if (sub[q][p] != expected)
				throw std::runtime_error("test failed: decompose sub[" +
					std::to_string(q) + "][" + std::to_string(p) + "]");
		}
	}

	// Validation: both preconditions stated in the doc must throw
	// std::invalid_argument.
	bool threw = false;
	try { polyphase_decompose(h, 0); }
	catch (const std::invalid_argument&) { threw = true; }
	if (!threw) throw std::runtime_error("test failed: factor=0 should throw");

	threw = false;
	try {
		mtl::vec::dense_vector<double> empty;
		polyphase_decompose(empty, 3);
	}
	catch (const std::invalid_argument&) { threw = true; }
	if (!threw) throw std::runtime_error("test failed: empty taps should throw");

	std::cout << "  polyphase_decompose_helper: passed\n";
}

int main() {
	try {
		std::cout << "FIR Multirate Tests\n";

		test_polyphase_interpolator_impulse();
		test_polyphase_interpolator_equivalence();
		test_polyphase_decimator_equivalence();
		test_polyphase_reset();
		test_polyphase_decimator_in_posit();
		test_polyphase_decimator_in_cfloat();
		test_polyphase_decompose_helper();
		test_overlap_add_equivalence();
		test_overlap_save_equivalence();
		test_overlap_add_default_block();
		test_overlap_save_default_block();
		test_overlap_add_streaming();
		test_overlap_convolver_state_persistence();

		std::cout << "All FIR multirate tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << '\n';
		return 1;
	}
}
