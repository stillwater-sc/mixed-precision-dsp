#pragma once
// bluestein.hpp: Bluestein's chirp-z algorithm for arbitrary-length DFT
//
// Converts an N-point DFT into circular convolution of length M
// (next power-of-2 >= 2N-1), computed via three radix-2 FFTs.
// Total complexity: O(N log N) for any N, not just powers of 2.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/spectral/fft.hpp>

namespace sw::dsp::spectral {

namespace detail {

inline std::size_t next_power_of_2(std::size_t n) {
	if (n <= 1) return 1;
	constexpr std::size_t max_pow2 =
		std::size_t{1} << (std::numeric_limits<std::size_t>::digits - 1);
	if (n > max_pow2)
		throw std::length_error("bluestein: convolution length too large");
	std::size_t m = 1;
	while (m < n) m <<= 1;
	return m;
}

} // namespace detail

// Bluestein forward DFT for arbitrary length N.
// Input: complex vector of length N.
// Output: complex vector of length N (the DFT).
template <DspField T>
mtl::vec::dense_vector<complex_for_t<T>> bluestein_forward(
		const mtl::vec::dense_vector<complex_for_t<T>>& x) {
	using complex_t = complex_for_t<T>;
	using std::conj;
	using std::cos;
	using std::sin;
	const std::size_t N = x.size();
	if (N == 0) return {};
	if (N == 1) return mtl::vec::dense_vector<complex_t>({x[0]});

	if (N > (std::numeric_limits<std::size_t>::max() - 1) / 2)
		throw std::length_error("bluestein: input length too large");
	const std::size_t M = detail::next_power_of_2(2 * N - 1);

	// Chirp sequence: w[n] = exp(-j * pi * n^2 / N)
	mtl::vec::dense_vector<complex_t> chirp(N);
	for (std::size_t n = 0; n < N; ++n) {
		T angle = -pi_v<T> * static_cast<T>(n) * static_cast<T>(n)
		          / static_cast<T>(N);
		chirp[n] = complex_t(cos(angle), sin(angle));
	}

	// a[n] = x[n] * chirp[n], zero-padded to length M
	mtl::vec::dense_vector<complex_t> a(M, complex_t{});
	for (std::size_t n = 0; n < N; ++n) {
		a[n] = x[n] * chirp[n];
	}

	// b[n] = conj(chirp[n]) with wrap-around for circular convolution
	mtl::vec::dense_vector<complex_t> b(M, complex_t{});
	b[0] = conj(chirp[0]);
	for (std::size_t n = 1; n < N; ++n) {
		complex_t cj = conj(chirp[n]);
		b[n] = cj;
		b[M - n] = cj;
	}

	// Circular convolution via FFT: C = IFFT(FFT(a) .* FFT(b))
	fft_forward<T>(a);
	fft_forward<T>(b);

	for (std::size_t i = 0; i < M; ++i) {
		a[i] = a[i] * b[i];
	}

	fft_inverse<T>(a);

	// X[k] = chirp[k] * C[k]
	mtl::vec::dense_vector<complex_t> X(N);
	for (std::size_t k = 0; k < N; ++k) {
		X[k] = chirp[k] * a[k];
	}
	return X;
}

// Bluestein inverse DFT for arbitrary length N.
// Conjugate-transform-conjugate-scale approach.
template <DspField T>
mtl::vec::dense_vector<complex_for_t<T>> bluestein_inverse(
		const mtl::vec::dense_vector<complex_for_t<T>>& X) {
	using complex_t = complex_for_t<T>;
	using std::conj;
	const std::size_t N = X.size();
	if (N == 0) return {};

	mtl::vec::dense_vector<complex_t> conj_X(N);
	for (std::size_t i = 0; i < N; ++i) {
		conj_X[i] = conj(X[i]);
	}

	auto result = bluestein_forward<T>(conj_X);

	T inv_N = T{1} / static_cast<T>(N);
	for (std::size_t i = 0; i < N; ++i) {
		result[i] = conj(result[i]) * inv_N;
	}
	return result;
}

// Auto-dispatching forward DFT: uses radix-2 FFT for power-of-2,
// Bluestein for arbitrary lengths.
template <DspField T>
mtl::vec::dense_vector<complex_for_t<T>> dft_forward(
		const mtl::vec::dense_vector<complex_for_t<T>>& x) {
	if (detail::is_power_of_2(x.size())) {
		auto data = x;
		fft_forward<T>(data);
		return data;
	}
	return bluestein_forward<T>(x);
}

// Auto-dispatching inverse DFT: uses radix-2 FFT for power-of-2,
// Bluestein for arbitrary lengths.
template <DspField T>
mtl::vec::dense_vector<complex_for_t<T>> dft_inverse(
		const mtl::vec::dense_vector<complex_for_t<T>>& x) {
	if (detail::is_power_of_2(x.size())) {
		auto data = x;
		fft_inverse<T>(data);
		return data;
	}
	return bluestein_inverse<T>(x);
}

} // namespace sw::dsp::spectral
