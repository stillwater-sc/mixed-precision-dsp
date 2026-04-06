#pragma once
// dft.hpp: Discrete Fourier Transform (naive O(N^2))
//
// Reference implementation for small sizes and verification.
// For production use, prefer FFT (O(N log N)).
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>

namespace sw::dsp::spectral {

// Forward DFT: X[k] = sum_{n=0}^{N-1} x[n] * exp(-j*2*pi*k*n/N)
// Input: real signal. Output: complex spectrum (N points).
template <DspField T>
mtl::vec::dense_vector<complex_for_t<T>> dft(const mtl::vec::dense_vector<T>& x) {
	using complex_t = complex_for_t<T>;
	std::size_t N = x.size();
	mtl::vec::dense_vector<complex_t> X(N);

	for (std::size_t k = 0; k < N; ++k) {
		complex_t sum{};
		for (std::size_t n = 0; n < N; ++n) {
			double angle = -two_pi * static_cast<double>(k) * static_cast<double>(n) / static_cast<double>(N);
			complex_t twiddle(static_cast<T>(std::cos(angle)), static_cast<T>(std::sin(angle)));
			sum = sum + complex_t(x[n]) * twiddle;
		}
		X[k] = sum;
	}
	return X;
}

// Inverse DFT: x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(j*2*pi*k*n/N)
template <DspField T>
mtl::vec::dense_vector<T> idft(const mtl::vec::dense_vector<complex_for_t<T>>& X) {
	using complex_t = complex_for_t<T>;
	std::size_t N = X.size();
	mtl::vec::dense_vector<T> x(N);

	T inv_N = T{1} / static_cast<T>(N);
	for (std::size_t n = 0; n < N; ++n) {
		complex_t sum{};
		for (std::size_t k = 0; k < N; ++k) {
			double angle = two_pi * static_cast<double>(k) * static_cast<double>(n) / static_cast<double>(N);
			complex_t twiddle(static_cast<T>(std::cos(angle)), static_cast<T>(std::sin(angle)));
			sum = sum + X[k] * twiddle;
		}
		x[n] = sum.real() * inv_N;
	}
	return x;
}

} // namespace sw::dsp::spectral
