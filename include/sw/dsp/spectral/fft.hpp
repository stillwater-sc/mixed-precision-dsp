#pragma once
// fft.hpp: Fast Fourier Transform (Cooley-Tukey radix-2)
//
// In-place decimation-in-time FFT with precomputed twiddle factors.
// Requires power-of-2 input length.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>

namespace sw::dsp::spectral {

namespace detail {

inline bool is_power_of_2(std::size_t n) {
	return n > 0 && (n & (n - 1)) == 0;
}

// Bit-reversal permutation
inline std::size_t bit_reverse(std::size_t x, int log2n) {
	std::size_t result = 0;
	for (int i = 0; i < log2n; ++i) {
		result = (result << 1) | (x & 1);
		x >>= 1;
	}
	return result;
}

} // namespace detail

// Forward FFT: in-place Cooley-Tukey radix-2 decimation-in-time.
// Input/output: complex vector of length N (must be power of 2).
//
// Twiddle factors are computed in T so non-native CoeffScalar callers
// (posit, cfloat, etc.) run the sin/cos through their own math library,
// not through IEEE double. ADL trig (using std::cos; using std::sin)
// selects sw::universal::{cos,sin} for Universal types.
template <DspField T>
void fft_forward(mtl::vec::dense_vector<complex_for_t<T>>& data) {
	using std::cos; using std::sin;
	using complex_t = complex_for_t<T>;
	std::size_t N = data.size();
	if (!detail::is_power_of_2(N))
		throw std::invalid_argument("fft: size must be a power of 2");

	// Compute log2(N)
	int log2n = 0;
	for (std::size_t tmp = N; tmp > 1; tmp >>= 1) ++log2n;

	// Bit-reversal permutation
	for (std::size_t i = 0; i < N; ++i) {
		std::size_t j = detail::bit_reverse(i, log2n);
		if (i < j) {
			complex_t tmp = data[i];
			data[i] = data[j];
			data[j] = tmp;
		}
	}

	// Butterfly stages. Twiddle generation stays in T.
	constexpr T two_pi_T = T(two_pi);
	for (int s = 1; s <= log2n; ++s) {
		std::size_t m = std::size_t{1} << s;
		std::size_t m2 = m >> 1;
		const T angle_step = -two_pi_T / T(m);

		for (std::size_t k = 0; k < N; k += m) {
			for (std::size_t j = 0; j < m2; ++j) {
				const T angle = angle_step * T(j);
				complex_t w(cos(angle), sin(angle));
				complex_t t = w * data[k + j + m2];
				complex_t u = data[k + j];
				data[k + j] = u + t;
				data[k + j + m2] = u - t;
			}
		}
	}
}

// Inverse FFT: conjugate input, forward FFT, conjugate output, scale by 1/N.
template <DspField T>
void fft_inverse(mtl::vec::dense_vector<complex_for_t<T>>& data) {
	using complex_t = complex_for_t<T>;
	using std::conj;
	std::size_t N = data.size();

	// Conjugate
	for (std::size_t i = 0; i < N; ++i) {
		data[i] = complex_t(data[i].real(), T{} - data[i].imag());
	}

	fft_forward<T>(data);

	// Conjugate and scale
	T inv_N = T{1} / static_cast<T>(N);
	for (std::size_t i = 0; i < N; ++i) {
		data[i] = complex_t(data[i].real() * inv_N, (T{} - data[i].imag()) * inv_N);
	}
}

// Convenience: forward FFT of a real signal.
// Zero-pads to next power of 2 if necessary.
template <DspField T>
mtl::vec::dense_vector<complex_for_t<T>> fft(const mtl::vec::dense_vector<T>& x) {
	using complex_t = complex_for_t<T>;

	// Find next power of 2
	std::size_t N = 1;
	while (N < x.size()) N <<= 1;

	mtl::vec::dense_vector<complex_t> data(N, complex_t{});
	for (std::size_t i = 0; i < x.size(); ++i) {
		data[i] = complex_t(x[i]);
	}

	fft_forward<T>(data);
	return data;
}

// Convenience: inverse FFT returning real part.
template <DspField T>
mtl::vec::dense_vector<T> ifft_real(const mtl::vec::dense_vector<complex_for_t<T>>& X) {
	using complex_t = complex_for_t<T>;

	mtl::vec::dense_vector<complex_t> data(X.size());
	for (std::size_t i = 0; i < X.size(); ++i) data[i] = X[i];

	fft_inverse<T>(data);

	mtl::vec::dense_vector<T> result(data.size());
	for (std::size_t i = 0; i < data.size(); ++i) {
		result[i] = data[i].real();
	}
	return result;
}

// Magnitude spectrum in dB
template <DspField T>
mtl::vec::dense_vector<double> magnitude_spectrum_db(
		const mtl::vec::dense_vector<complex_for_t<T>>& X,
		double min_db = -120.0) {
	using std::abs;
	mtl::vec::dense_vector<double> mag(X.size());
	for (std::size_t i = 0; i < X.size(); ++i) {
		double m = static_cast<double>(abs(X[i]));
		mag[i] = (m > 0.0) ? 20.0 * std::log10(m) : min_db;
	}
	return mag;
}

} // namespace sw::dsp::spectral
