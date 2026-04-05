#pragma once
// generators.hpp: signal generator functions
//
// Free functions that produce standard test signals as mtl::vec::dense_vector<T>.
// All generators are templated on DspField T for mixed-precision support.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <random>
#include <span>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>

namespace sw::dsp {

// ============================================================================
// Periodic waveforms
// ============================================================================

// Pure sinusoid: x[n] = amplitude * sin(2*pi*frequency*n/sample_rate + phase)
template <DspField T>
mtl::vec::dense_vector<T> sine(std::size_t length, T frequency, T sample_rate,
                    T amplitude = T{1}, T phase = T{}) {
	mtl::vec::dense_vector<T> out(length);
	const T w = two_pi_v<T> * frequency / sample_rate;
	for (std::size_t n = 0; n < length; ++n) {
		out[n] = amplitude * static_cast<T>(
			std::sin(static_cast<double>(w * static_cast<T>(n) + phase)));
	}
	return out;
}

// Cosine: x[n] = amplitude * cos(2*pi*frequency*n/sample_rate + phase)
template <DspField T>
mtl::vec::dense_vector<T> cosine(std::size_t length, T frequency, T sample_rate,
                      T amplitude = T{1}, T phase = T{}) {
	return sine(length, frequency, sample_rate, amplitude,
	            phase + half_pi_v<T>);
}

// Triangle wave via piecewise linear construction.
// Fundamental at `frequency` with odd harmonics rolling off as 1/n^2.
template <DspField T>
mtl::vec::dense_vector<T> triangle(std::size_t length, T frequency, T sample_rate,
                        T amplitude = T{1}) {
	mtl::vec::dense_vector<T> out(length);
	const double period = static_cast<double>(sample_rate / frequency);
	for (std::size_t n = 0; n < length; ++n) {
		// Phase in [0, 1)
		double phase = std::fmod(static_cast<double>(n) / period, 1.0);
		// Triangle: rises 0→1 in first half, falls 1→-1 in second half
		double value;
		if (phase < 0.25) {
			value = 4.0 * phase;
		} else if (phase < 0.75) {
			value = 2.0 - 4.0 * phase;
		} else {
			value = -4.0 + 4.0 * phase;
		}
		out[n] = amplitude * static_cast<T>(value);
	}
	return out;
}

// Square wave with configurable duty cycle.
// duty_cycle = 0.5 gives a symmetric square wave (odd harmonics, 1/n rolloff).
template <DspField T>
mtl::vec::dense_vector<T> square(std::size_t length, T frequency, T sample_rate,
                      T amplitude = T{1}, T duty_cycle = T{0.5}) {
	mtl::vec::dense_vector<T> out(length);
	const double period = static_cast<double>(sample_rate / frequency);
	const double duty = static_cast<double>(duty_cycle);
	for (std::size_t n = 0; n < length; ++n) {
		double phase = std::fmod(static_cast<double>(n) / period, 1.0);
		out[n] = (phase < duty) ? amplitude : (T{} - amplitude);
	}
	return out;
}

// Sawtooth wave: rises linearly from -amplitude to +amplitude over each period.
// Contains all harmonics with 1/n rolloff.
template <DspField T>
mtl::vec::dense_vector<T> sawtooth(std::size_t length, T frequency, T sample_rate,
                        T amplitude = T{1}) {
	mtl::vec::dense_vector<T> out(length);
	const double period = static_cast<double>(sample_rate / frequency);
	for (std::size_t n = 0; n < length; ++n) {
		double phase = std::fmod(static_cast<double>(n) / period, 1.0);
		out[n] = amplitude * static_cast<T>(2.0 * phase - 1.0);
	}
	return out;
}

// ============================================================================
// Aperiodic signals
// ============================================================================

// Unit impulse (Kronecker delta): x[delay] = amplitude, zero elsewhere.
template <DspField T>
mtl::vec::dense_vector<T> impulse(std::size_t length, std::size_t delay = 0,
                       T amplitude = T{1}) {
	mtl::vec::dense_vector<T> out(length, T{});
	if (delay < length) {
		out[delay] = amplitude;
	}
	return out;
}

// Unit step: x[n] = amplitude for n >= delay, zero for n < delay.
template <DspField T>
mtl::vec::dense_vector<T> step(std::size_t length, std::size_t delay = 0,
                    T amplitude = T{1}) {
	mtl::vec::dense_vector<T> out(length, T{});
	for (std::size_t n = delay; n < length; ++n) {
		out[n] = amplitude;
	}
	return out;
}

// Linear ramp: x[n] = slope * n.
template <DspField T>
mtl::vec::dense_vector<T> ramp(std::size_t length, T slope = T{1}) {
	mtl::vec::dense_vector<T> out(length);
	for (std::size_t n = 0; n < length; ++n) {
		out[n] = slope * static_cast<T>(n);
	}
	return out;
}

// ============================================================================
// Noise
// ============================================================================

// White noise: uniformly distributed in [-amplitude, +amplitude].
// Deterministic when seed != 0.
template <DspField T>
mtl::vec::dense_vector<T> white_noise(std::size_t length, T amplitude = T{1},
                           unsigned seed = 0) {
	mtl::vec::dense_vector<T> out(length);
	std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);
	for (std::size_t n = 0; n < length; ++n) {
		out[n] = amplitude * static_cast<T>(dist(gen));
	}
	return out;
}

// Gaussian white noise: normally distributed with mean=0, stddev=amplitude.
template <DspField T>
mtl::vec::dense_vector<T> gaussian_noise(std::size_t length, T amplitude = T{1},
                              unsigned seed = 0) {
	mtl::vec::dense_vector<T> out(length);
	std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
	std::normal_distribution<double> dist(0.0, 1.0);
	for (std::size_t n = 0; n < length; ++n) {
		out[n] = amplitude * static_cast<T>(dist(gen));
	}
	return out;
}

// Pink noise (1/f): Voss-McCartney algorithm.
// Approximates 1/f spectrum using a sum of random generators
// updated at different rates.
template <DspField T>
mtl::vec::dense_vector<T> pink_noise(std::size_t length, T amplitude = T{1},
                          unsigned seed = 0) {
	constexpr int num_rows = 12;  // number of octave bands
	mtl::vec::dense_vector<T> out(length);
	std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);

	double rows[num_rows]{};
	double running_sum = 0.0;
	for (int i = 0; i < num_rows; ++i) {
		rows[i] = dist(gen);
		running_sum += rows[i];
	}

	const double scale = 1.0 / (num_rows + 1);
	for (std::size_t n = 0; n < length; ++n) {
		// Determine which rows to update based on trailing zeros of n
		std::size_t k = n;
		int row = 0;
		while (row < num_rows && (k & 1) == 0 && k > 0) {
			running_sum -= rows[row];
			rows[row] = dist(gen);
			running_sum += rows[row];
			k >>= 1;
			++row;
		}
		double white = dist(gen);
		out[n] = amplitude * static_cast<T>((running_sum + white) * scale);
	}
	return out;
}

// ============================================================================
// Composite signals
// ============================================================================

// Linear chirp (frequency sweep): frequency increases linearly from
// start_freq to end_freq over the signal duration.
template <DspField T>
mtl::vec::dense_vector<T> chirp(std::size_t length, T start_freq, T end_freq,
                     T sample_rate, T amplitude = T{1}) {
	mtl::vec::dense_vector<T> out(length);
	const double duration = static_cast<double>(length) / static_cast<double>(sample_rate);
	const double f0 = static_cast<double>(start_freq);
	const double f1 = static_cast<double>(end_freq);
	const double rate = (f1 - f0) / duration;

	for (std::size_t n = 0; n < length; ++n) {
		double t = static_cast<double>(n) / static_cast<double>(sample_rate);
		double phase = 2.0 * pi * (f0 * t + 0.5 * rate * t * t);
		out[n] = amplitude * static_cast<T>(std::sin(phase));
	}
	return out;
}

// Multitone: sum of sinusoids at specified frequencies.
// Useful for filter demos: place tones below and above cutoff to
// visualize passband/stopband behavior.
template <DspField T>
mtl::vec::dense_vector<T> multitone(std::size_t length,
                         std::span<const T> frequencies,
                         T sample_rate, T amplitude = T{1}) {
	mtl::vec::dense_vector<T> out(length, T{});
	if (frequencies.empty()) return out;

	// Scale each tone so the total amplitude matches `amplitude`
	const double per_tone = static_cast<double>(amplitude) /
	                        static_cast<double>(frequencies.size());

	for (const auto& freq : frequencies) {
		const double w = 2.0 * pi * static_cast<double>(freq) /
		                 static_cast<double>(sample_rate);
		for (std::size_t n = 0; n < length; ++n) {
			out[n] = out[n] + static_cast<T>(per_tone * std::sin(w * static_cast<double>(n)));
		}
	}
	return out;
}

} // namespace sw::dsp
