#pragma once
// realtime_spectrum.hpp: streaming spectrum estimator (overlapping FFTs).
//
// The "live FFT" engine of an FFT-based spectrum analyzer. A continuous
// input stream is pushed in via push(); the engine accumulates fft_size
// samples in a ring buffer and emits one FFT every hop_size samples.
// Output rate is decoupled from input rate by the overlap fraction:
// at hop_size = fft_size/2 (50% overlap, the default for analyzer use),
// one FFT comes out every N/2 input samples.
//
// Internally:
//   1. Ring buffer of fft_size samples (write-pointer only).
//   2. After each `hop_size` pushes (and the first time the ring fills),
//      copy the latest fft_size samples in time order, multiply by the
//      window, run fft_forward<CoeffScalar>, store the complex result.
//   3. Compute magnitude in dB with a -200 dB floor for log10(0)
//      protection.
//
// No samples are dropped: the ring buffer overwrites the OLDEST sample
// when full, so the LATEST fft_size samples are always available for
// the next windowed FFT.
//
// Mixed-precision contract:
//   - SampleScalar:  input ring storage. Memory bandwidth is dominated
//     by this; narrowing it is the cheap precision knob.
//   - WindowScalar:  window-coefficient scalar. The windowing multiply
//     happens in CoeffScalar (cast both factors).
//   - CoeffScalar:   FFT twiddle precision. Today the existing
//     fft_forward<T> uses a single T for both twiddles and butterflies,
//     so CoeffScalar drives all FFT-internal arithmetic.
//   - StateScalar:   reserved for a future fft_forward<Twiddle, State>
//     split. For now StateScalar must equal CoeffScalar.
//   - Magnitude/dB output is always double, matching the analyzer
//     pipeline's reporting-layer convention.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/spectral/fft.hpp>

namespace sw::dsp::spectrum {

namespace detail {

inline bool is_power_of_two(std::size_t n) {
	return n > 0 && (n & (n - 1)) == 0;
}

} // namespace detail

template <DspField CoeffScalar  = double,
          DspField StateScalar  = CoeffScalar,
          DspScalar SampleScalar = StateScalar,
          class WindowScalar     = CoeffScalar>
class RealtimeSpectrum {
public:
	using coeff_scalar  = CoeffScalar;
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;
	using window_scalar = WindowScalar;
	using complex_t     = complex_for_t<CoeffScalar>;

	// fft_size: must be a power of 2 (the library's fft_forward requires
	//           this; we pre-check for a clearer error).
	// hop_size: in [1, fft_size]. fft_size/2 is the conventional 50%
	//           overlap; hop_size = fft_size is non-overlapping; hop_size
	//           = 1 is fully overlapping (one FFT per input sample,
	//           expensive).
	// window:   length must equal fft_size. Hann at fft_size/2 hop
	//           satisfies COLA; rectangular (all-ones) doesn't.
	RealtimeSpectrum(std::size_t fft_size,
	                 std::size_t hop_size,
	                 std::span<const WindowScalar> window)
		: fft_size_(fft_size),
		  hop_size_(hop_size),
		  ring_(fft_size),
		  window_(fft_size),
		  work_(fft_size),
		  latest_complex_(fft_size),
		  latest_magnitude_db_(fft_size) {
		if (!detail::is_power_of_two(fft_size))
			throw std::invalid_argument(
				"RealtimeSpectrum: fft_size must be a power of 2 (got "
				+ std::to_string(fft_size) + ")");
		if (hop_size == 0 || hop_size > fft_size)
			throw std::invalid_argument(
				"RealtimeSpectrum: hop_size must be in [1, fft_size] (got "
				+ std::to_string(hop_size) + " for fft_size="
				+ std::to_string(fft_size) + ")");
		if (window.size() != fft_size)
			throw std::invalid_argument(
				"RealtimeSpectrum: window length "
				+ std::to_string(window.size())
				+ " must equal fft_size " + std::to_string(fft_size));

		for (std::size_t i = 0; i < fft_size; ++i) window_[i] = window[i];
		// ring_, latest_*, work_ are zero-initialized by mtl::vec ctor.
	}

	// Push input samples. Returns the number of complete FFTs produced
	// by this call (0 if still accumulating, possibly several for a
	// long input block). Subsequent calls to latest_*() reflect the
	// most recent FFT in this batch (or whichever was most recent
	// before, if zero FFTs were produced).
	std::size_t push(std::span<const SampleScalar> input) {
		std::size_t n_ffts = 0;
		for (const auto& x : input) {
			ring_[write_pos_] = x;
			write_pos_ = (write_pos_ + 1) % fft_size_;
			if (samples_buffered_ < fft_size_) {
				++samples_buffered_;
				if (samples_buffered_ == fft_size_) {
					// First FFT: ring just filled. Subsequent FFTs are
					// triggered by the hop counter below.
					compute_fft();
					samples_since_fft_ = 0;
					++n_ffts;
				}
			} else {
				++samples_since_fft_;
				if (samples_since_fft_ == hop_size_) {
					compute_fft();
					samples_since_fft_ = 0;
					++n_ffts;
				}
			}
		}
		return n_ffts;
	}

	// Read the most recent FFT's complex bins. Returns an empty span
	// before the first FFT has been produced (use total_ffts() == 0
	// or first_fft_ready() to detect this case).
	[[nodiscard]] std::span<const complex_t> latest_complex() const {
		if (total_ffts_ == 0) return {};
		return std::span<const complex_t>(
			latest_complex_.data(), fft_size_);
	}

	// Read the most recent FFT's magnitude in dB with a -200 dB floor.
	[[nodiscard]] std::span<const double> latest_magnitude_db() const {
		if (total_ffts_ == 0) return {};
		return std::span<const double>(
			latest_magnitude_db_.data(), fft_size_);
	}

	// Reset accumulator state but keep configuration (fft_size,
	// hop_size, window). Use between independent stream segments.
	void reset() {
		for (std::size_t i = 0; i < fft_size_; ++i)
			ring_[i] = SampleScalar{};
		write_pos_         = 0;
		samples_buffered_  = 0;
		samples_since_fft_ = 0;
		total_ffts_        = 0;
		// latest_* contents stay (callers see empty spans because of
		// the total_ffts_ == 0 check). Avoids a wasted O(N) clear.
	}

	[[nodiscard]] std::size_t fft_size()         const { return fft_size_; }
	[[nodiscard]] std::size_t hop_size()         const { return hop_size_; }
	[[nodiscard]] std::size_t total_ffts()       const { return total_ffts_; }
	[[nodiscard]] bool        first_fft_ready()  const { return total_ffts_ > 0; }

private:
	void compute_fft() {
		// Walk the ring in time order (oldest first). After fft_size_
		// samples have been pushed, write_pos_ points at the oldest
		// sample (about to be overwritten on the next push).
		// Multiply by the window in CoeffScalar precision; pack into
		// a complex buffer for the FFT.
		for (std::size_t i = 0; i < fft_size_; ++i) {
			const std::size_t r = (write_pos_ + i) % fft_size_;
			const CoeffScalar x_c = static_cast<CoeffScalar>(ring_[r]);
			const CoeffScalar w_c = static_cast<CoeffScalar>(window_[i]);
			work_[i] = complex_t(x_c * w_c, CoeffScalar{});
		}
		sw::dsp::spectral::fft_forward<CoeffScalar>(work_);
		// Store complex result and dB magnitude.
		for (std::size_t i = 0; i < fft_size_; ++i) {
			latest_complex_[i] = work_[i];
			const double re = static_cast<double>(work_[i].real());
			const double im = static_cast<double>(work_[i].imag());
			const double mag2 = re * re + im * im;
			// Floor at -200 dB to keep log10 finite. mag2 corresponding
			// to -200 dB is 10^(-20) which is well below any
			// realistic FFT noise floor.
			constexpr double mag2_floor = 1e-20;
			const double m2 = std::max(mag2, mag2_floor);
			latest_magnitude_db_[i] = 10.0 * std::log10(m2);
		}
		++total_ffts_;
	}

	std::size_t fft_size_;
	std::size_t hop_size_;
	std::size_t write_pos_         = 0;
	std::size_t samples_buffered_  = 0;
	std::size_t samples_since_fft_ = 0;
	std::size_t total_ffts_        = 0;

	mtl::vec::dense_vector<SampleScalar> ring_;
	mtl::vec::dense_vector<WindowScalar> window_;
	// FFT working buffer in complex<CoeffScalar>. Re-used across calls.
	mtl::vec::dense_vector<complex_t>    work_;
	mtl::vec::dense_vector<complex_t>    latest_complex_;
	mtl::vec::dense_vector<double>       latest_magnitude_db_;

	// StateScalar is templated to document the design intent (see the
	// file header) but the current fft_forward<T> uses a single T for
	// twiddles and butterflies. Until that splits, StateScalar exists
	// only as a contract marker. This static_assert keeps the public
	// signature honest and prevents silent precision drift if a user
	// instantiates with mismatched (CoeffScalar, StateScalar).
	static_assert(std::is_same_v<CoeffScalar, StateScalar>,
		"RealtimeSpectrum: StateScalar must equal CoeffScalar today; the "
		"library's fft_forward<T> uses one type for twiddles and "
		"butterflies. The split is reserved for a future API extension.");
};

} // namespace sw::dsp::spectrum
