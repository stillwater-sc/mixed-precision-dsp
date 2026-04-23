#pragma once
// decimation_chain.hpp: composable multi-stage decimation pipeline
//
// High-rate acquisition systems achieve large decimation ratios by
// cascading heterogeneous filters:
//
//   ADC -> CIC (bulk rate reduction) -> half-band -> half-band -> FIR -> baseband
//
// DecimationChain holds a variadic tuple of stage instances and threads
// samples through them. Each stage may use a different internal
// coefficient/state precision; the stream type at the stage boundaries
// is a single Sample template parameter.
//
// Streaming contract: process(x) returns {true, y} only when the final
// stage emits. Internally the chain short-circuits as soon as any stage
// is between emit cycles, so per-input-sample cost is low.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/acquisition/detail/decimator_step.hpp>
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/math/constants.hpp>

namespace sw::dsp {

namespace detail {

// Query a stage's decimation ratio via whatever query method it exposes.
//   CICDecimator:      decimation_ratio()
//   PolyphaseDecimator: factor()
//   HalfBandFilter:    has process_decimate(), always 2:1
//
// Validates that the returned value is strictly positive so callers
// (total_decimation, stage_ratios) cannot divide by zero or wrap on cast
// from a signed integer. The existing library stages already enforce
// positive ratios in their constructors, so this is defensive for
// user-provided custom stages.
template <class T>
std::size_t decimation_ratio_of(const T& t) {
	auto validate = [](auto r) -> std::size_t {
		if (!(r > 0))
			throw std::invalid_argument(
				"decimation_ratio_of: stage decimation ratio must be positive");
		return static_cast<std::size_t>(r);
	};
	if constexpr (requires { t.decimation_ratio(); }) {
		return validate(t.decimation_ratio());
	} else if constexpr (requires { t.factor(); }) {
		return validate(t.factor());
	} else if constexpr (requires (T x, typename T::sample_scalar s) { x.process_decimate(s); }) {
		return 2;
	} else {
		static_assert(sizeof(T) == 0,
			"decimation_ratio_of: stage must expose decimation_ratio(), factor(), or process_decimate()");
	}
}

} // namespace detail

// Multi-stage decimation chain.
//
// Sample:  stream type that flows between stages (e.g., double, float, posit<32,2>)
// Stages:  variadic list of decimator types. Each stage must implement one of:
//            process(Sample)           returning std::pair<bool, Sample>
//            process_decimate(Sample)  returning std::pair<bool, Sample>
//            push(Sample) + output()   (CIC-style)
//          and must expose a decimation-ratio query (see decimation_ratio_of).
template <DspField Sample, class... Stages>
class DecimationChain {
public:
	using sample_t = Sample;
	static constexpr std::size_t num_stages = sizeof...(Stages);

	// Construct with an input sample rate (in Hz) and the stage instances.
	// Stages are moved into an internal std::tuple.
	explicit DecimationChain(Sample input_rate, Stages... stages)
		: stages_(std::move(stages)...),
		  input_rate_(input_rate) {}

	// Streaming: feed one input sample. Returns {true, y} when the final stage
	// emits, else {false, 0}.
	std::pair<bool, Sample> process(Sample in) {
		return process_impl<0>(in);
	}

	// Block: process a span of inputs and return all emitted outputs.
	mtl::vec::dense_vector<Sample> process_block(std::span<const Sample> input) {
		return process_block_impl(input);
	}

	// Dense-vector overload.
	mtl::vec::dense_vector<Sample> process_block(
			const mtl::vec::dense_vector<Sample>& input) {
		return process_block_impl(input);
	}

	// Reset every stage.
	void reset() {
		reset_impl<0>();
	}

	// Rate queries.
	Sample input_rate()  const { return input_rate_; }
	Sample output_rate() const { return input_rate_ / static_cast<Sample>(total_decimation()); }

	// Product of per-stage decimation ratios.
	std::size_t total_decimation() const {
		return total_decimation_impl<0>();
	}

	// Per-stage decimation ratios (input-order).
	std::array<std::size_t, num_stages> stage_ratios() const {
		std::array<std::size_t, num_stages> result{};
		fill_ratios_impl<0>(result);
		return result;
	}

	// Per-stage output rates: element i is the rate at the output of stage i.
	// stage_rates()[num_stages - 1] == output_rate().
	std::array<Sample, num_stages> stage_rates() const {
		std::array<Sample, num_stages> result{};
		auto ratios = stage_ratios();
		Sample rate = input_rate_;
		for (std::size_t i = 0; i < num_stages; ++i) {
			rate = rate / static_cast<Sample>(ratios[i]);
			result[i] = rate;
		}
		return result;
	}

	// Accessors to individual stages.
	template <std::size_t I>
	auto& stage()       { return std::get<I>(stages_); }
	template <std::size_t I>
	const auto& stage() const { return std::get<I>(stages_); }

private:
	std::tuple<Stages...> stages_;
	Sample                input_rate_;

	// Shared block-processing body. Input may be any container with .size()
	// and operator[], so both std::span and mtl::vec::dense_vector work.
	//
	// Uses mtl::vec::dense_vector (not std::vector) for signal staging per
	// the library's container convention. The worst-case emit count is
	// ceil(input.size() / total_decimation) + 1; we allocate that and
	// return a right-sized dense_vector built from the produced samples.
	template <class Input>
	mtl::vec::dense_vector<Sample> process_block_impl(const Input& input) {
		std::size_t dec = total_decimation();
		std::size_t max_out = (dec > 0) ? (input.size() / dec + 1) : input.size();
		mtl::vec::dense_vector<Sample> staging(max_out);
		std::size_t produced = 0;
		for (std::size_t n = 0; n < input.size(); ++n) {
			auto [ready, y] = process(input[n]);
			if (ready) staging[produced++] = y;
		}
		mtl::vec::dense_vector<Sample> out(produced);
		for (std::size_t i = 0; i < produced; ++i) out[i] = staging[i];
		return out;
	}

	template <std::size_t I>
	std::pair<bool, Sample> process_impl(Sample in) {
		if constexpr (I == sizeof...(Stages)) {
			return {true, in};
		} else {
			auto& stg = std::get<I>(stages_);
			auto [ready, y] = detail::step_decimator(stg, in);
			if (!ready) return {false, Sample{}};
			return process_impl<I + 1>(y);
		}
	}

	template <std::size_t I>
	void reset_impl() {
		if constexpr (I < sizeof...(Stages)) {
			std::get<I>(stages_).reset();
			reset_impl<I + 1>();
		}
	}

	template <std::size_t I>
	std::size_t total_decimation_impl() const {
		if constexpr (I == sizeof...(Stages)) {
			return 1;
		} else {
			return detail::decimation_ratio_of(std::get<I>(stages_))
			     * total_decimation_impl<I + 1>();
		}
	}

	template <std::size_t I>
	void fill_ratios_impl(std::array<std::size_t, num_stages>& out) const {
		if constexpr (I < num_stages) {
			out[I] = detail::decimation_ratio_of(std::get<I>(stages_));
			fill_ratios_impl<I + 1>(out);
		}
	}
};

// Class template argument deduction guide.
template <DspField Sample, class... Stages>
DecimationChain(Sample, Stages...) -> DecimationChain<Sample, Stages...>;

// ---------------------------------------------------------------------------
// CIC droop compensator design (frequency sampling)
// ---------------------------------------------------------------------------
//
// A CIC decimator's passband magnitude response (referred to the low-rate
// output) is
//
//     |H(f)| = | sinc(pi * f * D)      |^M
//              | ---------------       |
//              | sinc(pi * f * D / R)  |
//
// where f is the *output-rate*-normalized frequency in [0, 0.5). This
// response droops substantially within any non-trivial passband; a short
// compensation FIR run at the CIC output rate inverts the droop.
//
// This design uses the frequency-sampling method: sample the desired
// magnitude at num_taps uniformly-spaced points, symmetrize for real impulse
// response, IDFT, apply a Hamming window. Simple and analytic. For a more
// aggressive equaliser, use a Remez-based design.
//
// num_taps:          FIR length; an odd value gives a linear-phase centered tap.
// cic_stages:        M (number of integrator/comb sections).
// cic_ratio:         R (decimation ratio).
// passband:          normalized passband edge in (0, 0.5) at the CIC output rate.
// differential_delay: D (comb stage delay; defaults to 1, the usual case).
//
// The CIC magnitude referred to the output-rate normalized frequency f is
//   |H_cic(f)| = | sin(pi f D) / (R D sin(pi f / R)) |^M
// The compensator approximates 1/|H_cic(f)| across [0, passband], rolling off
// smoothly toward 0 past the passband.
//
// Intermediate math is performed in T so that calls with posit, cfloat, or
// other custom scalars run their design-time arithmetic at the caller's
// declared precision — a requirement for embedded mixed-precision deployments
// where the compensator design may run on the target.
template <DspField T>
mtl::vec::dense_vector<T> design_cic_compensator(
		std::size_t num_taps,
		int cic_stages,
		int cic_ratio,
		T passband,
		int differential_delay = 1) {
	// ADL-friendly dispatch so sw::universal::{sin,cos,pow,abs} are found
	// for non-native T; std::{sin,...} for native float/double.
	using std::abs; using std::sin; using std::cos; using std::pow;
	if (num_taps < 3)
		throw std::invalid_argument("design_cic_compensator: num_taps must be >= 3");
	if (cic_stages < 1)
		throw std::invalid_argument("design_cic_compensator: cic_stages must be >= 1");
	if (cic_ratio < 2)
		throw std::invalid_argument("design_cic_compensator: cic_ratio must be >= 2");
	if (differential_delay < 1)
		throw std::invalid_argument("design_cic_compensator: differential_delay must be >= 1");

	const T zero{};
	const T one   = T(1);
	const T half  = T(1) / T(2);
	const T pi_T  = T(pi);                 // runtime init — avoids posit constexpr path
	const T R     = T(cic_ratio);
	const T M     = T(cic_stages);
	const T D     = T(differential_delay);
	const T N_T   = T(num_taps);

	if (!(passband > zero) || !(passband < half))
		throw std::invalid_argument("design_cic_compensator: passband must be in (0, 0.5)");

	// tiny threshold for |sin(pi f / R)| near zero (f = 0): we handle f == 0
	// via the DC-gain unit-value branch below, so this guard is a safety net.
	const T tiny = T(1) / T(1'000'000'000'000LL);

	// Compute the desired magnitude at num_taps frequency points in [0, 0.5].
	// Within the passband: 1 / |H_cic(f)|. Beyond: smooth cosine rolloff to 0.
	auto desired = [&](T f) -> T {
		if (f == zero) return one;  // CIC is unit-gain after normalization at DC
		T s_num = sin(pi_T * f * D);
		T s_den = R * D * sin(pi_T * f / R);
		if (abs(s_den) < tiny) return zero;
		T ratio = s_num / s_den;
		T mag = pow(abs(ratio), M);
		if (mag < tiny) return zero;
		if (f <= passband) {
			return one / mag;
		} else if (f < half) {
			// Cosine rolloff between passband and Nyquist.
			T t = (f - passband) / (half - passband);
			T roll = half * (one + cos(pi_T * t));
			return (one / mag) * roll;
		}
		return zero;
	};

	// Frequency-sampling design for a type I (odd length, symmetric) linear-phase FIR.
	// h[n] = (1/N) * sum_{k=0..N-1} H_k * exp(j 2 pi k n / N), with conjugate-symmetric H
	// yielding a real h. H_k = desired(k/N) * exp(-j 2 pi k (N-1)/2 / N) to center.
	std::size_t N = num_taps;
	mtl::vec::dense_vector<T> taps(N);
	const T shift = T(N - 1) * half;
	const T two_pi_T = T(two_pi);

	for (std::size_t n = 0; n < N; ++n) {
		T h = zero;
		const T n_T = T(n);
		for (std::size_t k = 0; k < N; ++k) {
			const T k_T = T(k);
			T fk = k_T / N_T;
			// Mirror frequencies above 0.5 for conjugate symmetry.
			T f_mag = (fk > half) ? (one - fk) : fk;
			T Hk_mag = desired(f_mag);
			T phase = -two_pi_T * k_T * shift / N_T
			        +  two_pi_T * k_T * n_T   / N_T;
			h = h + Hk_mag * cos(phase);
		}
		taps[n] = h / N_T;
	}

	// Apply a Hamming window to suppress ripple from finite-length truncation.
	const T a0 = T(54) / T(100);
	const T a1 = T(46) / T(100);
	for (std::size_t n = 0; n < N; ++n) {
		T w = a0 - a1 * cos(two_pi_T * T(n) / T(N - 1));
		taps[n] = taps[n] * w;
	}

	// Normalize to unit DC gain so the compensator preserves signal scale.
	T dc_gain = zero;
	for (std::size_t n = 0; n < N; ++n) dc_gain = dc_gain + taps[n];
	if (!(dc_gain == zero)) {
		for (std::size_t n = 0; n < N; ++n) taps[n] = taps[n] / dc_gain;
	}
	return taps;
}

} // namespace sw::dsp
