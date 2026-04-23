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
#include <tuple>
#include <utility>
#include <vector>
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
template <class T>
std::size_t decimation_ratio_of(const T& t) {
	if constexpr (requires { t.decimation_ratio(); }) {
		return static_cast<std::size_t>(t.decimation_ratio());
	} else if constexpr (requires { t.factor(); }) {
		return t.factor();
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
	template <class Input>
	mtl::vec::dense_vector<Sample> process_block_impl(const Input& input) {
		std::vector<Sample> tmp;
		std::size_t dec = total_decimation();
		if (dec > 0) tmp.reserve(input.size() / dec + 1);
		for (std::size_t n = 0; n < input.size(); ++n) {
			auto [ready, y] = process(input[n]);
			if (ready) tmp.push_back(y);
		}
		mtl::vec::dense_vector<Sample> out(tmp.size());
		for (std::size_t i = 0; i < tmp.size(); ++i) out[i] = tmp[i];
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
template <DspField T>
mtl::vec::dense_vector<T> design_cic_compensator(
		std::size_t num_taps,
		int cic_stages,
		int cic_ratio,
		T passband,
		int differential_delay = 1) {
	using std::abs; using std::sin; using std::cos; using std::pow;
	if (num_taps < 3)
		throw std::invalid_argument("design_cic_compensator: num_taps must be >= 3");
	if (cic_stages < 1)
		throw std::invalid_argument("design_cic_compensator: cic_stages must be >= 1");
	if (cic_ratio < 2)
		throw std::invalid_argument("design_cic_compensator: cic_ratio must be >= 2");
	if (differential_delay < 1)
		throw std::invalid_argument("design_cic_compensator: differential_delay must be >= 1");
	double pb = static_cast<double>(passband);
	if (!(pb > 0.0 && pb < 0.5))
		throw std::invalid_argument("design_cic_compensator: passband must be in (0, 0.5)");

	// Compute the desired magnitude at num_taps frequency points in [0, 0.5].
	// Within the passband: 1 / |H_cic(f)|. Beyond: smooth cosine rolloff to 0.
	auto desired = [&](double f) -> double {
		double R = static_cast<double>(cic_ratio);
		double M = static_cast<double>(cic_stages);
		double D = static_cast<double>(differential_delay);
		// H_cic(f) at the output-rate normalized frequency:
		//   numerator   : sin(pi * f * D)
		//   denominator : R * D * sin(pi * f / R)
		double s_num = std::sin(pi * f * D);
		double s_den = R * D * std::sin(pi * f / R);
		double ratio = (f == 0.0 || s_den == 0.0) ? 1.0 : s_num / s_den;
		double mag = std::pow(std::abs(ratio), M);
		if (mag < 1e-12) return 0.0;
		if (f <= pb) {
			return 1.0 / mag;
		} else if (f < 0.5) {
			// Cosine rolloff between passband and Nyquist.
			double t = (f - pb) / (0.5 - pb);
			double roll = 0.5 * (1.0 + std::cos(pi * t));
			return (1.0 / mag) * roll;
		}
		return 0.0;
	};

	// Frequency-sampling design for a type I (odd length, symmetric) linear-phase FIR.
	// h[n] = (1/N) * sum_{k=0..N-1} H_k * exp(j 2 pi k n / N), with conjugate-symmetric H
	// yielding a real h. We set H_k = desired(k/N) * exp(-j 2 pi k (N-1)/2 / N) to center.
	std::size_t N = num_taps;
	mtl::vec::dense_vector<T> taps(N);
	double shift = static_cast<double>(N - 1) / 2.0;

	for (std::size_t n = 0; n < N; ++n) {
		double h = 0.0;
		for (std::size_t k = 0; k < N; ++k) {
			double fk = static_cast<double>(k) / static_cast<double>(N);
			// Mirror frequencies above 0.5 for conjugate symmetry.
			double f_mag = (fk > 0.5) ? (1.0 - fk) : fk;
			double Hk_mag = desired(f_mag);
			double phase = -2.0 * pi * static_cast<double>(k) * shift / static_cast<double>(N)
			             + 2.0 * pi * static_cast<double>(k) * static_cast<double>(n) / static_cast<double>(N);
			h += Hk_mag * std::cos(phase);
		}
		h /= static_cast<double>(N);
		taps[n] = static_cast<T>(h);
	}

	// Apply a Hamming window to suppress ripple from finite-length truncation.
	for (std::size_t n = 0; n < N; ++n) {
		double w = 0.54 - 0.46 * std::cos(2.0 * pi * static_cast<double>(n) /
			static_cast<double>(N - 1));
		taps[n] = taps[n] * static_cast<T>(w);
	}

	// Normalize to unit DC gain so the compensator preserves signal scale.
	T dc_gain{};
	for (std::size_t n = 0; n < N; ++n) dc_gain = dc_gain + taps[n];
	if (!(static_cast<double>(dc_gain) == 0.0)) {
		for (std::size_t n = 0; n < N; ++n) taps[n] = taps[n] / dc_gain;
	}
	return taps;
}

} // namespace sw::dsp
