#pragma once
// cic.hpp: Cascaded Integrator-Comb (CIC) decimation and interpolation filters
//
// CIC filters perform sample-rate conversion using only additions and
// subtractions — no multiplications. This makes them ideal for the first
// decimation stage after a high-speed ADC where the input rate can exceed
// GHz and multiplier resources are scarce.
//
// Two-scalar parameterization:
//   StateScalar  — integrator/comb accumulators (must accommodate bit growth)
//   SampleScalar — input/output samples
//
// Bit growth: B_out = B_in + M * ceil(log2(R * D))
// where R = decimation ratio, D = differential delay, M = number of stages.
// StateScalar must be at least B_out bits wide to avoid overflow.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

// Compute the bit growth for a CIC filter: M * ceil(log2(R * D))
inline int cic_bit_growth(int stages, int decimation, int delay = 1) {
	if (stages <= 0 || decimation <= 0 || delay <= 0) return 0;
	return stages * static_cast<int>(std::ceil(
		std::log2(static_cast<double>(decimation) * static_cast<double>(delay))));
}

namespace detail {
inline void validate_cic_params(int ratio, int stages, int delay, const char* name) {
	if (ratio < 1)
		throw std::invalid_argument(std::string(name) + ": ratio must be >= 1");
	if (stages < 1)
		throw std::invalid_argument(std::string(name) + ": number of stages must be >= 1");
	if (delay < 1)
		throw std::invalid_argument(std::string(name) + ": differential delay must be >= 1");
}
} // namespace detail

// CIC decimation filter
//
// Structure: M integrator stages (full rate) -> downsample by R -> M comb stages (decimated rate)
//
// The integrator sections run at the input sample rate, accumulating
// every sample. After R input samples, one output is produced by
// passing the integrator output through M comb (differencing) stages.
template <DspScalar StateScalar = double,
          DspScalar SampleScalar = StateScalar>
class CICDecimator {
public:
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;

	CICDecimator(int decimation_ratio, int num_stages, int differential_delay = 1)
		: R_((detail::validate_cic_params(decimation_ratio, num_stages,
		       differential_delay, "CICDecimator"), decimation_ratio)),
		  M_(num_stages), D_(differential_delay),
		  integrators_(static_cast<std::size_t>(num_stages), StateScalar{}),
		  comb_state_(static_cast<std::size_t>(num_stages), StateScalar{}),
		  comb_delay_(static_cast<std::size_t>(num_stages) *
		              static_cast<std::size_t>(differential_delay), StateScalar{}),
		  comb_write_(static_cast<std::size_t>(num_stages), 0),
		  count_(0) {
	}

	// Feed one input sample. Returns true when an output is ready.
	bool push(SampleScalar in) {
		StateScalar x = static_cast<StateScalar>(in);

		// Integrator cascade (runs at full rate)
		for (std::size_t i = 0; i < static_cast<std::size_t>(M_); ++i) {
			integrators_[i] = integrators_[i] + x;
			x = integrators_[i];
		}

		++count_;
		if (count_ < R_) return false;
		count_ = 0;

		// Comb cascade (runs at decimated rate)
		for (std::size_t i = 0; i < static_cast<std::size_t>(M_); ++i) {
			std::size_t base = i * static_cast<std::size_t>(D_);
			std::size_t wi = comb_write_[i];
			StateScalar delayed = comb_delay_[base + wi];
			comb_delay_[base + wi] = x;
			comb_write_[i] = (wi + 1) % static_cast<std::size_t>(D_);
			x = x - delayed;
		}

		last_output_ = x;
		return true;
	}

	// Retrieve the most recent output (valid after push() returns true)
	SampleScalar output() const {
		return static_cast<SampleScalar>(last_output_);
	}

	// Process a block of input samples, appending decimated output
	void process_block(std::span<const SampleScalar> input,
	                   std::vector<SampleScalar>& output) {
		for (auto s : input) {
			if (push(s)) {
				output.push_back(this->output());
			}
		}
	}

	// Dense-vector overload for signal containers
	mtl::vec::dense_vector<SampleScalar> process_block(
			const mtl::vec::dense_vector<SampleScalar>& input) {
		std::vector<SampleScalar> tmp;
		tmp.reserve(input.size() / static_cast<std::size_t>(R_) + 1);
		for (std::size_t i = 0; i < input.size(); ++i) {
			if (push(input[i])) tmp.push_back(this->output());
		}
		mtl::vec::dense_vector<SampleScalar> result(tmp.size());
		for (std::size_t i = 0; i < tmp.size(); ++i) result[i] = tmp[i];
		return result;
	}

	// Reset all internal state
	void reset() {
		for (auto& v : integrators_) v = StateScalar{};
		for (auto& v : comb_state_)  v = StateScalar{};
		for (auto& v : comb_delay_)  v = StateScalar{};
		for (auto& v : comb_write_)  v = 0;
		count_ = 0;
		last_output_ = StateScalar{};
	}

	int decimation_ratio()    const { return R_; }
	int num_stages()          const { return M_; }
	int differential_delay()  const { return D_; }
	int bit_growth()          const { return cic_bit_growth(M_, R_, D_); }

	// DC gain = (R * D)^M
	double dc_gain() const {
		return std::pow(static_cast<double>(R_) * static_cast<double>(D_),
		                static_cast<double>(M_));
	}

private:
	int R_;  // decimation ratio
	int M_;  // number of stages
	int D_;  // differential delay

	std::vector<StateScalar> integrators_;
	std::vector<StateScalar> comb_state_;
	std::vector<StateScalar> comb_delay_;
	std::vector<std::size_t> comb_write_;
	int count_;
	StateScalar last_output_{};
};

// CIC interpolation filter
//
// Structure: M comb stages (low rate) -> upsample by R -> M integrator stages (high rate)
//
// For each low-rate input sample, R output samples are produced.
// The comb stages run at the low rate, then R-1 zeros are inserted,
// and the integrator cascade smooths the result at the high output rate.
template <DspScalar StateScalar = double,
          DspScalar SampleScalar = StateScalar>
class CICInterpolator {
public:
	using state_scalar  = StateScalar;
	using sample_scalar = SampleScalar;

	CICInterpolator(int interpolation_ratio, int num_stages, int differential_delay = 1)
		: R_((detail::validate_cic_params(interpolation_ratio, num_stages,
		       differential_delay, "CICInterpolator"), interpolation_ratio)),
		  M_(num_stages), D_(differential_delay),
		  integrators_(static_cast<std::size_t>(num_stages), StateScalar{}),
		  comb_delay_(static_cast<std::size_t>(num_stages) *
		              static_cast<std::size_t>(differential_delay), StateScalar{}),
		  comb_write_(static_cast<std::size_t>(num_stages), 0),
		  phase_(0) {
	}

	// Feed one low-rate input sample. Call output() R times to get interpolated samples.
	void push(SampleScalar in) {
		// Comb cascade (runs at low rate)
		StateScalar x = static_cast<StateScalar>(in);
		for (std::size_t i = 0; i < static_cast<std::size_t>(M_); ++i) {
			std::size_t base = i * static_cast<std::size_t>(D_);
			std::size_t wi = comb_write_[i];
			StateScalar delayed = comb_delay_[base + wi];
			comb_delay_[base + wi] = x;
			comb_write_[i] = (wi + 1) % static_cast<std::size_t>(D_);
			x = x - delayed;
		}
		comb_output_ = x;
		phase_ = 0;
	}

	// Get next interpolated output sample. Call R times per push().
	SampleScalar output() {
		StateScalar x = (phase_ == 0) ? comb_output_ : StateScalar{};

		// Integrator cascade (runs at high rate)
		for (std::size_t i = 0; i < static_cast<std::size_t>(M_); ++i) {
			integrators_[i] = integrators_[i] + x;
			x = integrators_[i];
		}

		++phase_;
		return static_cast<SampleScalar>(x);
	}

	// Process a block: for each input sample, produce R output samples
	void process_block(std::span<const SampleScalar> input,
	                   std::vector<SampleScalar>& out) {
		for (auto s : input) {
			push(s);
			for (int r = 0; r < R_; ++r) {
				out.push_back(output());
			}
		}
	}

	// Dense-vector overload for signal containers
	mtl::vec::dense_vector<SampleScalar> process_block(
			const mtl::vec::dense_vector<SampleScalar>& input) {
		std::size_t out_size = input.size() * static_cast<std::size_t>(R_);
		mtl::vec::dense_vector<SampleScalar> result(out_size);
		std::size_t idx = 0;
		for (std::size_t i = 0; i < input.size(); ++i) {
			push(input[i]);
			for (int r = 0; r < R_; ++r) result[idx++] = output();
		}
		return result;
	}

	void reset() {
		for (auto& v : integrators_) v = StateScalar{};
		for (auto& v : comb_delay_)  v = StateScalar{};
		for (auto& v : comb_write_)  v = 0;
		phase_ = 0;
		comb_output_ = StateScalar{};
	}

	int interpolation_ratio()  const { return R_; }
	int num_stages()           const { return M_; }
	int differential_delay()   const { return D_; }
	int bit_growth()           const { return cic_bit_growth(M_, R_, D_); }

	// The z-transform DC gain is (R*D)^M, same as the decimator.
	// To normalize interpolator output, divide by (R*D)^M.
	double dc_gain() const {
		return std::pow(static_cast<double>(R_) * static_cast<double>(D_),
		                static_cast<double>(M_));
	}

private:
	int R_;
	int M_;
	int D_;

	std::vector<StateScalar> integrators_;
	std::vector<StateScalar> comb_delay_;
	std::vector<std::size_t> comb_write_;
	int phase_;
	StateScalar comb_output_{};
};

} // namespace sw::dsp
