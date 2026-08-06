#pragma once
// ofdm.hpp: OFDM modulation and demodulation.
//
// OFDM carries data on N orthogonal subcarriers by treating the frequency
// domain as the thing being transmitted: load the subcarriers, inverse
// transform to get a time-domain symbol, send it. The receiver transforms
// back and reads the subcarriers off directly.
//
// THE CYCLIC PREFIX IS THE WHOLE TRICK. Copying the tail of each time-domain
// symbol to its front makes the channel's LINEAR convolution look CIRCULAR
// over the retained window, and circular convolution is per-bin
// multiplication in the frequency domain. A multipath channel that would
// otherwise smear symbols into each other becomes one complex gain per
// subcarrier, which a single division undoes. That is why equalizing OFDM is
// a divide rather than an adaptive filter — and why the prefix must be at
// least as long as the channel's impulse response, a condition this header
// cannot check and the caller must respect.
//
// SUBCARRIER LAYOUT. Index 0 is DC and is always nulled: it sits at the
// receiver's LO leakage and DC offset. `guard_subcarriers` further nulls a
// band centred on index N/2, the folding edge, giving the analog filters room
// to roll off. What remains is active, and every `pilot_spacing`-th active
// subcarrier carries a known reference instead of data.
//
// CHANNEL ESTIMATION is least-squares at the pilots — H = Y/P, which for a
// unit-magnitude pilot is just Y times the conjugate — then linearly
// interpolated across the data subcarriers between them. Outside the outermost
// pilots the nearest estimate is held rather than extrapolated, since
// extrapolating a channel estimate is how an edge subcarrier acquires
// confident nonsense.
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
#include <sw/dsp/spectral/fft.hpp>

namespace sw::dsp::sdr {

struct OfdmConfig {
	std::size_t fft_size          = 64;   // must be a power of two
	std::size_t cyclic_prefix     = 16;   // samples; >= channel impulse length
	std::size_t guard_subcarriers = 8;    // nulled band centred on index N/2
	std::size_t pilot_spacing     = 8;    // every Nth active subcarrier
};

// Which subcarriers carry what. Computed once from a config and shared by the
// modulator and demodulator, so the two cannot disagree about the layout.
class OfdmLayout {
public:
	explicit OfdmLayout(const OfdmConfig& cfg) : cfg_(cfg) {
		validate(cfg_);
		const std::size_t N = cfg_.fft_size;
		const std::size_t half_guard = cfg_.guard_subcarriers / 2;
		for (std::size_t k = 1; k < N; ++k) {          // index 0 is DC, nulled
			// Null a band centred on N/2 — the spectral fold point.
			const std::size_t d = (k > N / 2) ? (k - N / 2) : (N / 2 - k);
			if (d <= half_guard) continue;
			active_.push_back(k);
		}
		if (active_.empty())
			throw std::invalid_argument(
				"OfdmLayout: guard_subcarriers leaves no active subcarriers");
		for (std::size_t i = 0; i < active_.size(); ++i) {
			if (i % cfg_.pilot_spacing == 0) pilot_.push_back(active_[i]);
			else                              data_.push_back(active_[i]);
		}
		if (pilot_.size() < 2)
			throw std::invalid_argument(
				"OfdmLayout: need at least 2 pilots to interpolate a channel "
				"estimate; reduce pilot_spacing or guard_subcarriers");
		if (data_.empty())
			throw std::invalid_argument("OfdmLayout: no data subcarriers");
	}

	const OfdmConfig& config() const { return cfg_; }
	const std::vector<std::size_t>& active() const { return active_; }
	const std::vector<std::size_t>& data() const { return data_; }
	const std::vector<std::size_t>& pilots() const { return pilot_; }

	std::size_t num_data() const { return data_.size(); }
	std::size_t num_pilots() const { return pilot_.size(); }
	// Time-domain samples per OFDM symbol, prefix included.
	std::size_t symbol_length() const { return cfg_.fft_size + cfg_.cyclic_prefix; }

private:
	static void validate(const OfdmConfig& c) {
		if (c.fft_size < 8 || (c.fft_size & (c.fft_size - 1)))
			throw std::invalid_argument(
				"OfdmConfig: fft_size must be a power of two and >= 8");
		if (c.cyclic_prefix == 0 || c.cyclic_prefix > c.fft_size)
			throw std::invalid_argument(
				"OfdmConfig: cyclic_prefix must be in (0, fft_size]");
		if (c.guard_subcarriers >= c.fft_size)
			throw std::invalid_argument(
				"OfdmConfig: guard_subcarriers must be < fft_size");
		if (c.pilot_spacing < 2)
			throw std::invalid_argument(
				"OfdmConfig: pilot_spacing must be >= 2, otherwise every "
				"subcarrier is a pilot and none carries data");
	}
	OfdmConfig cfg_;
	std::vector<std::size_t> active_, data_, pilot_;
};

// ============================================================================
// Modulator
// ============================================================================

template <DspField StateScalar = double,
          typename SampleScalar = complex_for_t<StateScalar>>
class OfdmModulator {
public:
	using complex_state = complex_for_t<StateScalar>;

	explicit OfdmModulator(const OfdmConfig& cfg)
		: layout_(cfg),
		  pilot_(static_cast<StateScalar>(1), static_cast<StateScalar>(0)) {}

	const OfdmLayout& layout() const { return layout_; }
	// The reference every pilot subcarrier carries.
	complex_state pilot_symbol() const { return pilot_; }

	// One OFDM symbol: num_data() constellation points in, symbol_length()
	// time samples out (cyclic prefix first).
	//
	// Pre:  data.size() == layout().num_data().
	mtl::vec::dense_vector<SampleScalar>
	modulate(std::span<const SampleScalar> data) {
		if (data.size() != layout_.num_data())
			throw std::invalid_argument(
				"OfdmModulator::modulate: expected " +
				std::to_string(layout_.num_data()) + " data symbols, got " +
				std::to_string(data.size()));

		const std::size_t N = layout_.config().fft_size;
		mtl::vec::dense_vector<complex_state> grid(N, complex_state{});
		for (std::size_t i = 0; i < layout_.data().size(); ++i)
			grid[layout_.data()[i]] =
				complex_state(static_cast<StateScalar>(data[i].real()),
				              static_cast<StateScalar>(data[i].imag()));
		for (std::size_t k : layout_.pilots()) grid[k] = pilot_;

		sw::dsp::spectral::fft_inverse<StateScalar>(grid);

		// Cyclic prefix: the last cp samples, repeated in front.
		const std::size_t cp = layout_.config().cyclic_prefix;
		mtl::vec::dense_vector<SampleScalar> out(N + cp);
		for (std::size_t i = 0; i < cp; ++i)
			out[i] = to_sample(grid[N - cp + i]);
		for (std::size_t i = 0; i < N; ++i)
			out[cp + i] = to_sample(grid[i]);
		return out;
	}

private:
	static SampleScalar to_sample(const complex_state& c) {
		using C = std::remove_cvref_t<decltype(SampleScalar{}.real())>;
		return SampleScalar(static_cast<C>(c.real()), static_cast<C>(c.imag()));
	}
	OfdmLayout    layout_;
	complex_state pilot_;
};

// ============================================================================
// Demodulator
// ============================================================================

template <DspField StateScalar = double,
          typename SampleScalar = complex_for_t<StateScalar>>
class OfdmDemodulator {
public:
	using complex_state = complex_for_t<StateScalar>;

	explicit OfdmDemodulator(const OfdmConfig& cfg)
		: layout_(cfg),
		  pilot_(static_cast<StateScalar>(1), static_cast<StateScalar>(0)),
		  channel_(cfg.fft_size, complex_state{}) {}

	const OfdmLayout& layout() const { return layout_; }

	// The channel estimate from the most recent symbol, one entry per
	// subcarrier. Zero on nulled subcarriers.
	const mtl::vec::dense_vector<complex_state>& channel_estimate() const {
		return channel_;
	}

	// One OFDM symbol in, num_data() equalized constellation points out.
	//
	// Pre:  symbol.size() == layout().symbol_length().
	// Post: the channel estimate is refreshed from this symbol's pilots.
	mtl::vec::dense_vector<SampleScalar>
	demodulate(std::span<const SampleScalar> symbol) {
		if (symbol.size() != layout_.symbol_length())
			throw std::invalid_argument(
				"OfdmDemodulator::demodulate: expected " +
				std::to_string(layout_.symbol_length()) + " samples, got " +
				std::to_string(symbol.size()));

		const std::size_t N  = layout_.config().fft_size;
		const std::size_t cp = layout_.config().cyclic_prefix;

		// Drop the prefix — its job is done once the channel has convolved.
		mtl::vec::dense_vector<complex_state> grid(N);
		for (std::size_t i = 0; i < N; ++i)
			grid[i] = complex_state(static_cast<StateScalar>(symbol[cp + i].real()),
			                        static_cast<StateScalar>(symbol[cp + i].imag()));
		sw::dsp::spectral::fft_forward<StateScalar>(grid);

		estimate_channel(grid);

		mtl::vec::dense_vector<SampleScalar> out(layout_.num_data());
		for (std::size_t i = 0; i < layout_.data().size(); ++i) {
			const std::size_t k = layout_.data()[i];
			out[i] = to_sample(divide(grid[k], channel_[k]));
		}
		return out;
	}

private:
	// Least squares at the pilots, linear interpolation between them, nearest
	// held outside. H = Y/P.
	void estimate_channel(const mtl::vec::dense_vector<complex_state>& grid) {
		const auto& p = layout_.pilots();
		std::vector<complex_state> hp(p.size());
		for (std::size_t i = 0; i < p.size(); ++i)
			hp[i] = divide(grid[p[i]], pilot_);

		for (std::size_t k = 0; k < channel_.size(); ++k)
			channel_[k] = complex_state{};

		// Between consecutive pilots, interpolate on subcarrier index.
		for (std::size_t i = 0; i + 1 < p.size(); ++i) {
			const std::size_t a = p[i], b = p[i + 1];
			const StateScalar span = static_cast<StateScalar>(b - a);
			for (std::size_t k = a; k <= b; ++k) {
				const StateScalar t = static_cast<StateScalar>(k - a) / span;
				channel_[k] = complex_state(
					hp[i].real() + (hp[i + 1].real() - hp[i].real()) * t,
					hp[i].imag() + (hp[i + 1].imag() - hp[i].imag()) * t);
			}
		}
		// Hold the nearest estimate outside the pilot span rather than
		// extrapolate — extrapolation is how an edge subcarrier acquires a
		// confident wrong answer.
		for (std::size_t k = 0; k < p.front(); ++k)      channel_[k] = hp.front();
		for (std::size_t k = p.back() + 1; k < channel_.size(); ++k)
			channel_[k] = hp.back();
	}

	static complex_state divide(const complex_state& a, const complex_state& b) {
		const StateScalar d = b.real() * b.real() + b.imag() * b.imag();
		if (!(d > StateScalar{})) return complex_state{};
		return complex_state((a.real() * b.real() + a.imag() * b.imag()) / d,
		                     (a.imag() * b.real() - a.real() * b.imag()) / d);
	}
	static SampleScalar to_sample(const complex_state& c) {
		using C = std::remove_cvref_t<decltype(SampleScalar{}.real())>;
		return SampleScalar(static_cast<C>(c.real()), static_cast<C>(c.imag()));
	}

	OfdmLayout    layout_;
	complex_state pilot_;
	mtl::vec::dense_vector<complex_state> channel_;
};

// ============================================================================
// PAPR
// ============================================================================

// Peak-to-average power ratio of a time-domain block, in dB.
//
// OFDM's characteristic weakness: summing many independent subcarriers gives
// a near-Gaussian time signal whose peaks run far above its mean power, which
// is what forces a linear power amplifier to be backed off. Grows roughly as
// 10*log10(N) in the worst case.
template <typename SampleScalar>
double papr_db(std::span<const SampleScalar> block) {
	if (block.empty())
		throw std::invalid_argument("papr_db: empty block");
	double peak = 0.0, sum = 0.0;
	for (const auto& s : block) {
		const double re = static_cast<double>(s.real());
		const double im = static_cast<double>(s.imag());
		const double p = re * re + im * im;
		peak = std::max(peak, p);
		sum += p;
	}
	const double mean = sum / static_cast<double>(block.size());
	if (!(mean > 0.0))
		throw std::invalid_argument("papr_db: block has zero average power");
	return 10.0 * std::log10(peak / mean);
}

} // namespace sw::dsp::sdr
