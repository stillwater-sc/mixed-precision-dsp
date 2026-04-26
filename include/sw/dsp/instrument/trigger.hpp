#pragma once
// trigger.hpp: Trigger primitives for instrument-style data acquisition.
//
// A composable library of streaming primitives that turn a stream of samples
// into a stream of trigger events. Foundational building block for both
// the digital oscilloscope (#133) and the spectrum analyzer (#134, zero-span
// mode and sweep-control gating) demonstrators.
//
// Primitives:
//   EdgeTrigger     — rising / falling / either crossing of a level
//                     with optional hysteresis to suppress noise re-triggers
//   LevelTrigger    — convenience for EdgeTrigger with Slope::Either
//   SlopeTrigger    — fires when |x[n] - x[n-1]| exceeds a threshold
//                     with a sign (positive / negative / either)
//   HoldoffWrapper  — wraps any trigger and suppresses re-triggering for
//                     N samples after a fire (the inner trigger is still
//                     driven during the holdoff so its state stays sane)
//   AutoTriggerWrapper — wraps any trigger and force-fires after N samples
//                     of inactivity. A real inner fire preempts the auto-fire
//                     and resets the timeout counter.
//   QualifierAnd    — fires when two triggers both fire within a window
//   QualifierOr     — fires when either of two triggers fires
//   CrossChannelTrigger — multi-channel scope-style trigger composing two
//                     channels via one of seven named modes (OnlyA / OnlyB
//                     / AandB / AorB / AxorB / AnotB / BnotA), with a
//                     configurable inter-channel coincidence window.
//
// All primitives are parameterized on a single SampleScalar type. Comparisons
// are precision-sensitive at low signal amplitudes — narrow types may
// quantize the comparison threshold below the smallest detectable edge.
// The tests exercise this characterization explicitly.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp::instrument {

// Direction of a trigger fire.
//
// `Rising`  fires only when the signal crosses up through the level
// `Falling` fires only when the signal crosses down through the level
// `Either`  fires on both directions
enum class Slope { Rising, Falling, Either };

// =============================================================================
// EdgeTrigger
//
// Tracks whether the signal is currently in the "above" or "below" region,
// where the boundary is `level ± hysteresis/2`. A fire happens when the signal
// crosses out of one region and into the other (with the slope filter applied).
// Hysteresis = 0 means the level itself is the boundary.
// =============================================================================
template <DspScalar SampleScalar>
class EdgeTrigger {
public:
	using sample_scalar = SampleScalar;

	EdgeTrigger(SampleScalar level,
	            Slope        slope     = Slope::Rising,
	            SampleScalar hysteresis = SampleScalar{0})
		: level_(level),
		  slope_(slope),
		  hysteresis_(hysteresis),
		  upper_(level + half(hysteresis)),
		  lower_(level - half(hysteresis)),
		  state_(Region::Unknown),
		  samples_since_(0) {
		if (hysteresis < SampleScalar{0})
			throw std::invalid_argument("EdgeTrigger: hysteresis must be >= 0");
	}

	// Push one sample. Returns true on this sample if it caused a trigger event.
	bool process(SampleScalar x) {
		++samples_since_;

		// Determine which side of the hysteresis band we're in.
		Region next = state_;
		if (x > upper_)       next = Region::Above;
		else if (x < lower_)  next = Region::Below;
		// If we're inside the band, state is unchanged (this is the
		// hysteresis behavior — small noise excursions don't flip state).

		const bool rose  = (state_ == Region::Below)   && (next == Region::Above);
		const bool fell  = (state_ == Region::Above)   && (next == Region::Below);
		state_ = next;

		bool fire = false;
		switch (slope_) {
			case Slope::Rising:  fire = rose;          break;
			case Slope::Falling: fire = fell;          break;
			case Slope::Either:  fire = rose || fell;  break;
		}
		if (fire) samples_since_ = 0;
		return fire;
	}

	// Number of samples processed since the last fire (or since construction
	// if no fires yet). Reset to 0 when a fire occurs.
	std::size_t samples_since_trigger() const { return samples_since_; }

	void reset() {
		state_ = Region::Unknown;
		samples_since_ = 0;
	}

private:
	enum class Region { Unknown, Above, Below };

	// Compute hysteresis/2 in SampleScalar without requiring division
	// (DspScalar doesn't include /). Multiplication by 0.5 works through
	// the SampleScalar(double) ctor for every supported numeric type.
	static SampleScalar half(SampleScalar h) {
		return h * static_cast<SampleScalar>(0.5);
	}

	SampleScalar level_;
	Slope        slope_;
	SampleScalar hysteresis_;
	SampleScalar upper_;
	SampleScalar lower_;
	Region       state_;
	std::size_t  samples_since_;
};

// =============================================================================
// LevelTrigger — EdgeTrigger with Slope::Either, named for ergonomics
// =============================================================================
template <DspScalar SampleScalar>
class LevelTrigger {
public:
	using sample_scalar = SampleScalar;

	LevelTrigger(SampleScalar level,
	             SampleScalar hysteresis = SampleScalar{0})
		: edge_(level, Slope::Either, hysteresis) {}

	bool        process(SampleScalar x)         { return edge_.process(x); }
	std::size_t samples_since_trigger() const   { return edge_.samples_since_trigger(); }
	void        reset()                         { edge_.reset(); }

private:
	EdgeTrigger<SampleScalar> edge_;
};

// =============================================================================
// SlopeTrigger
//
// Fires when the first difference x[n] - x[n-1] exceeds threshold in magnitude
// with the requested sign. Useful for detecting fast transitions independent
// of absolute level.
// =============================================================================
template <DspScalar SampleScalar>
class SlopeTrigger {
public:
	using sample_scalar = SampleScalar;

	SlopeTrigger(SampleScalar threshold, Slope sign = Slope::Rising)
		: threshold_(threshold),
		  sign_(sign),
		  prev_(SampleScalar{0}),
		  has_prev_(false),
		  samples_since_(0) {
		if (threshold < SampleScalar{0})
			throw std::invalid_argument("SlopeTrigger: threshold must be >= 0");
	}

	bool process(SampleScalar x) {
		++samples_since_;
		if (!has_prev_) {
			prev_     = x;
			has_prev_ = true;
			return false;
		}
		const SampleScalar diff = x - prev_;
		prev_ = x;

		const bool pos_fire = (diff >  threshold_);
		const bool neg_fire = (diff < -threshold_);

		bool fire = false;
		switch (sign_) {
			case Slope::Rising:  fire = pos_fire;            break;
			case Slope::Falling: fire = neg_fire;            break;
			case Slope::Either:  fire = pos_fire || neg_fire; break;
		}
		if (fire) samples_since_ = 0;
		return fire;
	}

	std::size_t samples_since_trigger() const { return samples_since_; }

	void reset() {
		has_prev_      = false;
		prev_          = SampleScalar{0};
		samples_since_ = 0;
	}

private:
	SampleScalar threshold_;
	Slope        sign_;
	SampleScalar prev_;
	bool         has_prev_;
	std::size_t  samples_since_;
};

// =============================================================================
// HoldoffWrapper
//
// Wraps any trigger primitive and suppresses re-triggering for `holdoff`
// samples after a fire. The inner trigger is STILL driven during the holdoff
// window so its internal state (e.g., the prev-sample for SlopeTrigger or the
// region state for EdgeTrigger) stays consistent — otherwise the very next
// sample after holdoff would see a stale internal state and potentially fire
// spuriously.
// =============================================================================
template <class Inner>
class HoldoffWrapper {
public:
	using sample_scalar = typename Inner::sample_scalar;

	HoldoffWrapper(Inner inner, std::size_t holdoff_samples)
		: inner_(std::move(inner)),
		  holdoff_(holdoff_samples),
		  remaining_(0),
		  samples_since_(0) {}

	bool process(sample_scalar x) {
		++samples_since_;

		// During holdoff: drive inner for state continuity, suppress fire.
		if (remaining_ > 0) {
			--remaining_;
			(void)inner_.process(x);
			return false;
		}

		if (inner_.process(x)) {
			remaining_     = holdoff_;
			samples_since_ = 0;
			return true;
		}
		return false;
	}

	std::size_t samples_since_trigger() const { return samples_since_; }

	void reset() {
		inner_.reset();
		remaining_     = 0;
		samples_since_ = 0;
	}

private:
	Inner       inner_;
	std::size_t holdoff_;
	std::size_t remaining_;
	std::size_t samples_since_;
};

// =============================================================================
// AutoTriggerWrapper
//
// Wraps any trigger primitive and force-fires after `timeout_samples` of
// inactivity if no real fire has occurred. Without this, a scope viewing
// a slow or absent signal would sit blank forever waiting for the trigger
// condition; with it, the user always sees something — a recent capture,
// even if no edge crossed the threshold.
//
// The inner trigger is always driven so its internal state stays sane.
// A real fire from the inner trigger preempts any pending auto-fire and
// resets the timeout counter; an auto-fire on timeout also resets the
// counter, so on a flat signal auto-fires happen at regular intervals
// (one every `timeout_samples` samples after the first one).
// =============================================================================
template <class Inner>
class AutoTriggerWrapper {
public:
	using sample_scalar = typename Inner::sample_scalar;

	AutoTriggerWrapper(Inner inner, std::size_t timeout_samples)
		: inner_(std::move(inner)),
		  timeout_(timeout_samples),
		  since_fire_(0) {
		// timeout=0 would force-fire on every sample (since_fire_ becomes
		// 1 after the first increment, which is >= 0). That's degenerate
		// and almost certainly a user error; reject it. Use a sufficiently
		// large value (e.g., std::numeric_limits<std::size_t>::max()) if
		// effectively-no-timeout behavior is wanted.
		if (timeout_samples == 0)
			throw std::invalid_argument(
				"AutoTriggerWrapper: timeout_samples must be > 0");
	}

	bool process(sample_scalar x) {
		// Always drive inner so its state is current.
		const bool real_fire = inner_.process(x);
		++since_fire_;

		if (real_fire) {
			since_fire_ = 0;
			return true;
		}
		// No real fire — has the timeout elapsed?
		if (since_fire_ >= timeout_) {
			since_fire_ = 0;
			return true;   // forced auto-fire
		}
		return false;
	}

	std::size_t samples_since_trigger() const { return since_fire_; }

	void reset() {
		inner_.reset();
		since_fire_ = 0;
	}

private:
	Inner       inner_;
	std::size_t timeout_;
	std::size_t since_fire_;
};

// =============================================================================
// QualifierAnd
//
// Fires when both wrapped triggers have fired within `window` samples of
// each other. The window is the maximum age (in samples) of an A fire that
// can still be "joined" by a fresh B fire (or vice versa).
//
// On an AND fire, both triggers' age counters reset, so the same coincidence
// can't fire twice.
// =============================================================================
template <class TrigA, class TrigB>
class QualifierAnd {
public:
	using sample_a = typename TrigA::sample_scalar;
	using sample_b = typename TrigB::sample_scalar;

	QualifierAnd(TrigA a, TrigB b, std::size_t window_samples = 1)
		: a_(std::move(a)),
		  b_(std::move(b)),
		  window_(window_samples),
		  since_a_(kInf),
		  since_b_(kInf) {
		// kInf is the internal "never fired" sentinel. window_samples ==
		// kInf would make `since_X <= window_` trivially true for the
		// sentinel, firing AandB before either inner trigger has fired.
		// Reject explicitly. Same guard pattern as CrossChannelTrigger.
		if (window_samples == kInf)
			throw std::invalid_argument(
				"QualifierAnd: window_samples must be < "
				"std::numeric_limits<std::size_t>::max() "
				"(use max() - 1 for an effectively-unlimited window)");
	}

	bool process(sample_a xa, sample_b xb) {
		if (since_a_ != kInf && since_a_ < kInf - 1) ++since_a_;
		if (since_b_ != kInf && since_b_ < kInf - 1) ++since_b_;

		if (a_.process(xa)) since_a_ = 0;
		if (b_.process(xb)) since_b_ = 0;

		if (since_a_ <= window_ && since_b_ <= window_) {
			since_a_ = kInf;
			since_b_ = kInf;
			return true;
		}
		return false;
	}

	void reset() {
		a_.reset();
		b_.reset();
		since_a_ = kInf;
		since_b_ = kInf;
	}

private:
	static constexpr std::size_t kInf = std::numeric_limits<std::size_t>::max();
	TrigA       a_;
	TrigB       b_;
	std::size_t window_;
	std::size_t since_a_;
	std::size_t since_b_;
};

// =============================================================================
// QualifierOr — fires whenever either inner trigger fires on the same sample.
// =============================================================================
template <class TrigA, class TrigB>
class QualifierOr {
public:
	using sample_a = typename TrigA::sample_scalar;
	using sample_b = typename TrigB::sample_scalar;

	QualifierOr(TrigA a, TrigB b)
		: a_(std::move(a)), b_(std::move(b)) {}

	bool process(sample_a xa, sample_b xb) {
		const bool fa = a_.process(xa);
		const bool fb = b_.process(xb);
		return fa || fb;
	}

	void reset() {
		a_.reset();
		b_.reset();
	}

private:
	TrigA a_;
	TrigB b_;
};

// =============================================================================
// CrossChannelTrigger
//
// Multi-channel scope-style trigger: composes two channel triggers via one
// of seven named modes. Each `process` call takes one synchronized sample
// from each channel.
//
// Modes:
//   OnlyA  — fire when channel A's inner trigger fires (B is ignored but
//            still driven so its state stays current)
//   OnlyB  — symmetric
//   AandB  — coincidence: fire when both A and B fire within `window` samples
//            of each other. Both `since_` counters reset to "never" on fire so
//            the same coincidence can't fire twice.
//   AorB   — fire when either A or B fires this sample
//   AxorB  — fire when exactly one of A/B is active in the window: A fires
//            with no recent B, OR B fires with no recent A. Same-sample
//            (A and B both fire) does NOT fire (both are "active"). No
//            counter reset — repeated solo events keep firing.
//   AnotB  — qualified A: A fires AND B has not fired in the past `window`
//            samples. (Useful for "trigger on data line edge but not when
//            clock is high".)
//   BnotA  — symmetric.
//
// `window_samples` is the maximum allowed age (in samples) of a prior
// fire on the OTHER channel for it to count as "recent". The check is
// inclusive: `since_other <= window_samples` means "still recent."
//
//   window_samples = 0 → only same-sample (both fires must land on the
//                         same `process()` call, since=0 for both)
//   window_samples = 1 (default) → admits same-sample AND one-sample-apart
//   window_samples = N → admits ages 0..N inclusive
//
// `window_samples == std::numeric_limits<std::size_t>::max()` is rejected
// because it collides with the internal "never fired" sentinel (would
// cause AandB/AnotB/etc. to fire spuriously before any inner trigger has
// fired). Callers wanting an effectively-unlimited window should use
// `max() - 1` instead.
//
// Sample synchronization: this wrapper assumes channels A and B are sample-
// aligned (same rate, same time origin). Sub-sample alignment between
// channels is a separate concern handled by #150 (multi-channel time
// alignment).
//
// The two inner triggers can have different SampleScalar types (e.g.,
// channel A in float, channel B in fixpnt) — each side is templated
// independently.
// =============================================================================

enum class CrossChannelMode {
	OnlyA,
	OnlyB,
	AandB,
	AorB,
	AxorB,
	AnotB,
	BnotA,
};

template <class TrigA, class TrigB>
class CrossChannelTrigger {
public:
	using sample_a = typename TrigA::sample_scalar;
	using sample_b = typename TrigB::sample_scalar;

	CrossChannelTrigger(TrigA a, TrigB b,
	                    CrossChannelMode mode,
	                    std::size_t window_samples = 1)
		: a_(std::move(a)),
		  b_(std::move(b)),
		  mode_(mode),
		  window_(window_samples),
		  since_a_(kInf),
		  since_b_(kInf) {
		// kInf is the internal "never fired" sentinel. If a caller passes
		// kInf as window_samples, the `since_X <= window_` checks become
		// trivially true even for the sentinel and AandB/AnotB/etc. would
		// fire before any inner trigger has fired. Reject explicitly.
		if (window_samples == kInf)
			throw std::invalid_argument(
				"CrossChannelTrigger: window_samples must be < "
				"std::numeric_limits<std::size_t>::max() "
				"(use max() - 1 for an effectively-unlimited window)");
	}

	bool process(sample_a xa, sample_b xb) {
		// Increment age counters BEFORE running inner triggers, so a fire
		// this sample lands at since_X = 0.
		if (since_a_ != kInf && since_a_ < kInf - 1) ++since_a_;
		if (since_b_ != kInf && since_b_ < kInf - 1) ++since_b_;

		// Drive both inner triggers unconditionally so their state stays
		// current regardless of which channel the wrapper "cares about".
		const bool a_fired = a_.process(xa);
		const bool b_fired = b_.process(xb);
		if (a_fired) since_a_ = 0;
		if (b_fired) since_b_ = 0;

		switch (mode_) {
			case CrossChannelMode::OnlyA: return a_fired;
			case CrossChannelMode::OnlyB: return b_fired;

			case CrossChannelMode::AandB: {
				if (since_a_ <= window_ && since_b_ <= window_) {
					// Coincidence consumed — clear both so the same pair
					// can't fire repeatedly while both ages are < window.
					since_a_ = kInf;
					since_b_ = kInf;
					return true;
				}
				return false;
			}

			case CrossChannelMode::AorB:
				return a_fired || b_fired;

			case CrossChannelMode::AxorB: {
				// "Exactly one is active in the window" — A fires with no
				// recent B, or B fires with no recent A. Same-sample
				// firings of both fail both checks (since_other == 0).
				const bool a_solo = a_fired && (since_b_ > window_);
				const bool b_solo = b_fired && (since_a_ > window_);
				return a_solo || b_solo;
			}

			case CrossChannelMode::AnotB:
				return a_fired && (since_b_ > window_);

			case CrossChannelMode::BnotA:
				return b_fired && (since_a_ > window_);
		}
		return false;  // unreachable; quiet the compiler on enum coverage
	}

	void reset() {
		a_.reset();
		b_.reset();
		since_a_ = kInf;
		since_b_ = kInf;
	}

private:
	static constexpr std::size_t kInf = std::numeric_limits<std::size_t>::max();
	TrigA            a_;
	TrigB            b_;
	CrossChannelMode mode_;
	std::size_t      window_;
	std::size_t      since_a_;
	std::size_t      since_b_;
};

} // namespace sw::dsp::instrument
