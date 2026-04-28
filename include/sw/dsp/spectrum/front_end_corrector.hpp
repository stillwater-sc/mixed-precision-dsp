#pragma once
// front_end_corrector.hpp: spectrum-analyzer front-end calibration
// (alias for instrument::EqualizerFilter).
//
// In a spectrum analyzer the analog front end (probe + amplifier +
// ADC) imposes a non-flat magnitude response on the input. The
// front-end corrector applies the inverse of that response so the
// digital pipeline measures the source spectrum, not the
// front-end-distorted spectrum.
//
// The math is identical to what the scope demo's calibration stage
// does (the same FIR equalizer designed by frequency-sampling from a
// CalibrationProfile). What differs is the *use* of the equalizer's
// output:
//   - Scope:    cares about edge fidelity and group delay; uses the
//               equalized signal as a faithful time-domain
//               reconstruction.
//   - Analyzer: cares about magnitude flatness across frequency;
//               uses the equalized signal so the FFT bin amplitudes
//               match the source spectrum within the analyzer's
//               specifications.
//
// Since the streaming math is identical, this header is a thin
// using-alias rather than a wrapper class. Spectrum-namespace users
// get a self-documenting name (`FrontEndCorrector`) without a
// duplicated implementation. Both names refer to the same template
// instantiation; behavior is bit-identical.
//
// If the analyzer pipeline ever needs calibration semantics that
// genuinely differ from the scope's (e.g., a different design
// algorithm, magnitude-only correction skipping phase, or built-in
// group-delay compensation), promote this alias to a proper class
// at that point. Today the EqualizerFilter design covers both use
// cases.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/instrument/calibration.hpp>

namespace sw::dsp::spectrum {

// Bring CalibrationProfile into the spectrum namespace for ergonomics.
// CalibrationProfile is a non-templated value type — there's nothing
// analyzer-specific about it; the same profile representation works
// for both instrument paths (scope, analyzer).
using CalibrationProfile = sw::dsp::instrument::CalibrationProfile;

// Front-end equalizer for the spectrum-analyzer input path. Identical
// template signature to instrument::EqualizerFilter; defaults match.
template <DspField CoeffScalar  = double,
          DspField StateScalar  = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
using FrontEndCorrector =
	sw::dsp::instrument::EqualizerFilter<CoeffScalar, StateScalar, SampleScalar>;

} // namespace sw::dsp::spectrum
