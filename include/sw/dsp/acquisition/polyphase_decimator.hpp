#pragma once
// polyphase_decimator.hpp: high-rate data acquisition entry point for the
// general-purpose polyphase FIR decimator.
//
// The PolyphaseDecimator class itself lives in the FIR module
// (<sw/dsp/filter/fir/polyphase.hpp>) because it's a general multirate
// FIR primitive — used both by the data-acquisition pipeline (DDC,
// DecimationChain) and by anyone designing FIR multirate flows
// directly. This header makes it discoverable via the acquisition
// module's namespace and adds documentation specific to the
// high-rate-data-acquisition use case.
//
// Polyphase decimation by M decomposes an N-tap FIR prototype into M
// sub-filters of length ceil(N/M). Each sub-filter advances once per
// output sample (not once per input sample), so the multiplier
// budget is ~N mults per output instead of ~N*M for the naive
// filter-then-downsample. For the high-rate data-acquisition chain
// (CIC -> half-band -> polyphase FIR -> baseband), this is the
// channel-shaping stage at the lowest sample rate where the
// arithmetic cost is amortized over fewer outputs.
//
// Three-scalar parameterization:
//   CoeffScalar  - tap coefficients (designed in high precision)
//   StateScalar  - delay-line accumulation precision
//   SampleScalar - input/output samples
//
// Public design helper polyphase_decompose<T>(taps, factor) returns
// the M sub-tap arrays for inspection or external sub-filter use.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/filter/fir/polyphase.hpp>

namespace sw::dsp {
// PolyphaseDecimator and PolyphaseInterpolator are defined in
// <sw/dsp/filter/fir/polyphase.hpp> and are visible here via the
// transitive include — re-stating them as `using` declarations would
// add an extra name without functional benefit. Documentation
// referring to "the acquisition polyphase decimator" should point
// readers at PolyphaseDecimator from this header.
} // namespace sw::dsp
