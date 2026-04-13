#pragma once
// dsp.hpp: umbrella header for the sw::dsp library
//
// Include this single header to get the entire library.
// For faster compile times, include individual module headers instead.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

// Version
#include <sw/dsp/version.hpp>

// Concepts
#include <sw/dsp/concepts/scalar.hpp>
#include <sw/dsp/concepts/signal.hpp>
#include <sw/dsp/concepts/filter.hpp>

// Types
#include <sw/dsp/types/complex_pair.hpp>
#include <sw/dsp/types/pole_zero_pair.hpp>
#include <sw/dsp/types/biquad_coefficients.hpp>
#include <sw/dsp/types/transfer_function.hpp>
#include <sw/dsp/types/filter_kind.hpp>

// Math utilities
#include <sw/dsp/math/constants.hpp>
#include <sw/dsp/math/denormal.hpp>
#include <sw/dsp/math/quadratic.hpp>
#include <sw/dsp/math/polynomial.hpp>

// Signal generators
#include <sw/dsp/signals/signal.hpp>
#include <sw/dsp/signals/generators.hpp>
#include <sw/dsp/signals/sampling.hpp>

// Windows
#include <sw/dsp/windows/windows.hpp>

// Quantization
#include <sw/dsp/quantization/quantization.hpp>

// Filters (IIR + FIR)
#include <sw/dsp/filter/filter.hpp>
#include <sw/dsp/filter/iir/iir.hpp>
#include <sw/dsp/filter/fir/fir.hpp>

// Spectral methods
#include <sw/dsp/spectral/spectral.hpp>

// Signal conditioning
#include <sw/dsp/conditioning/conditioning.hpp>

// Estimation (Kalman, LMS, RLS)
#include <sw/dsp/estimation/estimation.hpp>

// Image processing
#include <sw/dsp/image/image.hpp>

// Numerical analysis
#include <sw/dsp/analysis/analysis.hpp>

// I/O (WAV, CSV, raw)
#include <sw/dsp/io/io.hpp>

// Visualization
#include <sw/dsp/viz/viz.hpp>
