#pragma once
// sdr.hpp: umbrella header for the software-defined radio module (Epic #85)
//
// Modulation and demodulation for digital communication links. The receive
// front end — NCO, CIC, DDC, polyphase decimation — lives in
// <sw/dsp/acquisition/acquisition.hpp> and is reused rather than duplicated
// here.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/sdr/agc.hpp>
#include <sw/dsp/sdr/carrier_recovery.hpp>
#include <sw/dsp/sdr/constellation.hpp>
#include <sw/dsp/sdr/loop_filter.hpp>
#include <sw/dsp/sdr/metrics.hpp>
#include <sw/dsp/sdr/rrc.hpp>
#include <sw/dsp/sdr/timing_recovery.hpp>
