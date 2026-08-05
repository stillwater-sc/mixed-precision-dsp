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

#include <sw/dsp/sdr/constellation.hpp>
