#pragma once
// rectangular.hpp: rectangular (boxcar) window
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

template <DspField T>
mtl::vec::dense_vector<T> rectangular_window(std::size_t length) {
	return mtl::vec::dense_vector<T>(length, T{1});
}

} // namespace sw::dsp
