#pragma once
// signal.hpp: concepts for signal containers
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <concepts>
#include <cstddef>
#include <span>
#include <sw/dsp/concepts/scalar.hpp>

namespace sw::dsp {

// A readable signal container: provides indexed access and size
template <typename C>
concept SignalContainer = requires(C c, std::size_t i) {
	{ c[i] };
	{ c.size() } -> std::convertible_to<std::size_t>;
};

// A writable signal container: also supports assignment
template <typename C>
concept MutableSignalContainer = SignalContainer<C> && requires(C c, std::size_t i) {
	{ c[i] = c[i] };
};

// A contiguous signal container: supports data() for span conversion
template <typename C>
concept ContiguousSignalContainer = SignalContainer<C> && requires(C c) {
	{ c.data() };
	{ std::span{c.data(), c.size()} };
};

} // namespace sw::dsp
