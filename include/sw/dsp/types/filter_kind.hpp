#pragma once
// filter_kind.hpp: enumeration of filter response types
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <string_view>

namespace sw::dsp {

enum class FilterKind : std::uint8_t {
	low_pass,
	high_pass,
	band_pass,
	band_stop,
	low_shelf,
	high_shelf,
	band_shelf,
	all_pass,
	other
};

constexpr std::string_view to_string(FilterKind kind) {
	switch (kind) {
	case FilterKind::low_pass:   return "Low Pass";
	case FilterKind::high_pass:  return "High Pass";
	case FilterKind::band_pass:  return "Band Pass";
	case FilterKind::band_stop:  return "Band Stop";
	case FilterKind::low_shelf:  return "Low Shelf";
	case FilterKind::high_shelf: return "High Shelf";
	case FilterKind::band_shelf: return "Band Shelf";
	case FilterKind::all_pass:   return "All Pass";
	case FilterKind::other:      return "Other";
	}
	return "Unknown";
}

} // namespace sw::dsp
