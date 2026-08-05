#pragma once
// demo_output.hpp: where demonstration programs write their output.
//
// Demos default their CSV path to a bare filename, which resolves against
// the working directory — so running one from the repository root dropped
// its output into the source tree. dsp_add_application() bakes a per-build
// output directory into every demo via the DSP_DEMO_OUTPUT_DIR compile
// definition, and the helpers here turn that into a path.
//
// Callers can still override: every demo keeps its --csv=<path> flag (or
// positional output directory), and an explicit path is used verbatim.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <filesystem>
#include <string>

#ifndef DSP_DEMO_OUTPUT_DIR
// Compiling a demo outside the project's CMake (a one-off g++ invocation,
// say) leaves the definition unset; fall back to the working directory so
// the demo still runs.
#define DSP_DEMO_OUTPUT_DIR "."
#endif

namespace sw::dsp::demo {

// The directory demos write to unless the caller overrides it. Set by the
// build to <binary-dir>/demo-output.
inline std::string output_dir() {
	return std::string(DSP_DEMO_OUTPUT_DIR);
}

// Join `filename` onto the demo output directory, creating that directory
// if it is missing. The build creates it at configure time; this covers a
// build tree whose output directory was cleaned out afterwards.
//
// Directory-creation failure is deliberately swallowed — the subsequent
// ofstream open is what reports a genuine problem, with a path in the
// message, and that is a better error than one raised from here.
inline std::string output_path(const std::string& filename) {
	const std::filesystem::path dir(output_dir());
	std::error_code ec;
	std::filesystem::create_directories(dir, ec);
	return (dir / filename).string();
}

} // namespace sw::dsp::demo
