#pragma once
// raw.hpp: raw binary sample reader/writer
//
// Writes/reads sample data as direct binary representation of the
// template type T. No headers, no metadata — just samples.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <span>
#include <vector>

namespace sw::dsp::io {

// Write samples as raw binary.
template <typename T>
void write_raw(const std::string& path, std::span<const T> samples) {
	std::ofstream ofs(path, std::ios::binary);
	if (!ofs) throw std::runtime_error("raw: cannot open " + path + " for writing");

	ofs.write(reinterpret_cast<const char*>(samples.data()),
	          static_cast<std::streamsize>(samples.size() * sizeof(T)));
}

// Read count samples from a raw binary file.
template <typename T>
std::vector<T> read_raw(const std::string& path, std::size_t count) {
	std::ifstream ifs(path, std::ios::binary);
	if (!ifs) throw std::runtime_error("raw: cannot open " + path + " for reading");

	std::vector<T> result(count);
	ifs.read(reinterpret_cast<char*>(result.data()),
	         static_cast<std::streamsize>(count * sizeof(T)));

	// Resize to actual number read
	auto bytes_read = ifs.gcount();
	result.resize(static_cast<std::size_t>(bytes_read) / sizeof(T));
	return result;
}

// Read all samples from a raw binary file (deduces count from file size).
template <typename T>
std::vector<T> read_raw(const std::string& path) {
	std::ifstream ifs(path, std::ios::binary | std::ios::ate);
	if (!ifs) throw std::runtime_error("raw: cannot open " + path + " for reading");

	auto file_size = ifs.tellg();
	ifs.seekg(0, std::ios::beg);

	std::size_t count = static_cast<std::size_t>(file_size) / sizeof(T);
	std::vector<T> result(count);
	ifs.read(reinterpret_cast<char*>(result.data()),
	         static_cast<std::streamsize>(count * sizeof(T)));
	return result;
}

} // namespace sw::dsp::io
