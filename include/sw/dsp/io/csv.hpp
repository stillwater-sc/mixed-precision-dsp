#pragma once
// csv.hpp: CSV signal reader/writer
//
// Writes signals as comma-separated values, one row per sample.
// Compatible with Python numpy.loadtxt() and MATLAB csvread().
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <span>
#include <vector>

namespace sw::dsp::io {

// Write a single column of samples.
template <typename T>
void write_csv(const std::string& path, std::span<const T> samples,
               const std::string& header = "sample") {
	std::ofstream ofs(path);
	if (!ofs) throw std::runtime_error("csv: cannot open " + path + " for writing");

	if (!header.empty()) {
		ofs << header << '\n';
	}

	ofs << std::setprecision(15);
	for (std::size_t i = 0; i < samples.size(); ++i) {
		ofs << static_cast<double>(samples[i]) << '\n';
	}
}

// Write two columns (e.g., time + signal, or input + output).
template <typename T1, typename T2>
void write_csv(const std::string& path,
               std::span<const T1> col1, std::span<const T2> col2,
               const std::string& header1 = "col1",
               const std::string& header2 = "col2") {
	std::ofstream ofs(path);
	if (!ofs) throw std::runtime_error("csv: cannot open " + path + " for writing");

	ofs << header1 << ',' << header2 << '\n';
	ofs << std::setprecision(15);

	std::size_t n = std::max(col1.size(), col2.size());
	for (std::size_t i = 0; i < n; ++i) {
		if (i < col1.size()) ofs << static_cast<double>(col1[i]);
		ofs << ',';
		if (i < col2.size()) ofs << static_cast<double>(col2[i]);
		ofs << '\n';
	}
}

// Write multiple columns from a vector of spans.
template <typename T>
void write_csv(const std::string& path,
               const std::vector<std::span<const T>>& columns,
               const std::vector<std::string>& headers = {}) {
	std::ofstream ofs(path);
	if (!ofs) throw std::runtime_error("csv: cannot open " + path + " for writing");

	// Write header row
	if (!headers.empty()) {
		for (std::size_t c = 0; c < headers.size(); ++c) {
			if (c > 0) ofs << ',';
			ofs << headers[c];
		}
		ofs << '\n';
	}

	ofs << std::setprecision(15);

	// Find max length
	std::size_t max_len = 0;
	for (const auto& col : columns) {
		max_len = std::max(max_len, col.size());
	}

	for (std::size_t i = 0; i < max_len; ++i) {
		for (std::size_t c = 0; c < columns.size(); ++c) {
			if (c > 0) ofs << ',';
			if (i < columns[c].size()) {
				ofs << static_cast<double>(columns[c][i]);
			}
		}
		ofs << '\n';
	}
}

// Read a single column from a CSV file.
// Skips the first row if it doesn't parse as a number (header detection).
template <typename T>
std::vector<T> read_csv(const std::string& path, int column = 0) {
	std::ifstream ifs(path);
	if (!ifs) throw std::runtime_error("csv: cannot open " + path + " for reading");

	std::vector<T> result;
	std::string line;
	bool first_line = true;

	while (std::getline(ifs, line)) {
		if (line.empty()) continue;

		// Split by comma
		std::vector<std::string> fields;
		std::stringstream ss(line);
		std::string field;
		while (std::getline(ss, field, ',')) {
			fields.push_back(field);
		}

		if (column >= static_cast<int>(fields.size())) continue;

		// Try to parse as number; skip header row
		try {
			double val = std::stod(fields[column]);
			result.push_back(static_cast<T>(val));
		} catch (const std::invalid_argument&) {
			if (first_line) {
				first_line = false;
				continue;  // skip header
			}
			// Non-header non-numeric: skip
		}
		first_line = false;
	}

	return result;
}

} // namespace sw::dsp::io
