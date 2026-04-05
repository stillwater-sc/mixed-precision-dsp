#pragma once
// plot.hpp: text-based plotting for console output
//
// Renders 1D data as ASCII art to std::ostream. Supports line plots,
// bar charts, and scatter modes. Auto-scales or accepts fixed axis ranges.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <span>
#include <string>
#include <vector>

namespace sw::dsp::viz {

struct PlotConfig {
	int width  = 72;    // plot area width in characters (excluding labels)
	int height = 20;    // plot area height in characters
	std::string title;
	std::string x_label;
	std::string y_label;
	double y_min = std::numeric_limits<double>::quiet_NaN();  // NaN = auto
	double y_max = std::numeric_limits<double>::quiet_NaN();
	bool show_axis = true;
	char fill_char = '*';
	char bar_char  = '#';
};

// Render a line plot of y-values to an ostream.
// X-axis is sample index (or mapped to x_values if provided).
template <typename T>
void plot_line(std::ostream& os, std::span<const T> data, const PlotConfig& cfg = {}) {
	if (data.empty()) return;

	// Convert to double
	std::vector<double> y(data.size());
	for (std::size_t i = 0; i < data.size(); ++i) {
		y[i] = static_cast<double>(data[i]);
	}

	// Determine Y range
	double ymin = cfg.y_min, ymax = cfg.y_max;
	if (std::isnan(ymin) || std::isnan(ymax)) {
		auto [it_min, it_max] = std::minmax_element(y.begin(), y.end());
		if (std::isnan(ymin)) ymin = *it_min;
		if (std::isnan(ymax)) ymax = *it_max;
	}
	if (ymax == ymin) { ymax += 1.0; ymin -= 1.0; }

	const int w = cfg.width;
	const int h = cfg.height;
	const int label_w = 10;  // width reserved for Y-axis labels

	// Title
	if (!cfg.title.empty()) {
		int pad = (label_w + w - static_cast<int>(cfg.title.size())) / 2;
		if (pad < 0) pad = 0;
		os << std::string(pad, ' ') << cfg.title << '\n';
	}

	// For each column, compute the min/max envelope of all samples
	// that map to it. This gives a clear waveform display even when
	// many samples share a column (dense signals).
	std::vector<double> col_min(w, ymax);
	std::vector<double> col_max(w, ymin);
	std::vector<bool>   col_hit(w, false);

	for (std::size_t i = 0; i < y.size(); ++i) {
		int col = static_cast<int>(static_cast<double>(i) / (y.size() - 1) * (w - 1));
		col = std::clamp(col, 0, w - 1);
		col_min[col] = std::min(col_min[col], y[i]);
		col_max[col] = std::max(col_max[col], y[i]);
		col_hit[col] = true;
	}

	// Build character grid from the envelope
	std::vector<std::string> grid(h, std::string(w, ' '));

	for (int c = 0; c < w; ++c) {
		if (!col_hit[c]) continue;

		int row_top = std::clamp(
			static_cast<int>((ymax - col_max[c]) / (ymax - ymin) * (h - 1)), 0, h - 1);
		int row_bot = std::clamp(
			static_cast<int>((ymax - col_min[c]) / (ymax - ymin) * (h - 1)), 0, h - 1);

		// Fill the envelope range for this column
		for (int r = row_top; r <= row_bot; ++r) {
			grid[r][c] = cfg.fill_char;
		}
	}

	// Render with Y-axis labels
	for (int r = 0; r < h; ++r) {
		if (cfg.show_axis) {
			double yval = ymax - static_cast<double>(r) / (h - 1) * (ymax - ymin);
			std::ostringstream lbl;
			lbl << std::setw(label_w - 2) << std::setprecision(3) << std::fixed << yval;
			os << lbl.str() << " |";
		}
		os << grid[r] << '\n';
	}

	// X-axis line
	if (cfg.show_axis) {
		os << std::string(label_w - 1, ' ') << '+' << std::string(w, '-') << '\n';
	}

	// X-axis label
	if (!cfg.x_label.empty()) {
		int pad = label_w + (w - static_cast<int>(cfg.x_label.size())) / 2;
		if (pad < 0) pad = 0;
		os << std::string(pad, ' ') << cfg.x_label << '\n';
	}
}

// Render a bar chart of y-values.
template <typename T>
void plot_bars(std::ostream& os, std::span<const T> data, const PlotConfig& cfg = {}) {
	if (data.empty()) return;

	std::vector<double> y(data.size());
	for (std::size_t i = 0; i < data.size(); ++i) {
		y[i] = static_cast<double>(data[i]);
	}

	double ymin = cfg.y_min, ymax = cfg.y_max;
	if (std::isnan(ymin)) ymin = 0.0;
	if (std::isnan(ymax)) ymax = *std::max_element(y.begin(), y.end());
	if (ymax <= ymin) ymax = ymin + 1.0;

	const int w = cfg.width;
	const int h = cfg.height;
	const int label_w = 10;

	if (!cfg.title.empty()) {
		int pad = (label_w + w - static_cast<int>(cfg.title.size())) / 2;
		if (pad < 0) pad = 0;
		os << std::string(pad, ' ') << cfg.title << '\n';
	}

	// Each bar gets equal width
	int bar_w = std::max(1, w / static_cast<int>(y.size()));

	for (int r = 0; r < h; ++r) {
		double threshold = ymax - static_cast<double>(r) / (h - 1) * (ymax - ymin);
		if (cfg.show_axis) {
			std::ostringstream lbl;
			lbl << std::setw(label_w - 2) << std::setprecision(3) << std::fixed << threshold;
			os << lbl.str() << " |";
		}
		for (std::size_t i = 0; i < y.size(); ++i) {
			char c = (y[i] >= threshold) ? cfg.bar_char : ' ';
			for (int k = 0; k < bar_w && static_cast<int>(i) * bar_w + k < w; ++k) {
				os << c;
			}
		}
		os << '\n';
	}

	if (cfg.show_axis) {
		os << std::string(label_w - 1, ' ') << '+' << std::string(w, '-') << '\n';
	}
}

} // namespace sw::dsp::viz
