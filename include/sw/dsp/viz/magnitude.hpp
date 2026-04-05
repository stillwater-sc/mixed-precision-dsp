#pragma once
// magnitude.hpp: plot magnitude response of a filter cascade
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <iostream>
#include <span>
#include <vector>
#include <sw/dsp/viz/plot.hpp>

namespace sw::dsp::viz {

// Compute magnitude response in dB at num_points frequencies from 0 to Nyquist.
// Returns a vector of dB values.
template <typename CascadeType>
std::vector<double> magnitude_response_db(const CascadeType& cascade,
                                           int num_points = 512,
                                           double min_db = -80.0) {
	std::vector<double> mag_db(num_points);
	for (int i = 0; i < num_points; ++i) {
		double f = static_cast<double>(i) / (num_points - 1) * 0.5;  // 0 to 0.5
		auto r = cascade.response(f);
		double mag = std::abs(r);
		double db = (mag > 0.0) ? 20.0 * std::log10(mag) : min_db;
		mag_db[i] = std::max(db, min_db);
	}
	return mag_db;
}

// Plot magnitude response of a cascade to an ostream.
// sample_rate is used only for X-axis labeling.
template <typename CascadeType>
void plot_magnitude_response(std::ostream& os,
                              const CascadeType& cascade,
                              double sample_rate = 1.0,
                              const PlotConfig& base_cfg = {}) {
	auto mag_db = magnitude_response_db(cascade, base_cfg.width > 0 ? base_cfg.width : 72);

	PlotConfig cfg = base_cfg;
	if (cfg.title.empty()) cfg.title = "Magnitude Response";
	if (cfg.y_label.empty()) cfg.y_label = "dB";
	if (cfg.x_label.empty()) {
		if (sample_rate > 1.0) {
			cfg.x_label = "0 Hz -> " + std::to_string(static_cast<int>(sample_rate / 2)) + " Hz";
		} else {
			cfg.x_label = "0 -> 0.5 (normalized frequency)";
		}
	}
	if (std::isnan(cfg.y_min)) cfg.y_min = -80.0;
	if (std::isnan(cfg.y_max)) cfg.y_max = 6.0;

	plot_line(os, std::span<const double>(mag_db), cfg);
}

} // namespace sw::dsp::viz
