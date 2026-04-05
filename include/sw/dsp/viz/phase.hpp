#pragma once
// phase.hpp: plot phase response of a filter cascade
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <iostream>
#include <span>
#include <vector>
#include <sw/dsp/viz/plot.hpp>

namespace sw::dsp::viz {

// Compute phase response in degrees at num_points frequencies.
template <typename CascadeType>
std::vector<double> phase_response_degrees(const CascadeType& cascade,
                                            int num_points = 512) {
	std::vector<double> phase(num_points);
	for (int i = 0; i < num_points; ++i) {
		double f = static_cast<double>(i) / (num_points - 1) * 0.5;
		auto r = cascade.response(f);
		phase[i] = std::arg(r) * 180.0 / 3.14159265358979323846;
	}
	return phase;
}

// Plot phase response of a cascade.
template <typename CascadeType>
void plot_phase_response(std::ostream& os,
                          const CascadeType& cascade,
                          double sample_rate = 1.0,
                          const PlotConfig& base_cfg = {}) {
	auto phase = phase_response_degrees(cascade, base_cfg.width > 0 ? base_cfg.width : 72);

	PlotConfig cfg = base_cfg;
	if (cfg.title.empty()) cfg.title = "Phase Response";
	if (cfg.y_label.empty()) cfg.y_label = "degrees";
	if (cfg.x_label.empty()) {
		if (sample_rate > 1.0) {
			cfg.x_label = "0 Hz -> " + std::to_string(static_cast<int>(sample_rate / 2)) + " Hz";
		} else {
			cfg.x_label = "0 -> 0.5 (normalized frequency)";
		}
	}

	plot_line(os, std::span<const double>(phase), cfg);
}

} // namespace sw::dsp::viz
