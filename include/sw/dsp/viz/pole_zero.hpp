#pragma once
// pole_zero.hpp: text-based pole-zero plot with unit circle
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

namespace sw::dsp::viz {

// Plot poles and zeros on a character grid with the unit circle.
// Poles are marked 'x', zeros are marked 'o'.
template <typename CascadeType>
void plot_pole_zero(std::ostream& os, const CascadeType& cascade, int size = 21) {
	// Collect poles and zeros from biquad coefficients
	std::vector<std::complex<double>> poles, zeros;

	for (int s = 0; s < cascade.num_stages(); ++s) {
		const auto& c = cascade.stage(s);
		double a1 = static_cast<double>(c.a1);
		double a2 = static_cast<double>(c.a2);
		double b0 = static_cast<double>(c.b0);
		double b1 = static_cast<double>(c.b1);
		double b2 = static_cast<double>(c.b2);

		// Poles: roots of z^2 + a1*z + a2
		double disc_p = a1 * a1 - 4.0 * a2;
		if (disc_p >= 0) {
			poles.push_back({(-a1 + std::sqrt(disc_p)) / 2.0, 0.0});
			poles.push_back({(-a1 - std::sqrt(disc_p)) / 2.0, 0.0});
		} else {
			poles.push_back({-a1 / 2.0, std::sqrt(-disc_p) / 2.0});
			poles.push_back({-a1 / 2.0, -std::sqrt(-disc_p) / 2.0});
		}

		// Zeros: roots of b0*z^2 + b1*z + b2
		if (b0 == 0.0) {
			if (b1 != 0.0) zeros.push_back({-b2 / b1, 0.0});
		} else {
			double disc_z = b1 * b1 - 4.0 * b0 * b2;
			if (disc_z >= 0) {
				zeros.push_back({(-b1 + std::sqrt(disc_z)) / (2.0 * b0), 0.0});
				zeros.push_back({(-b1 - std::sqrt(disc_z)) / (2.0 * b0), 0.0});
			} else {
				zeros.push_back({-b1 / (2.0 * b0), std::sqrt(-disc_z) / (2.0 * b0)});
				zeros.push_back({-b1 / (2.0 * b0), -std::sqrt(-disc_z) / (2.0 * b0)});
			}
		}
	}

	// Grid: map [-1.5, 1.5] x [-1.5, 1.5] to character grid
	const double range = 1.5;
	std::vector<std::string> grid(size, std::string(size * 2 - 1, ' '));
	int cx = size - 1;  // center column (in the wider grid)
	int cy = size / 2;  // center row

	// Draw unit circle
	for (int i = 0; i < 360; ++i) {
		double angle = i * 3.14159265358979323846 / 180.0;
		double re = std::cos(angle);
		double im = std::sin(angle);
		int col = static_cast<int>((re / range + 1.0) * 0.5 * (size * 2 - 2));
		int row = static_cast<int>((1.0 - im / range) * 0.5 * (size - 1));
		if (row >= 0 && row < size && col >= 0 && col < size * 2 - 1) {
			if (grid[row][col] == ' ') grid[row][col] = '.';
		}
	}

	// Draw axes
	for (int c = 0; c < size * 2 - 1; ++c) {
		if (grid[cy][c] == ' ') grid[cy][c] = '-';
	}
	for (int r = 0; r < size; ++r) {
		if (grid[r][cx] == ' ') grid[r][cx] = '|';
	}
	grid[cy][cx] = '+';

	// Place zeros
	for (const auto& z : zeros) {
		int col = static_cast<int>((z.real() / range + 1.0) * 0.5 * (size * 2 - 2));
		int row = static_cast<int>((1.0 - z.imag() / range) * 0.5 * (size - 1));
		if (row >= 0 && row < size && col >= 0 && col < size * 2 - 1) {
			grid[row][col] = 'o';
		}
	}

	// Place poles (overwrite zeros if coincident)
	for (const auto& p : poles) {
		int col = static_cast<int>((p.real() / range + 1.0) * 0.5 * (size * 2 - 2));
		int row = static_cast<int>((1.0 - p.imag() / range) * 0.5 * (size - 1));
		if (row >= 0 && row < size && col >= 0 && col < size * 2 - 1) {
			grid[row][col] = 'x';
		}
	}

	// Title
	os << "  Pole-Zero Plot (x=pole, o=zero)\n";

	// Render
	for (int r = 0; r < size; ++r) {
		os << "  " << grid[r] << '\n';
	}
	os << '\n';
}

} // namespace sw::dsp::viz
