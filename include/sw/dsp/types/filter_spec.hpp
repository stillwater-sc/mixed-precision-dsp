#pragma once
// filter_spec.hpp: filter specification parameters
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/types/filter_kind.hpp>

namespace sw::dsp {

// Common filter specification parameters.
// Not all fields are used by all filter types; unused fields are zero.
struct FilterSpec {
	double sample_rate{44100.0};
	double cutoff_frequency{0.0};     // for low/high pass
	double center_frequency{0.0};     // for band pass/stop
	double bandwidth{0.0};            // for band pass/stop
	double gain_db{0.0};              // for shelf filters
	double ripple_db{0.0};            // passband ripple (Chebyshev I, Elliptic)
	double stopband_db{0.0};          // stopband attenuation (Chebyshev II, Elliptic)
	double rolloff{0.0};              // Elliptic rolloff parameter
	double q{0.707106781186548};      // quality factor (RBJ filters)
	double shelf_slope{1.0};          // shelf slope (RBJ filters)
	int    order{2};                  // filter order

	// Normalized cutoff frequency = cutoff / sample_rate
	double normalized_cutoff() const {
		return cutoff_frequency / sample_rate;
	}

	// Normalized center frequency = center / sample_rate
	double normalized_center() const {
		return center_frequency / sample_rate;
	}

	// Normalized bandwidth = bandwidth / sample_rate
	double normalized_bandwidth() const {
		return bandwidth / sample_rate;
	}
};

} // namespace sw::dsp
