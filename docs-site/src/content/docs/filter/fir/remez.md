---
title: Equiripple FIR Design (Remez)
description: Parks-McClellan optimal equiripple FIR filter design via the Remez exchange algorithm
---

The Parks-McClellan algorithm (1972) designs FIR filters that are
**optimal in the Chebyshev (minimax) sense**: for a given filter order,
no other FIR filter achieves a smaller maximum deviation from the desired
response. The result is an **equiripple** filter -- the approximation
error oscillates between equal-magnitude extremes across each band.

## Why equiripple?

Windowed FIR design (e.g., Kaiser window) is simple but wasteful: the
error is concentrated near the band edges while the rest of the band
is over-specified. The Remez exchange algorithm redistributes the error
uniformly, achieving tighter specifications for the same filter order --
or equivalently, a shorter filter for the same specifications.

| Design method | Error distribution | Order efficiency |
|---------------|-------------------|-----------------|
| Windowed sinc | Concentrated at transitions | Baseline |
| Least-squares | Minimizes total energy | Good for smooth specs |
| **Equiripple** | Uniform maximum across bands | **Optimal (minimax)** |

## The Remez exchange algorithm

The algorithm iterates four phases:

1. **Grid construction.** Build a dense frequency grid across all
   specified bands. Grid density controls accuracy vs. computation time.

2. **Extremal set initialization.** Select $L+2$ candidate frequencies
   uniformly from the grid, where $L$ is half the filter order (rounded
   up). These are the initial guesses for the error extrema.

3. **Remez exchange.** At each iteration:
   - Solve for the polynomial approximation and peak deviation $\delta$
     using barycentric Lagrange interpolation.
   - Evaluate the weighted error across the full grid.
   - Find the new set of $L+2$ frequencies where the error achieves
     alternating-sign extrema.
   - If the extremal set didn't change, convergence is reached.

4. **Coefficient extraction.** Compute the final filter taps from the
   optimal frequency response via inverse DCT (Type I/II for symmetric
   filters) or inverse DST (Type III/IV for antisymmetric filters).

## Filter types

The library supports three filter types via `RemezBandType`:

| Type | Symmetry | Use case |
|------|----------|----------|
| `bandpass` | Symmetric coefficients | Lowpass, highpass, bandpass, bandstop |
| `differentiator` | Antisymmetric coefficients | Discrete differentiators |
| `hilbert` | Antisymmetric coefficients | Hilbert transformers (90-degree phase shift) |

## API

### General interface

```cpp
#include <sw/dsp/filter/fir/remez.hpp>

auto taps = sw::dsp::remez<double>(
    num_taps,       // filter length (must be >= 3)
    bands,          // band edges: {f1_start, f1_stop, f2_start, f2_stop, ...}
    desired,        // desired gain at each band edge
    weights,        // relative weight per band
    type,           // RemezBandType::bandpass (default)
    max_iterations, // default: 40
    grid_density    // default: 16
);
```

All frequencies are normalized to $[0, 0.5]$ where $0.5$ is the Nyquist
frequency ($f_s / 2$).

### Convenience functions

```cpp
// Equiripple lowpass: passband 0 to f_pass, stopband f_stop to 0.5
auto lp = sw::dsp::design_fir_equiripple_lowpass<double>(
    num_taps, f_pass, f_stop, passband_weight, stopband_weight);

// Equiripple bandpass: pass from f_low to f_high
auto bp = sw::dsp::design_fir_equiripple_bandpass<double>(
    num_taps, f_low, f_high, transition_width, passband_weight, stopband_weight);
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_taps` | `std::size_t` | Filter length ($\geq 3$) |
| `bands` | `std::vector<T>` | Band edge pairs, normalized to $[0, 0.5]$ |
| `desired` | `std::vector<T>` | Desired response at each band edge |
| `weights` | `std::vector<T>` | Relative importance of each band |
| `type` | `RemezBandType` | Filter symmetry type |
| **Returns** | `dense_vector<T>` | FIR filter coefficients |

## Example: sharp lowpass

```cpp
// 101-tap equiripple lowpass at 0.2 * Nyquist
// Passband: [0, 0.2], Stopband: [0.25, 0.5]
// Stopband weighted 10x heavier than passband
auto taps = sw::dsp::remez<double>(
    101,
    {0.0, 0.2, 0.25, 0.5},  // bands
    {1.0, 1.0, 0.0, 0.0},   // desired
    {1.0, 10.0}              // weights
);
```

## Precision considerations

The Remez exchange algorithm involves solving a system of equations at
each iteration. The library performs all internal computation in `double`
precision regardless of the output type `T`. This ensures numerical
stability of the design process. The final tap values are then projected
to the target type.

For the three-scalar model, equiripple FIR taps are natural candidates
for `CoeffScalar`. Because the taps are computed once at design time,
using a high-precision coefficient type preserves the carefully optimized
equiripple structure. The convolution state (`StateScalar`) benefits
from extra precision in the accumulator to maintain the minimax
guarantee during processing.
