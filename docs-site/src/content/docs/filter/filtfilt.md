---
title: Zero-Phase Filtering (filtfilt)
description: Forward-backward filtering for zero phase distortion using biquad cascades
---

A causal IIR filter introduces frequency-dependent phase shift. For
applications where waveform shape must be preserved -- offline analysis,
feature extraction, biomedical signal processing -- the phase distortion
is unacceptable. **Zero-phase filtering** eliminates it by applying the
filter twice: once forward and once backward.

## How it works

The `filtfilt` algorithm processes a signal in four steps:

1. **Reflect** the signal at both ends to reduce startup transients.
2. **Forward pass** through the biquad cascade.
3. **Backward pass** through the same cascade (filter the reversed output).
4. **Extract** the central $N$ samples, discarding the reflected edges.

Because the filter is applied in both directions, the phase contributions
cancel exactly. The magnitude response becomes the square of the
single-pass response:

$$
|H_{\text{filtfilt}}(e^{j\omega})| = |H(e^{j\omega})|^2
$$

This means a Butterworth lowpass with $-3\,\text{dB}$ cutoff at $f_c$
becomes $-6\,\text{dB}$ at $f_c$ after `filtfilt`. To compensate, design
the filter with a slightly higher cutoff.

## Edge transient reduction

Naively filtering a finite signal produces large transients at the start
and end because the filter state starts at zero. `filtfilt` mitigates
this by reflecting the signal beyond both edges:

$$
\text{front: } 2x[0] - x[n_r], \;\ldots,\; 2x[0] - x[1]
$$

$$
\text{back: } 2x[N\!-\!1] - x[N\!-\!2], \;\ldots,\; 2x[N\!-\!1] - x[N\!-\!1\!-\!n_r]
$$

where $n_r = 3(2S + 1) - 1$ and $S$ is the number of biquad stages. The
reflection preserves signal continuity and first-derivative continuity at
the boundaries, giving the filter time to reach steady state before the
actual signal begins.

## API

```cpp
#include <sw/dsp/filter/filtfilt.hpp>

// With explicit state form
auto y = sw::dsp::filtfilt<DirectFormII>(cascade, input);

// Convenience overload (defaults to DirectFormII)
auto y = sw::dsp::filtfilt(cascade, input);
```

### Template parameters

| Parameter | Role |
|-----------|------|
| `StateForm` | Biquad realization (e.g., `DirectFormII`) |
| `CoeffScalar` | Coefficient precision (from the cascade) |
| `MaxStages` | Maximum biquad stages in the cascade |
| `SampleScalar` | Input/output sample precision |

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `cascade` | `Cascade<CoeffScalar, MaxStages>` | The biquad cascade to apply |
| `input` | `std::vector<SampleScalar>` | Input signal |
| **Returns** | `std::vector<SampleScalar>` | Zero-phase filtered output |

## Example: offline ECG filtering

```cpp
#include <sw/dsp/filter/iir/butterworth.hpp>
#include <sw/dsp/filter/filtfilt.hpp>

// Design a 4th-order Butterworth bandpass (0.5--40 Hz at 500 Hz sample rate)
auto cascade = sw::dsp::butterworth_bandpass<double>(4, 0.5 / 250.0, 40.0 / 250.0);

// Zero-phase filter preserves QRS complex morphology
auto ecg_clean = sw::dsp::filtfilt(cascade, ecg_raw);
```

## Comparison with single-pass filtering

| Property | Single pass | `filtfilt` |
|----------|------------|-----------|
| Phase distortion | Yes (frequency-dependent) | None |
| Magnitude | $\|H(e^{j\omega})\|$ | $\|H(e^{j\omega})\|^2$ |
| Effective order | $N$ | $2N$ |
| Causality | Causal (real-time capable) | Non-causal (requires full signal) |
| Latency | Group delay dependent | Zero |

## Precision considerations

The forward and backward passes each accumulate state in `StateForm`. For
narrow-band filters with poles near the unit circle, the state variables
can span many orders of magnitude. Using a high-precision `StateScalar`
(e.g., `posit<32,2>`) for the biquad realization helps maintain accuracy
through both passes, while the input and output can remain in a compact
`SampleScalar` like `posit<16,1>`.
