---
title: Automatic Gain Control
description: Feedback-loop AGC with adaptive gain for communications and audio normalization
---

## Overview

Automatic Gain Control (AGC) adjusts the amplitude of a signal so that
the output maintains a consistent target level regardless of input
variations. Unlike a compressor, which applies a static transfer curve,
AGC uses a **feedback loop** that continuously adapts a multiplicative
gain factor. AGC is essential in communications receivers where incoming
signal power can vary by many orders of magnitude, and in audio
normalization pipelines where consistent loudness is required.

## Feedback Loop Architecture

The AGC operates in a closed loop:

1. Multiply the input by the current gain: $y[n] = g[n] \cdot x[n]$
2. Estimate the output power: $P[n]$
3. Update the gain for the next sample based on the error between the
   target power and the measured power.

### Power Estimation

The instantaneous power estimate uses exponential smoothing:

$$
P[n] = \alpha \cdot P[n-1] + (1 - \alpha) \cdot y^2[n]
$$

where $\alpha = e^{-1/(f_s \cdot \tau_p)}$ and $\tau_p$ is the power
averaging time constant.

### Gain Update Rule

The gain is adapted to drive the output power toward $P_{\text{target}}$:

$$
g[n+1] = g[n] \cdot \left(1 + \mu \left(P_{\text{target}} - P[n]\right)\right)
$$

Here $\mu$ controls the **adaptation rate**. A larger $\mu$ converges
faster but risks oscillation; a smaller $\mu$ is more stable but responds
slowly to power changes.

### Stability Constraint

For the feedback loop to remain stable the adaptation rate must satisfy:

$$
0 < \mu < \frac{2}{P_{\max}}
$$

where $P_{\max}$ is the maximum expected signal power. The library
enforces this bound by default.

### Gain Limits

To prevent unbounded amplification during silence or clipping on transients,
the gain is clamped:

$$
g_{\min} \leq g[n] \leq g_{\max}
$$

Typical values are $g_{\min} = 0.01$ ($-40\,\text{dB}$) and
$g_{\max} = 100$ ($+40\,\text{dB}$).

## Library API

```cpp
#include <sw/dsp/conditioning/agc.hpp>

using Scalar = float;

sw::dsp::AGC<Scalar> agc;
agc.set_sample_rate(48000.0);
agc.set_target_level(0.5);     // target RMS amplitude
agc.set_adaptation_rate(0.01); // mu
agc.set_gain_limits(0.01, 100.0);
agc.set_averaging_time(0.05);  // 50 ms power estimator

mtl::vec::dense_vector<Scalar> signal = /* ... */;
for (std::size_t i = 0; i < signal.size(); ++i) {
    signal[i] = agc(signal[i]);
}

// Query current state
Scalar current_gain = agc.gain();
Scalar current_power = agc.power();
```

### Parameter Reference

| Parameter | Method | Description |
|-----------|--------|-------------|
| Target level | `set_target_level(v)` | Desired output RMS amplitude |
| Adaptation rate | `set_adaptation_rate(mu)` | Loop gain; larger = faster convergence |
| Gain limits | `set_gain_limits(min, max)` | Prevent runaway gain or clipping |
| Averaging time | `set_averaging_time(s)` | Power estimator smoothing constant |
| Freeze | `set_freeze(bool)` | Hold gain constant (useful during calibration) |

## Applications

### Communications Receivers

Radio receivers encounter signal-strength variations of 80 dB or more due
to fading, distance, and interference. The AGC normalizes the baseband
signal before demodulation, keeping the signal within the ADC's dynamic
range and the demodulator's operating point.

### Audio Normalization

Podcast and broadcast chains use AGC to level-match speakers with
different microphone distances. A slow adaptation rate
($\mu \approx 0.001$) prevents audible gain riding.

### Sensor Front-Ends

Ultrasonic and radar receivers use time-varying gain (TVG) that is
functionally an AGC with a distance-dependent target level, compensating
for the $1/r^2$ propagation loss.

## Precision Considerations

The gain update involves a multiply-accumulate that is sensitive to the
resolution of $\mu$ and $P_{\text{target}}$. When $\mu$ is very small,
the product $\mu \cdot (P_{\text{target}} - P[n])$ can fall below the
representable range of narrow types, causing the gain to stall. A
higher-precision state scalar for the gain accumulator prevents this
while keeping the signal multiplication in a compact sample type.

The power estimate $P[n]$ is always non-negative, so unsigned or
posit types that concentrate precision near zero are a natural fit.
