---
title: Envelope Detection
description: Peak hold and exponential decay envelope followers for amplitude tracking
---

## Overview

Envelope detection extracts the slowly varying amplitude contour of a signal,
discarding the fine-grain oscillation underneath. Two dominant strategies exist:
**peak hold** (captures instantaneous maxima) and **exponential decay**
(smooths amplitude with separate attack and release time constants).

Both methods are essential building blocks for audio level metering,
AM demodulation, speech activity detection, and automatic gain control.

## Mathematical Foundation

### Rectification

The first step is full-wave rectification to obtain the instantaneous magnitude:

$$
x_{\text{rect}}[n] = |x[n]|
$$

### Exponential Envelope Follower

The envelope $e[n]$ tracks the rectified signal using asymmetric smoothing.
When the input exceeds the current envelope the **attack** coefficient applies;
otherwise the **release** coefficient governs the decay:

$$
e[n] =
\begin{cases}
\alpha_a \cdot e[n-1] + (1 - \alpha_a) \cdot x_{\text{rect}}[n], & x_{\text{rect}}[n] > e[n-1] \\
\alpha_r \cdot e[n-1], & \text{otherwise}
\end{cases}
$$

The smoothing coefficients are derived from user-specified time constants
$\tau_a$ (attack) and $\tau_r$ (release) and the sample rate $f_s$:

$$
\alpha = e^{-1/(f_s \cdot \tau)}
$$

A larger $\tau$ yields a smoother (slower) response. Typical values are
$\tau_a \approx 0.01\,\text{s}$ for fast attack and
$\tau_r \approx 0.1\,\text{s}$ for gradual release.

### Peak Hold

An alternative is the **peak hold** detector, which latches onto the
maximum value and decays only after a configurable hold time $T_h$:

$$
e[n] =
\begin{cases}
x_{\text{rect}}[n], & x_{\text{rect}}[n] \geq e[n-1] \\
\alpha_r \cdot e[n-1], & n - n_{\text{peak}} > T_h \cdot f_s \\
e[n-1], & \text{otherwise}
\end{cases}
$$

## Library API

The `sw::dsp` library provides `EnvelopeFollower<T>`, templated on the
sample type. Construction requires the sample rate and the attack/release
time constants.

```cpp
#include <sw/dsp/conditioning/envelope.hpp>

using Scalar = float;

// Create an envelope follower: sample rate, attack (s), release (s)
sw::dsp::EnvelopeFollower<Scalar> env(48000.0, 0.01, 0.1);

// Process a signal buffer
mtl::vec::dense_vector<Scalar> signal = /* ... */;
mtl::vec::dense_vector<Scalar> envelope(signal.size());

for (std::size_t i = 0; i < signal.size(); ++i) {
    envelope[i] = env(signal[i]);
}
```

### Configuration

| Parameter | Method | Description |
|-----------|--------|-------------|
| Attack time | `set_attack(tau)` | Time constant in seconds for rising edges |
| Release time | `set_release(tau)` | Time constant in seconds for falling edges |
| Hold time | `set_hold(seconds)` | Peak hold duration before decay begins |
| Mode | `set_mode(mode)` | `Peak` or `RMS` rectification |

### RMS Mode

Setting the mode to `RMS` replaces instantaneous rectification with a
running mean-square estimate, producing a smoother envelope at the cost
of slightly delayed tracking:

$$
e_{\text{rms}}[n] = \sqrt{\alpha \cdot e_{\text{rms}}^2[n-1] + (1 - \alpha) \cdot x^2[n]}
$$

## Use Cases

**Audio level metering.** VU meters and PPM meters use envelope followers
with standardized attack and release times (e.g., 300 ms attack for VU,
5 ms for PPM).

**AM demodulation.** The modulating signal is recovered by envelope-detecting
the AM carrier. The release time constant must be fast enough to track the
highest modulation frequency but slow enough to reject the carrier.

**Trigger and gating.** Comparing the envelope against a threshold produces
an activity gate for noise gates, voice activity detectors, and squelch
circuits.

## Precision Considerations

Because the exponential smoother is a first-order IIR filter, the same
mixed-precision concerns apply. When $\alpha$ is close to 1 (long time
constants), the subtraction $1 - \alpha$ loses significance in low-precision
types. Using a higher-precision `StateScalar` for the accumulator $e[n]$
prevents drift while keeping the sample path in a compact format.
