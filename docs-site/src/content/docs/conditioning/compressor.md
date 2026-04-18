---
title: Dynamic Range Compression
description: Compressor gain curve with threshold, ratio, knee, and attack/release dynamics
---

## Overview

A dynamic range compressor reduces the level difference between the loudest
and quietest parts of a signal. Signals above a **threshold** are attenuated
by a fixed **ratio**, while signals below the threshold pass unchanged. This
is fundamental to broadcast audio, music production, hearing aids, and
telecommunications systems where wide dynamic range must be squeezed into
a narrower output range.

## Static Transfer Curve

### Hard Knee

The simplest compressor applies a piecewise-linear gain curve.
For an input level $X$ in dB and threshold $T$ in dB with compression
ratio $R$:

$$
Y =
\begin{cases}
X, & X \leq T \\
T + \dfrac{X - T}{R}, & X > T
\end{cases}
$$

A ratio of $R = 1$ means no compression; $R = \infty$ produces a hard
limiter where the output never exceeds the threshold.

### Soft Knee

A soft knee smooths the transition around the threshold over a width
$W$ (in dB), preventing audible artefacts:

$$
Y =
\begin{cases}
X, & X \leq T - \dfrac{W}{2} \\[6pt]
X + \dfrac{(1/R - 1)(X - T + W/2)^2}{2W}, & T - \dfrac{W}{2} < X \leq T + \dfrac{W}{2} \\[6pt]
T + \dfrac{X - T}{R}, & X > T + \dfrac{W}{2}
\end{cases}
$$

### Gain Computation

The compressor gain in dB applied to the signal is:

$$
G_{\text{dB}} = Y - X
$$

which is converted to linear gain: $g = 10^{G_{\text{dB}}/20}$.

## Dynamics: Attack and Release

The static curve defines steady-state behaviour, but instantaneous gain
changes cause distortion. The gain is therefore smoothed with attack and
release filters identical to the envelope follower:

$$
g_s[n] =
\begin{cases}
\alpha_a \cdot g_s[n-1] + (1 - \alpha_a) \cdot g[n], & g[n] < g_s[n-1] \\
\alpha_r \cdot g_s[n-1] + (1 - \alpha_r) \cdot g[n], & g[n] \geq g_s[n-1]
\end{cases}
$$

where $\alpha_a = e^{-1/(f_s \cdot \tau_a)}$ and
$\alpha_r = e^{-1/(f_s \cdot \tau_r)}$.

Short attack times ($< 1\,\text{ms}$) catch transients but can introduce
pumping. Longer release times ($50$--$200\,\text{ms}$) yield transparent
compression but may allow overshoot on repeated peaks.

## Library API

```cpp
#include <sw/dsp/conditioning/compressor.hpp>

using Scalar = double;

sw::dsp::Compressor<Scalar> comp;
comp.set_sample_rate(48000.0);
comp.set_threshold(-20.0);   // dB
comp.set_ratio(4.0);         // 4:1
comp.set_knee(6.0);          // dB soft knee width
comp.set_attack(0.005);      // 5 ms
comp.set_release(0.100);     // 100 ms
comp.set_makeup_gain(6.0);   // dB post-compression boost

mtl::vec::dense_vector<Scalar> signal = /* ... */;
for (std::size_t i = 0; i < signal.size(); ++i) {
    signal[i] = comp(signal[i]);
}
```

### Parameter Reference

| Parameter | Method | Range | Description |
|-----------|--------|-------|-------------|
| Threshold | `set_threshold(dB)` | $-60$ to $0$ dB | Level above which compression begins |
| Ratio | `set_ratio(R)` | $1$ to $\infty$ | Input-to-output slope above threshold |
| Knee width | `set_knee(dB)` | $0$ to $20$ dB | Transition zone; 0 = hard knee |
| Attack | `set_attack(s)` | $0.0001$ to $0.1$ s | Time to reach full compression |
| Release | `set_release(s)` | $0.01$ to $1.0$ s | Time to return to unity gain |
| Makeup gain | `set_makeup_gain(dB)` | $0$ to $40$ dB | Static boost after compression |

## Side-Chain Filtering

The compressor optionally accepts a **side-chain** signal that drives the
gain computation while the main signal is compressed. This enables
frequency-selective compression (de-essing) and ducking:

```cpp
comp.set_sidechain(true);

for (std::size_t i = 0; i < signal.size(); ++i) {
    signal[i] = comp(signal[i], sidechain[i]);
}
```

## Precision Considerations

The dB-domain computations involve $\log_{10}$ and $10^x$ conversions that
are sensitive to low-precision types. Using `double` or a 32-bit posit for
the internal gain path avoids quantization steps in the transfer curve while
keeping the audio sample path in a compact type such as `float` or 16-bit
fixed point.
