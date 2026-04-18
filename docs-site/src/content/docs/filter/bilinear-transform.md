---
title: Bilinear Transform
description: The s-to-z mapping, frequency warping, prewarping, and Constantinides frequency transformations in sw::dsp
---

The **bilinear transform** maps a continuous-time (analog) transfer function
$H_a(s)$ to a discrete-time transfer function $H(z)$. It is the bridge
between classical analog prototype design and the digital biquad cascade.

## The s-to-z mapping

The substitution is:

$$
s = 2 f_s \frac{z - 1}{z + 1}
$$

where $f_s$ is the sample rate. This maps the entire left half of the
$s$-plane to the interior of the unit circle in the $z$-plane, so a stable
analog filter always produces a stable digital filter.

Evaluating on the unit circle $z = e^{j\omega}$ gives the frequency
relationship between the analog frequency $\Omega$ and the digital frequency
$\omega$:

$$
\Omega = 2 f_s \tan\!\left(\frac{\omega}{2}\right)
$$

## Frequency warping

The tangent mapping compresses the infinite analog frequency axis onto the
finite interval $[0, \pi]$. Near DC the mapping is approximately linear
($\Omega \approx \omega f_s$), but it becomes increasingly nonlinear toward
Nyquist. This means a filter designed for analog cutoff $\Omega_c$ will have
its digital cutoff shifted slightly.

## Prewarping

To ensure the digital filter's cutoff lands exactly at the desired frequency
$\omega_c$, we **prewarp** the analog prototype frequency:

$$
\Omega_c = 2 f_s \tan\!\left(\frac{\omega_c}{2}\right)
\quad\text{where}\quad
\omega_c = 2\pi \frac{f_c}{f_s}
$$

The analog prototype is then designed for $\Omega_c$ instead of $2\pi f_c$.
After bilinear transformation, the resulting digital filter matches $f_c$
exactly.

## Constantinides frequency transformations

A normalized analog lowpass prototype with cutoff $\Omega = 1$ can be
transformed to other response types before or during the bilinear step. These
are known as **Constantinides transformations** and operate as $s$-plane
substitutions.

### Lowpass to highpass

$$
s \leftarrow \frac{\Omega_c}{s}
$$

Mirrors the lowpass response about $\Omega_c$. Passband and stopband swap.

### Lowpass to bandpass

$$
s \leftarrow \frac{s^2 + \Omega_l \Omega_h}{s(\Omega_h - \Omega_l)}
$$

where $\Omega_l$ and $\Omega_h$ are the lower and upper band edges. An
$N$th-order prototype produces a $2N$th-order bandpass filter (each prototype
pole generates a conjugate pair).

### Lowpass to bandstop

$$
s \leftarrow \frac{s(\Omega_h - \Omega_l)}{s^2 + \Omega_l \Omega_h}
$$

The dual of the bandpass transformation. The rejection band sits between
$\Omega_l$ and $\Omega_h$.

## Library internals

The library implements these as composable function objects in
`<sw/dsp/filter/transform/>`:

| Class | Prototype mapping |
|---|---|
| `LowPassTransform` | Lowpass to lowpass (identity with prewarping) |
| `HighPassTransform` | Lowpass to highpass |
| `BandPassTransform` | Lowpass to bandpass |
| `BandStopTransform` | Lowpass to bandstop |

Each transform takes the analog prototype poles and zeros plus the target
edge frequencies and produces the digital biquad coefficients in a single
pass, fusing the Constantinides substitution with the bilinear map.

### Example: manual bilinear step

```cpp
#include <sw/dsp/filter/transform/bilinear.hpp>
using namespace sw::dsp;

double fs = 48000.0;
double fc = 1000.0;

// Prewarp
double wc = 2.0 * M_PI * fc / fs;
double Omega_c = 2.0 * fs * std::tan(wc / 2.0);

// LowPassTransform maps analog poles/zeros at Omega_c
// into z-plane biquad coefficients
LowPassTransform transform;
auto cascade = transform(prototype, fs, fc);
```

### Example: bandpass via Constantinides

```cpp
#include <sw/dsp/filter/iir/butterworth.hpp>
using namespace sw::dsp;

// 4th-order Butterworth bandpass (300 Hz -- 3400 Hz)
SimpleFilter<ButterworthBandPass<4>, double, double, float> bp;
bp.setup(4, 48000.0, 300.0, 3400.0);
// Internally: 4th-order LP prototype
//   -> BandPassTransform -> 8th-order digital bandpass
//   -> 4 biquad sections
```

The transform classes are rarely called directly. The design policies
(Butterworth, Chebyshev, Elliptic, Bessel) invoke the appropriate transform
automatically during `setup()`. Understanding the underlying math is useful
for diagnosing frequency-warping artifacts in narrowband or high-frequency
designs.
