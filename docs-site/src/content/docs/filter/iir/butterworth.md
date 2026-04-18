---
title: Butterworth Filters
description: Maximally flat magnitude response IIR filters -- theory, pole placement, and the sw::dsp Butterworth API
---

The Butterworth filter achieves the **maximally flat magnitude response** in
the passband. Of all polynomial transfer functions of a given order, the
Butterworth has the most monotonic magnitude -- no ripple in either passband or
stopband.

## Magnitude response

The squared magnitude of an $N$th-order analog Butterworth lowpass is:

$$
|H_a(j\Omega)|^2 = \frac{1}{1 + \left(\frac{\Omega}{\Omega_c}\right)^{2N}}
$$

At the cutoff frequency $\Omega = \Omega_c$, the gain is always
$1/\sqrt{2}$ ($-3\,\text{dB}$), regardless of order. Increasing $N$ steepens
the transition band: the asymptotic roll-off is $-20N\,\text{dB/decade}$.

## Pole placement

The $2N$ poles of $|H_a(j\Omega)|^2$ lie on a circle of radius $\Omega_c$ in
the $s$-plane, equally spaced at angles:

$$
s_k = \Omega_c \exp\!\left(j\frac{\pi(2k + N - 1)}{2N}\right),
\quad k = 0, 1, \ldots, 2N-1
$$

The stable filter $H_a(s)$ retains only the $N$ poles in the left half-plane.
Because Butterworth is an **all-pole** design (no finite zeros), the numerator
of $H_a(s)$ is simply $\Omega_c^N$.

## Properties

- **Monotonic** magnitude in both passband and stopband.
- **Maximally flat** at DC: the first $2N - 1$ derivatives of $|H|^2$ with
  respect to $\Omega^2$ are zero at $\Omega = 0$.
- Moderate group delay variation -- better than Chebyshev, worse than Bessel.
- No ripple means no overshoot in the step response (for low orders).

## Library API

The library provides design policies for all four response types:

| Class | Response |
|---|---|
| `ButterworthLowPass<N>` | Lowpass |
| `ButterworthHighPass<N>` | Highpass |
| `ButterworthBandPass<N>` | Bandpass |
| `ButterworthBandStop<N>` | Bandstop |

`N` is the maximum prototype order (compile-time upper bound on the number of
biquad stages).

### Lowpass example

```cpp
#include <sw/dsp/filter/iir/butterworth.hpp>
using namespace sw::dsp;

// 4th-order Butterworth lowpass at 1 kHz, sampled at 48 kHz
// Coefficients in double, state in double, samples in float
SimpleFilter<ButterworthLowPass<4>, double, double, float> lp;
lp.setup(4,        // order
         48000.0,  // sample rate (Hz)
         1000.0);  // cutoff frequency (Hz)

// Process samples
float y = lp.process(0.5f);
```

### Highpass example

```cpp
// Remove DC and low-frequency rumble below 80 Hz
SimpleFilter<ButterworthHighPass<2>, double, double, float> hp;
hp.setup(2, 48000.0, 80.0);
```

### Bandpass example

```cpp
// Voice-band filter: 300 Hz -- 3400 Hz
SimpleFilter<ButterworthBandPass<4>, double, double, float> bp;
bp.setup(4, 48000.0, 300.0, 3400.0);
// Produces 4 biquad sections (2N = 8 poles)
```

## Mixed-precision considerations

Butterworth coefficients are benign: no ripple means the polynomial
coefficients vary smoothly and do not cluster near representational
boundaries. A `posit<32,2>` `CoeffScalar` is typically sufficient for orders
up to 8. For higher orders ($N > 8$), the poles near Nyquist pack tightly
and `double` or `posit<64,3>` coefficients are recommended.

The all-pole structure means the numerator coefficients are simple binomial
weights, which are exactly representable in most formats. Precision pressure
concentrates in the denominator, where the $a_1$ coefficient of a biquad
section can approach $\pm 2$ for low cutoff frequencies.

## Choosing the order

A common design task: given the passband edge $f_p$, stopband edge $f_s$, and
minimum stopband attenuation $A_s$ (dB), find the minimum order:

$$
N \geq \frac{\log\!\left(10^{A_s/10} - 1\right)}{2\log\!\left(\Omega_s / \Omega_p\right)}
$$

where $\Omega_s$ and $\Omega_p$ are the prewarped frequencies.
