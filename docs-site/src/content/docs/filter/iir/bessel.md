---
title: Bessel Filters
description: Maximally flat group delay IIR filters for phase-preserving applications in sw::dsp
---

The Bessel filter (also called the **Bessel-Thomson** filter) is designed for
**maximally flat group delay** rather than maximally flat magnitude. It
approximates a linear-phase response more closely than any other classical IIR
design, making it the preferred choice when waveform shape must be preserved.

## Bessel polynomials

The denominator of the $N$th-order Bessel lowpass is the reverse Bessel
polynomial $\theta_N(s)$, defined by the recurrence:

$$
\theta_N(s) = (2N-1)\,\theta_{N-1}(s) + s^2\,\theta_{N-2}(s)
$$

with $\theta_0(s) = 1$ and $\theta_1(s) = s + 1$. The first few polynomials
are:

| Order | $\theta_N(s)$ |
|---|---|
| 1 | $s + 1$ |
| 2 | $s^2 + 3s + 3$ |
| 3 | $s^3 + 6s^2 + 15s + 15$ |
| 4 | $s^4 + 10s^3 + 45s^2 + 105s + 105$ |

## Group delay properties

The group delay of a Bessel filter is maximally flat at $\omega = 0$:

$$
\tau(\omega) = \tau_0 \left(1 + O(\omega^{2N})\right)
$$

The first $2N - 1$ derivatives of the group delay with respect to $\omega$
vanish at DC. This means the group delay is nearly constant across the
passband, so all frequency components experience the same time delay.

## Magnitude response

The price of constant group delay is a **gentle roll-off**. Compared to
Butterworth at the same order:

| Property | Butterworth | Bessel |
|---|---|---|
| $-3\,\text{dB}$ bandwidth | At design frequency | Below design frequency |
| Transition steepness | Moderate | Gentle |
| Passband flatness (magnitude) | Maximally flat | Approximately flat |
| Group delay variation | Moderate | Minimal |
| Step response overshoot | Small | Essentially none |

For a given order $N$, the Bessel filter has roughly half the stopband
attenuation of the Butterworth at the same frequency offset from cutoff.

## Applications

Bessel filters excel in scenarios where phase matters more than selectivity:

- **Audio monitoring** -- preserving transient shape in headphone amplifiers
  and studio monitors.
- **Instrumentation** -- anti-aliasing filters for oscilloscopes and data
  acquisition where pulse shape must be preserved.
- **Control systems** -- loop filters where phase margin is critical.
- **Pulse shaping** -- Gaussian approximation in communication systems (the
  Bessel impulse response closely approximates a Gaussian).

## Library API

```cpp
#include <sw/dsp/filter/iir/bessel.hpp>
using namespace sw::dsp;

// 4th-order Bessel lowpass at 5 kHz
SimpleFilter<BesselLowPass<4>, double, double, float> lp;
lp.setup(4,        // order
         48000.0,  // sample rate (Hz)
         5000.0);  // cutoff frequency (Hz)

float y = lp.process(1.0f);  // step response
```

Available design policies:

| Class | Response |
|---|---|
| `BesselLowPass<N>` | Lowpass |
| `BesselHighPass<N>` | Highpass |
| `BesselBandPass<N>` | Bandpass |
| `BesselBandStop<N>` | Bandstop |

### Bandpass example

```cpp
// Bessel bandpass preserving phase across 500--4000 Hz
SimpleFilter<BesselBandPass<3>, double, double, float> bp;
bp.setup(3, 48000.0, 500.0, 4000.0);
```

## Bilinear transform and group delay

The bilinear transform introduces frequency warping, which distorts the
group delay of the analog prototype. For Bessel filters this is particularly
noticeable because the entire design goal is delay flatness. Two mitigations:

1. **Oversample** -- use a sample rate much higher than the cutoff so that the
   warping region is far from the passband.
2. **Matched-$z$ design** -- an alternative mapping that better preserves
   delay characteristics (not yet in the library).

## Mixed-precision notes

Bessel filters are numerically well-behaved. The poles are far from the unit
circle (gentle roll-off means wide pole spacing), so the biquad coefficients
are not sensitive to quantization. Even `posit<16,1>` coefficients work
acceptably for orders up to 4. For higher orders, `posit<32,2>` or `float`
is sufficient.

The gentle magnitude response also means the internal state values stay small,
reducing the dynamic range requirements on `StateScalar`.
