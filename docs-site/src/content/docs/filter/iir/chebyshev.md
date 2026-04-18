---
title: Chebyshev Filters
description: Type I (equiripple passband) and Type II (equiripple stopband) Chebyshev IIR filter design in sw::dsp
---

Chebyshev filters trade monotonicity for a **sharper transition band** at the
same order as Butterworth. Two variants exist, each placing the equiripple
behavior in a different region.

## Chebyshev Type I (equiripple passband)

The squared magnitude response is:

$$
|H(j\Omega)|^2 = \frac{1}{1 + \epsilon^2\, T_N^2\!\left(\frac{\Omega}{\Omega_c}\right)}
$$

where $T_N$ is the Chebyshev polynomial of the first kind of degree $N$, and
$\epsilon$ controls the passband ripple. The passband ripple in decibels is:

$$
R_p = 10\log_{10}(1 + \epsilon^2) \quad\text{dB}
$$

### Properties

- Equiripple oscillation of magnitude within the passband.
- Monotonically decreasing magnitude in the stopband.
- Steeper transition band than Butterworth for the same order and ripple budget.
- Poles lie on an **ellipse** in the $s$-plane (not a circle), with semi-axes
  determined by $\epsilon$ and $N$.

### Pole placement

The $N$ left-half-plane poles are:

$$
s_k = \Omega_c\bigl(-\sinh\alpha\,\sin\theta_k + j\cosh\alpha\,\cos\theta_k\bigr)
$$

where $\theta_k = \frac{\pi(2k+1)}{2N}$ and
$\alpha = \frac{1}{N}\text{arcsinh}\!\left(\frac{1}{\epsilon}\right)$.

## Chebyshev Type II (equiripple stopband)

Also called the **inverse Chebyshev** filter. The squared magnitude is:

$$
|H(j\Omega)|^2 = \frac{1}{1 + \left[\epsilon^2\, T_N^2\!\left(\frac{\Omega_s}{\Omega}\right)\right]^{-1}}
$$

### Properties

- Monotonically decreasing in the passband (no passband ripple).
- Equiripple in the stopband with a specified minimum attenuation.
- Has both poles and finite zeros (zeros on the $j\Omega$ axis create the
  stopband notches).
- Less commonly used than Type I, but valuable when passband flatness matters
  more than transition steepness.

## Comparison

| Property | Type I | Type II |
|---|---|---|
| Passband | Equiripple | Monotonic |
| Stopband | Monotonic | Equiripple |
| Finite zeros | No | Yes |
| Transition sharpness (same order) | Sharper | Slightly less sharp |
| Group delay variation | Higher | Lower |

## Library API

### Type I

```cpp
#include <sw/dsp/filter/iir/chebyshev.hpp>
using namespace sw::dsp;

// 6th-order Chebyshev Type I lowpass
// 1 dB passband ripple, cutoff at 2 kHz
SimpleFilter<ChebyshevILowPass<6>, double, double, float> lp;
lp.setup(6,        // order
         48000.0,  // sample rate
         2000.0,   // cutoff frequency
         1.0);     // passband ripple (dB)
```

Available design policies:

| Class | Response |
|---|---|
| `ChebyshevILowPass<N>` | Lowpass |
| `ChebyshevIHighPass<N>` | Highpass |
| `ChebyshevIBandPass<N>` | Bandpass |
| `ChebyshevIBandStop<N>` | Bandstop |

### Type II

```cpp
// 6th-order Chebyshev Type II lowpass
// 60 dB stopband attenuation, stopband edge at 3 kHz
SimpleFilter<ChebyshevIILowPass<6>, double, double, float> lp;
lp.setup(6,        // order
         48000.0,  // sample rate
         3000.0,   // stopband edge frequency
         60.0);    // minimum stopband attenuation (dB)
```

| Class | Response |
|---|---|
| `ChebyshevIILowPass<N>` | Lowpass |
| `ChebyshevIIHighPass<N>` | Highpass |
| `ChebyshevIIBandPass<N>` | Bandpass |
| `ChebyshevIIBandStop<N>` | Bandstop |

## Mixed-precision notes

Type I filters have poles closer to the unit circle than Butterworth at the
same order, which makes the denominator coefficients $a_1$ more sensitive to
quantization. For ripple values below 0.5 dB with orders above 6, use
`double` or `posit<32,2>` coefficients.

Type II filters additionally have finite zeros, so the numerator coefficients
$b_1, b_2$ also carry precision requirements. The zeros sit on the unit
circle in the stopband, and small perturbations shift the notch frequencies.

## Choosing between types

- Use **Type I** when you need the sharpest possible transition and can
  tolerate passband ripple (e.g., anti-aliasing before decimation).
- Use **Type II** when the passband must be flat and you can tolerate a
  slightly wider transition band (e.g., audio equalization).
