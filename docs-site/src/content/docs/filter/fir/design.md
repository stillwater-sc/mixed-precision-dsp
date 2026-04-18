---
title: FIR Design Methods
description: Windowed sinc, Parks-McClellan, and linear phase FIR filter design in sw::dsp
---

Finite Impulse Response (FIR) filters implement the convolution sum directly:

$$
y[n] = \sum_{k=0}^{M-1} h[k]\, x[n-k]
$$

where $h[k]$ is the impulse response of length $M$. Unlike IIR filters, FIR
filters have no feedback, which guarantees two critical properties: **absolute
stability** and the possibility of **exact linear phase**.

## Advantages and disadvantages

| Advantage | Explanation |
|---|---|
| Guaranteed stability | All poles at $z = 0$ (inside unit circle) |
| Exact linear phase | Achievable with symmetric coefficients |
| No limit cycles | No feedback means no autonomous oscillation |
| Simple fixed-point design | No recursive accumulation of round-off |

| Disadvantage | Explanation |
|---|---|
| Higher order | Sharp transitions require hundreds of taps |
| Latency | Group delay is $(M-1)/2$ samples for linear phase |
| Computational cost | $M$ multiplications per sample |

## Linear phase types

FIR filters with symmetric or antisymmetric impulse responses have exactly
linear phase. Four types exist, classified by length (even/odd) and symmetry:

| Type | Length | Symmetry | $h[n] = $ | Suitable for |
|---|---|---|---|---|
| I | Odd ($M = 2L+1$) | Symmetric | $h[M-1-n]$ | Any filter type |
| II | Even ($M = 2L$) | Symmetric | $h[M-1-n]$ | LP, BP (zero at $\omega = \pi$) |
| III | Odd | Antisymmetric | $-h[M-1-n]$ | BP, differentiators (zeros at $0$ and $\pi$) |
| IV | Even | Antisymmetric | $-h[M-1-n]$ | HP, Hilbert (zero at $\omega = 0$) |

## Windowed sinc method

The ideal lowpass impulse response is a sinc function:

$$
h_{\text{ideal}}[n] = \frac{\sin(\omega_c n)}{\pi n}
$$

which has infinite length. The **windowed sinc** method truncates and tapers
this with a window function $w[n]$:

$$
h[n] = h_{\text{ideal}}[n] \cdot w[n]
$$

Common windows and their properties:

| Window | Main lobe width | Sidelobe level | Transition width |
|---|---|---|---|
| Rectangular | Narrowest | $-13\,\text{dB}$ | Narrowest |
| Hamming | Moderate | $-43\,\text{dB}$ | Moderate |
| Blackman | Wide | $-58\,\text{dB}$ | Wide |
| Kaiser | Adjustable ($\beta$) | Adjustable | Adjustable |

The Kaiser window is the most flexible: the parameter $\beta$ trades
transition width for sidelobe suppression continuously.

```cpp
#include <sw/dsp/filter/fir/windowed_sinc.hpp>
using namespace sw::dsp;

// 63-tap lowpass FIR, Kaiser window, cutoff at 4 kHz
auto h = windowed_sinc_lowpass<double>(
    63,       // number of taps
    48000.0,  // sample rate
    4000.0,   // cutoff frequency
    WindowType::Kaiser,
    5.0       // Kaiser beta
);
```

## Parks-McClellan (equiripple) method

The **Parks-McClellan** algorithm (also called the Remez exchange algorithm)
designs FIR filters that minimize the maximum weighted error (minimax /
Chebyshev criterion) over specified frequency bands:

$$
\min_{h} \max_{\omega \in \text{bands}} W(\omega)\,\bigl|D(\omega) - H(e^{j\omega})\bigr|
$$

where $D(\omega)$ is the desired response and $W(\omega)$ is a weighting
function.

This produces **equiripple** filters: the error oscillates between equal
positive and negative extremes in each band.

```cpp
#include <sw/dsp/filter/fir/parks_mcclellan.hpp>
using namespace sw::dsp;

// Equiripple lowpass: passband 0--4 kHz, stopband 5--24 kHz
auto h = parks_mcclellan<double>(
    51,                         // number of taps
    48000.0,                    // sample rate
    {0.0, 4000.0, 5000.0, 24000.0},  // band edges
    {1.0, 1.0, 0.0, 0.0},            // desired gains
    {1.0, 1.0}                        // band weights
);
```

## FIR filter class

The library provides `FIRFilter<T, MaxTaps>` for applying a designed impulse
response:

```cpp
#include <sw/dsp/filter/fir/fir_filter.hpp>
using namespace sw::dsp;

FIRFilter<double, 64> fir;
fir.setCoefficients(h);   // h from windowed_sinc or parks_mcclellan

double y = fir.process(x);
```

`FIRFilter` uses a circular buffer internally to avoid shifting the delay
line on every sample. The `MaxTaps` template parameter sets the compile-time
buffer size.

## Choosing FIR vs IIR

Use FIR when:

- Linear phase is mandatory (audio mastering, measurement).
- Stability must be guaranteed by construction (safety-critical systems).
- The transition band is not extremely narrow relative to the sample rate.

Use IIR when:

- Sharp transitions are needed with minimal latency.
- Phase linearity is not required.
- Computational budget is tight (a 4th-order IIR replaces a 50+ tap FIR).
