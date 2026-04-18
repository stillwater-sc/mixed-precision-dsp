---
title: Filter Design Overview
description: Introduction to digital filter design concepts and the mixed-precision three-scalar parameterization in sw::dsp
---

Digital filters are the core building blocks of any DSP system. They
selectively modify the frequency content of a signal, suppressing unwanted
components while preserving the ones that matter.

## FIR vs IIR

Filters fall into two families:

| Property | FIR | IIR |
|---|---|---|
| Impulse response | Finite length | Infinite length |
| Stability | Always stable | Requires pole placement inside the unit circle |
| Linear phase | Achievable with symmetric coefficients | Not achievable in general |
| Order for sharp cutoff | High (tens to hundreds of taps) | Low (typically 2--12) |
| Computational cost | $O(N)$ per sample | $O(N)$ per sample, but $N$ is much smaller |

## Magnitude, phase, and group delay

A causal LTI filter with transfer function $H(z)$ evaluated on the unit circle
$z = e^{j\omega}$ gives the **frequency response** $H(e^{j\omega})$. This
decomposes into:

- **Magnitude response** $|H(e^{j\omega})|$ -- the gain at each frequency.
- **Phase response** $\angle H(e^{j\omega})$ -- the phase shift at each frequency.
- **Group delay** $\tau(\omega) = -\frac{d}{d\omega}\angle H(e^{j\omega})$ --
  the time delay experienced by each frequency component.

Linear phase ($\angle H = -\alpha\omega$) means constant group delay, which
preserves waveform shape. FIR filters with symmetric or antisymmetric
coefficients achieve this exactly. IIR filters trade linear phase for
dramatically lower order.

## The analog-to-digital design pipeline

Classical IIR filter design follows a four-stage pipeline:

1. **Analog prototype** -- Design a normalized lowpass filter in the
   continuous-time $s$-domain (e.g., Butterworth, Chebyshev, Elliptic, Bessel).
2. **Frequency transformation** -- Map the prototype lowpass to the target
   response type (highpass, bandpass, bandstop) using Constantinides
   transformations.
3. **Bilinear $z$-transform** -- Map from the $s$-plane to the $z$-plane via
   $s = 2f_s \frac{z-1}{z+1}$, with prewarping to correct frequency compression.
4. **Biquad cascade** -- Factor the resulting transfer function into
   second-order sections for numerically robust implementation.

Each stage is a separate, composable operation in the library.

## The three-scalar model

Every filter in `sw::dsp` is parameterized on three independent scalar types:

| Scalar | Role | Typical choice |
|---|---|---|
| `CoeffScalar` | Filter coefficients (design precision) | `double`, `posit<32,2>` |
| `StateScalar` | Internal accumulator state (processing precision) | `double`, `posit<64,3>` |
| `SampleScalar` | Input/output samples (streaming precision) | `float`, `posit<16,1>` |

This separation lets you:

- **Design** coefficients in high precision to avoid quantization-induced
  frequency response distortion.
- **Accumulate** partial products in wider arithmetic to prevent overflow and
  reduce round-off noise.
- **Stream** samples in narrow, bandwidth-efficient formats when the signal
  itself does not need full precision.

```cpp
#include <sw/dsp/dsp.hpp>
using namespace sw::dsp;

// 4th-order Butterworth lowpass
// Coefficients in double, state in double, samples in float
SimpleFilter<ButterworthLowPass<4>, double, double, float> lp;
lp.setup(4, 48000.0, 1000.0);   // order, fs, fc

float x = 0.5f;
float y = lp.process(x);
```

## `SimpleFilter<Design>`

`SimpleFilter` is a convenience wrapper that owns a biquad cascade and
provides a single `process()` entry point. It is parameterized on:

- A **design policy** (e.g., `ButterworthLowPass<N>`).
- The three scalar types.

The design policy supplies the analog prototype and frequency transformation.
`SimpleFilter::setup()` runs the full pipeline and populates the internal
cascade. After setup, call `process(sample)` for each input sample.

```cpp
// Bandpass Chebyshev Type I, 6th order
SimpleFilter<ChebyshevIBandPass<6>, double, double, float> bp;
bp.setup(6, 48000.0, 300.0, 3400.0, 1.0);
// order, fs, f_low, f_high, ripple_dB
```

## Choosing a filter type

| Need | Recommended |
|---|---|
| Flattest passband | Butterworth |
| Sharpest transition band | Elliptic |
| Controlled passband ripple, sharp roll-off | Chebyshev Type I |
| Controlled stopband floor | Chebyshev Type II |
| Constant group delay | Bessel |
| Exact linear phase | FIR (windowed sinc or Parks-McClellan) |

Subsequent pages cover each design in detail, starting with the biquad
building block and the bilinear transform that underpins the entire IIR
pipeline.
