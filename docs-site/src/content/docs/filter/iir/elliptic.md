---
title: Elliptic (Cauer) Filters
description: Equiripple passband and stopband IIR filters with the steepest possible transition band, including both DSPFilters-style and Matlab-style APIs
---

The **elliptic** filter (also called a **Cauer** filter) achieves the steepest
transition band of any classical IIR design for a given order. It does so by
distributing approximation error as equiripple in **both** the passband and the
stopband.

## Magnitude response

The squared magnitude is:

$$
|H(j\Omega)|^2 = \frac{1}{1 + \epsilon^2\, R_N^2\!\left(\xi, \frac{\Omega}{\Omega_c}\right)}
$$

where $R_N(\xi, x)$ is a rational Chebyshev function constructed from Jacobi
elliptic functions, $\epsilon$ controls passband ripple, and $\xi$ is the
**selectivity factor** that sets the stopband edge relative to the passband
edge.

## Elliptic integrals and Jacobi functions

The design relies on the **complete elliptic integral of the first kind**:

$$
K(k) = \int_0^{\pi/2} \frac{d\theta}{\sqrt{1 - k^2 \sin^2\theta}}
$$

and the Jacobi elliptic sine $\operatorname{sn}(u, k)$. The zeros of $H(s)$
are placed on the $j\Omega$ axis at positions determined by
$\operatorname{sn}$ evaluations, creating the stopband notches.

The **degree equation** relates order, selectivity, and discrimination:

$$
N = \frac{K(k)\, K'(k_1)}{K'(k)\, K(k_1)}
$$

where $k = \Omega_p / \Omega_s$ is the selectivity parameter,
$k_1 = \epsilon_p / \epsilon_s$ is the discrimination parameter, and primes
denote the complementary integrals $K'(k) = K(\sqrt{1-k^2})$.

## Properties

- **Equiripple** in both passband and stopband.
- Steepest transition of any polynomial/rational filter at a given order.
- Both poles and finite zeros.
- Worst group delay variation of the classical designs.
- Optimal in the Chebyshev (minimax) sense.

## Library API -- two styles

### Style 1: DSPFilters-style (rolloff parameter)

This API parameterizes the stopband behavior with a **rolloff** factor
(0 to 1) that interpolates between Chebyshev Type I (rolloff = 0) and
maximum selectivity (rolloff = 1):

```cpp
#include <sw/dsp/filter/iir/elliptic.hpp>
using namespace sw::dsp;

// 4th-order elliptic lowpass
SimpleFilter<EllipticLowPass<4>, double, double, float> lp;
lp.setup(4,        // order
         48000.0,  // sample rate
         1000.0,   // cutoff frequency
         1.0,      // passband ripple (dB)
         0.5);     // rolloff factor [0, 1]
```

| Class | Response |
|---|---|
| `EllipticLowPass<N>` | Lowpass |
| `EllipticHighPass<N>` | Highpass |
| `EllipticBandPass<N>` | Bandpass |
| `EllipticBandStop<N>` | Bandstop |

### Style 2: Matlab/scipy-style (Cauer-Darlington specification)

This API accepts explicit passband ripple $A_p$, stopband attenuation $A_s$,
passband edge $f_p$, and stopband edge $f_s$:

```cpp
#include <sw/dsp/filter/iir/elliptic.hpp>
using namespace sw::dsp;

// Specification-driven elliptic lowpass
SimpleFilter<EllipticLowPassSpec<8>, double, double, float> lp;
lp.setup(48000.0,   // sample rate
         1000.0,    // passband edge fp (Hz)
         1200.0,    // stopband edge fs (Hz)
         0.5,       // passband ripple Ap (dB)
         60.0);     // stopband attenuation As (dB)
```

The `setup()` call internally computes the minimum order required to meet the
specification using the degree equation.

### Minimum order calculation

The free function `elliptic_minimum_order()` solves the degree equation:

```cpp
#include <sw/dsp/filter/iir/elliptic.hpp>
using namespace sw::dsp;

// What order do I need?
auto N = elliptic_minimum_order(
    0.5,     // passband ripple Ap (dB)
    60.0,    // stopband attenuation As (dB)
    1000.0,  // passband edge (Hz)
    1200.0,  // stopband edge (Hz)
    48000.0  // sample rate
);
// N is typically 5 or 6 for this specification
```

## When to use elliptic filters

Elliptic filters are ideal when:

- The transition band must be as narrow as possible (e.g., channelizers,
  anti-aliasing filters with tight spectral budgets).
- Both passband ripple and stopband ripple are acceptable.
- Group delay linearity is not critical (or will be equalized separately).

Avoid elliptic filters when:

- Flat passband is required (use Butterworth).
- Constant group delay is needed (use Bessel).
- The application is sensitive to phase distortion (e.g., audio monitoring).

## Mixed-precision notes

Elliptic filters concentrate the most precision pressure of any classical
design. The zeros on the $j\Omega$ axis map to zeros near the unit circle
after the bilinear transform. Small coefficient perturbations shift these
zeros, degrading stopband rejection. For stopband requirements beyond 60 dB,
use `double` or wider `CoeffScalar` types. The `StateScalar` should also be
wide because the sharp notches create large transient peaks in the internal
state.
