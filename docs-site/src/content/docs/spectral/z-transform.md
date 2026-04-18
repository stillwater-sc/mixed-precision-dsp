---
title: Z-Transform
description: Z-transform theory, transfer functions, poles and zeros, and stability analysis
---

The Z-transform is the fundamental tool for analyzing discrete-time
linear systems. It maps a sequence $x[n]$ into a function of the complex
variable $z$, enabling algebraic manipulation of difference equations.

## Definition

The (unilateral) Z-transform of a causal sequence $x[n]$ is:

$$
X(z) = \sum_{n=0}^{\infty} x[n]\, z^{-n}
$$

Each term $x[n] z^{-n}$ associates the sample $x[n]$ with a power of $z^{-1}$,
which represents a unit delay in the time domain.

## Region of convergence

The Z-transform converges only for values of $z$ where the sum is
absolutely convergent. The **region of convergence (ROC)** is the set of
all such $z$ in the complex plane. For a causal, stable system the ROC
is the exterior of a circle that contains all poles:

$$
|z| > \max_i |p_i|
$$

where $p_i$ are the poles of $X(z)$.

## Relationship to the DTFT

The Discrete-Time Fourier Transform (DTFT) is obtained by evaluating
the Z-transform on the unit circle:

$$
X(e^{j\omega}) = X(z)\big|_{z = e^{j\omega}}
$$

This relationship means that the frequency response of a digital filter
is its transfer function evaluated at $z = e^{j\omega}$.

## Transfer function

A linear time-invariant (LTI) discrete-time system with input $X(z)$ and
output $Y(z)$ is described by the transfer function:

$$
H(z) = \frac{Y(z)}{X(z)} = \frac{B(z)}{A(z)}
  = \frac{b_0 + b_1 z^{-1} + \cdots + b_M z^{-M}}{1 + a_1 z^{-1} + \cdots + a_N z^{-N}}
$$

The numerator coefficients $b_k$ determine the zeros and the denominator
coefficients $a_k$ determine the poles of the system.

## Poles and zeros

Factoring $H(z)$ gives the **pole-zero form**:

$$
H(z) = b_0 \frac{\prod_{k=1}^{M}(1 - q_k z^{-1})}{\prod_{k=1}^{N}(1 - p_k z^{-1})}
$$

- **Zeros** ($q_k$): values of $z$ where $H(z) = 0$. They create nulls
  in the frequency response.
- **Poles** ($p_k$): values of $z$ where $H(z) \to \infty$. They create
  peaks in the frequency response.

### Stability criterion

A causal LTI system is **stable** if and only if all poles lie strictly
inside the unit circle:

$$
|p_k| < 1 \quad \text{for all } k
$$

Poles on or outside the unit circle produce unbounded output for bounded
input. The library's stability analysis module checks this condition
directly from biquad coefficients.

## Library API

### Evaluating the transfer function

The `ztransform_eval()` function evaluates $H(z)$ at arbitrary points
in the complex plane:

```cpp
#include <sw/dsp/spectral/ztransform.hpp>

using namespace sw::dsp;

// Define a second-order system: H(z) = (b0 + b1*z^-1 + b2*z^-2)
//                                     / (1  + a1*z^-1 + a2*z^-2)
std::array<double, 3> b = {0.0675, 0.1349, 0.0675};  // numerator
std::array<double, 3> a = {1.0, -1.1430, 0.4128};    // denominator

// Evaluate on the unit circle at 64 equally spaced frequencies
size_t nfreqs = 64;
auto H = spectral::ztransform_eval(b, a, nfreqs);
```

### Frequency response

To compute the magnitude and phase response of a filter:

```cpp
auto [magnitude, phase] = spectral::freqz(b, a, 512);
```

This evaluates $H(e^{j\omega})$ at 512 frequencies from $0$ to $\pi$.

### Pole-zero analysis

The library integrates with the analysis module to extract poles and
zeros and check stability:

```cpp
#include <sw/dsp/analysis/stability.hpp>

// Extract poles from biquad coefficients
auto poles = analysis::extract_poles(a1, a2);

// Check if all poles are inside the unit circle
bool stable = analysis::is_stable(poles);
```

### Mixed-precision considerations

When filter coefficients are stored in a narrow type such as
`posit<8,0>`, quantization can push poles outside the unit circle and
make a nominally stable filter unstable. The Z-transform evaluation
provides a direct way to verify stability after quantization:

```cpp
using Narrow = sw::universal::posit<8, 0>;

// Quantize coefficients
std::array<Narrow, 3> a_q = {Narrow(1.0), Narrow(-1.143), Narrow(0.4128)};

// Check pole locations after quantization
auto poles_q = analysis::extract_poles(double(a_q[1]), double(a_q[2]));
bool still_stable = analysis::is_stable(poles_q);
```

This analysis is essential when selecting arithmetic types for
resource-constrained deployments.
