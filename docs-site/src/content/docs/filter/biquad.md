---
title: Biquad Sections
description: Second-order IIR building blocks -- transfer function, realization forms, and the library's BiquadCoefficients and Cascade types
---

A **biquad** (bi-quadratic) section is a second-order IIR filter. Every
higher-order IIR filter in `sw::dsp` is implemented as a cascade of biquads,
because second-order sections offer superior numerical behavior compared to a
single high-order direct form.

## Transfer function

The $z$-domain transfer function of a single biquad is:

$$
H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{1 + a_1 z^{-1} + a_2 z^{-2}}
$$

The numerator coefficients $(b_0, b_1, b_2)$ set the zeros; the denominator
coefficients $(a_1, a_2)$ set the poles. Stability requires both poles to lie
inside the unit circle, which for a second-order section means $|a_2| < 1$ and
$|a_1| < 1 + a_2$.

## Realization forms

### Direct Form I

Two separate delay lines -- one for the feedforward (FIR) path and one for the
feedback path:

$$
y[n] = b_0\,x[n] + b_1\,x[n-1] + b_2\,x[n-2]
       - a_1\,y[n-1] - a_2\,y[n-2]
$$

- Requires four state variables ($x[n-1], x[n-2], y[n-1], y[n-2]$).
- No intermediate large values when input is bounded.
- Straightforward for fixed-point because the output is computed in a single
  multiply-accumulate pass.

### Direct Form II

A single shared delay line $w[n]$:

$$
w[n] = x[n] - a_1\,w[n-1] - a_2\,w[n-2]
$$

$$
y[n] = b_0\,w[n] + b_1\,w[n-1] + b_2\,w[n-2]
$$

- Only two state variables.
- The intermediate signal $w[n]$ can overflow when poles are near the unit
  circle, even if input and output are bounded. This makes DF-II fragile in
  narrow arithmetic.

### Direct Form II Transposed

The library's default realization. State update:

$$
y[n] = b_0\,x[n] + s_1[n-1]
$$

$$
s_1[n] = b_1\,x[n] - a_1\,y[n] + s_2[n-1]
$$

$$
s_2[n] = b_2\,x[n] - a_2\,y[n]
$$

- Two state variables, same as DF-II.
- Superior numerical behavior: the state variables represent partial output
  sums rather than an intermediate signal, so they stay bounded when the output
  is bounded.
- Preferred form for floating-point and posit implementations.

## Cascading biquads

An $N$th-order transfer function $H(z)$ is factored into $\lceil N/2 \rceil$
second-order sections:

$$
H(z) = \prod_{k=1}^{K} H_k(z)
$$

Section ordering and gain distribution affect dynamic range. The library
orders sections by increasing pole radius (innermost poles first) to minimize
intermediate signal peaks.

## Library types

### `BiquadCoefficients<T>`

Stores the five normalized coefficients for one section:

```cpp
template<typename T>
struct BiquadCoefficients {
    T b0, b1, b2;   // numerator
    T a1, a2;        // denominator (a0 = 1 implied)
};
```

The denominator is normalized so that $a_0 = 1$ is never stored. Arithmetic
conversions between scalar types use explicit `static_cast`, so you can design
in `double` and quantize to `posit<16,1>` for deployment.

### `Cascade<T, MaxStages>`

A fixed-capacity container of biquad sections with associated state:

```cpp
template<typename CoeffT, typename StateT, std::size_t MaxStages>
struct Cascade {
    std::array<BiquadCoefficients<CoeffT>, MaxStages> stages;
    std::array<StateT, MaxStages * 2> state;  // DF-II transposed
    std::size_t numStages;

    template<typename SampleT>
    SampleT process(SampleT x);
};
```

`MaxStages` is a compile-time upper bound (typically `N/2` rounded up).
`numStages` tracks how many stages are actually in use. The `process()` method
feeds each sample through every active stage in sequence, using `StateT` for
all accumulation and reading/writing `SampleT` at the boundaries.

### Example

```cpp
#include <sw/dsp/filter/biquad.hpp>
using namespace sw::dsp;

// Manually populate a single biquad (lowpass, fc ~ fs/4)
BiquadCoefficients<double> coeffs;
coeffs.b0 =  0.0675;
coeffs.b1 =  0.1349;
coeffs.b2 =  0.0675;
coeffs.a1 = -1.1430;
coeffs.a2 =  0.4128;

Cascade<double, double, 1> cascade;
cascade.stages[0] = coeffs;
cascade.numStages = 1;

double y = cascade.process(1.0);  // step response, first sample
```

In practice you rarely populate coefficients by hand. The IIR design classes
fill the cascade automatically during `setup()`.
