---
title: Numerical Pitfalls in DSP
description: Critical numerical issues that mixed-precision arithmetic exposes in digital filter implementations, and how to diagnose them
---

Mixed-precision arithmetic amplifies numerical issues that already exist
in fixed-precision DSP but are masked by the generous headroom of 64-bit
`double`. When you narrow a coefficient or state type, these problems
surface quickly. Understanding them is essential to choosing precision
wisely rather than blindly.

## 1. Pole displacement under coefficient quantization

A biquad section has the denominator polynomial $z^2 + a_1 z + a_2 = 0$,
whose roots are the filter's poles:

$$
p_{1,2} = \frac{-a_1 \pm \sqrt{a_1^2 - 4 a_2}}{2}
$$

When coefficients are quantized from a wide type to a narrow one, $a_1$
and $a_2$ shift by small amounts $\Delta a_1$ and $\Delta a_2$. The
resulting pole displacement is:

$$
\Delta p_i \;\approx\; \frac{\partial p_i}{\partial a_1}\,\Delta a_1
\;+\; \frac{\partial p_i}{\partial a_2}\,\Delta a_2
$$

The partial derivatives for a biquad are:

$$
\frac{\partial p_1}{\partial a_1} = \frac{-1}{p_1 - p_2}, \qquad
\frac{\partial p_1}{\partial a_2} = \frac{p_1}{p_1 - p_2}
$$

When poles are close together ($p_1 \approx p_2$), the denominator
$p_1 - p_2$ is small and the sensitivity **explodes**. This is precisely
the situation in narrowband or high-Q filters, where conjugate poles
cluster near the unit circle.

**Example**: A pole at radius $r = 0.999$ in `double` is 0.001 from
the unit circle. If a `fixpnt<16,14>` coefficient type has a quantum of
$2^{-14} \approx 6.1 \times 10^{-5}$, the pole can shift by a comparable
amount. If the shift pushes $r$ past 1.0, the filter becomes **unstable** --
the output grows without bound.

```cpp
#include <sw/dsp/analysis/sensitivity.hpp>
using namespace sw::dsp;

BiquadCoefficients<double> bq{/* designed coefficients */};
auto sens = coefficient_sensitivity(bq);

// sens.dp_da1 and sens.dp_da2 tell you how fragile this section is
if (std::abs(sens.dp_da1) > 100.0) {
    // This section is highly sensitive to a1 quantization
}
```

## 2. Limit cycles in IIR filters

When the state type has limited precision, the feedback loop in an IIR
filter can enter a **limit cycle** -- a persistent low-amplitude
oscillation that continues even after the input goes to zero.

The mechanism is straightforward. Consider the first-order recursion:

$$
y[n] = -a_1\, y[n{-}1]
$$

With infinite precision, if $|a_1| < 1$, then $y[n] \to 0$. But if
the state is quantized after each step, the sequence can get trapped
in a repeating pattern. For example, with rounding to nearest:

$$
y[n] = \text{round}(-a_1 \cdot y[n{-}1])
$$

If $-a_1 \cdot y[n{-}1]$ rounds to $y[n{-}1]$ itself, the output
never decays. In second-order sections, limit cycles can produce
small sinusoidal oscillations at the pole frequency.

**Impact by type**:

| Type | Limit cycle behavior |
|------|---------------------|
| `double` | Vanishingly rare (52-bit mantissa) |
| `float` | Rare but possible in high-Q designs |
| `fixpnt<16,14>` | Common in narrowband filters |
| `posit<16,1>` | Less common (tapered precision helps near zero) |
| `integer<8>` | Almost guaranteed in any IIR filter |

Posit types mitigate limit cycles better than equal-width fixed-point
because their tapered precision provides finer resolution near zero,
exactly where the decaying signal needs it.

## 3. Overflow in narrow accumulators

An FIR filter computes:

$$
y[n] = \sum_{k=0}^{L-1} h[k]\, x[n{-}k]
$$

If $L = 128$ taps and each product $h[k] \cdot x[n{-}k]$ can be as large
as 1.0, the sum can reach 128.0. An accumulator type with a maximum value
of 65,504 (`half`) can hold this, but a `posit<8,2>` with a maximum of
64 cannot.

The danger is that overflow behavior differs by type:

- **IEEE floats** saturate to $\pm\infty$ (detectable but destructive).
- **Fixed-point** wraps around silently (catastrophic).
- **Posits** saturate to $\pm\text{maxpos}$ (bounded but lossy).

The state type must have enough dynamic range for the worst-case
accumulation. For an FIR with $L$ taps, the minimum dynamic range is:

$$
\text{DR}_{\min} = 20 \log_{10}(L \cdot \max|h[k]|) \;\text{dB}
$$

## 4. Catastrophic cancellation

When two nearly equal numbers are subtracted, the result loses most of
its significant bits. This is called **catastrophic cancellation** and
it is endemic in high-Q biquad sections.

Consider a biquad with $a_1 = -1.998$ and $a_2 = 0.999$. The feedback
computation includes:

$$
-a_1 \cdot y[n{-}1] - a_2 \cdot y[n{-}2] \approx 1.998 \cdot y - 0.999 \cdot y = 0.999 \cdot y
$$

The two terms being subtracted are nearly equal in magnitude. If each
has $p$ bits of precision, the difference retains only a few bits. In
`half` (10-bit mantissa), the subtraction $1.998 y - 0.999 y$ can
lose 9 of the 10 significant bits.

This is why the **state type** needs more precision than the sample type:
the state must survive cancellation at every sample step.

## 5. Coefficient sensitivity in high-order filters

A direct-form implementation of an $N$th-order filter has a single
polynomial:

$$
H(z) = \frac{\sum_{k=0}^{N} b_k z^{-k}}{1 + \sum_{k=1}^{N} a_k z^{-k}}
$$

The sensitivity of the poles to the coefficients $a_k$ grows
**combinatorially** with order. For a 10th-order filter, a 1-ULP change
in $a_{10}$ can shift all 10 poles simultaneously.

The **cascade of biquads** factorization avoids this. Each second-order
section controls only its own pair of poles:

$$
H(z) = \prod_{i=1}^{N/2} \frac{b_{0,i} + b_{1,i} z^{-1} + b_{2,i} z^{-2}}{1 + a_{1,i} z^{-1} + a_{2,i} z^{-2}}
$$

Quantizing one section's coefficients shifts only that section's poles.
The other poles are unaffected. This is why `sw::dsp` implements all
IIR filters as cascades of `BiquadCoefficients<T>`, never as
direct-form polynomials.

| Implementation | Pole sensitivity | Suitable for narrow types |
|----------------|-----------------|---------------------------|
| Direct form, order 2 | Low | Yes |
| Direct form, order 8 | Very high | No |
| Cascade of 4 biquads | Low (per section) | Yes |

## 6. Diagnosing issues with the analysis module

The library provides three diagnostic functions in `<sw/dsp/analysis/>`:

### `stability_margin(cascade)`

Returns $1 - \max_i |p_i|$, the distance from the nearest pole to the
unit circle. A margin of 0.0 means the filter is on the boundary of
instability. Negative means unstable.

```cpp
#include <sw/dsp/analysis/stability.hpp>
using namespace sw::dsp;

auto cascade = lp.cascade();
double margin = stability_margin(cascade);

// margin > 0.01 is comfortable
// margin < 0.001 is fragile under quantization
```

### `coefficient_sensitivity(biquad)`

Returns the partial derivatives $\partial |p| / \partial a_1$ and
$\partial |p| / \partial a_2$ for a single biquad section. Large values
mean the section is fragile.

```cpp
#include <sw/dsp/analysis/sensitivity.hpp>

auto sens = coefficient_sensitivity(cascade.stage(0));
// sens.dp_da1, sens.dp_da2
```

### `biquad_condition_number(biquad, num_freqs)`

Estimates how sensitive the frequency response is to small coefficient
perturbations, sampled across `num_freqs` frequency points. A condition
number above 100 suggests the section will behave differently in narrow
arithmetic.

```cpp
#include <sw/dsp/analysis/condition.hpp>

double cn = biquad_condition_number(cascade.stage(0), 256);
```

## Type comparison: where each excels and fails

| Scenario | `posit<16,1>` | `fixpnt<16,14>` | `cfloat<16,5,…>` (half) |
|----------|---------------|-----------------|--------------------------|
| Pole at $r = 0.999$ | Accurate (tapered precision near 1.0) | Marginal ($2^{-14}$ quantum) | Accurate (10-bit mantissa) |
| Accumulator overflow | Saturates gracefully | Wraps silently | Saturates to $\infty$ |
| Limit cycles | Reduced (fine resolution near 0) | Common | Possible |
| High dynamic range | 120 dB+ (wide exponent) | 84 dB (fixed) | 80 dB (5-bit exponent) |
| Coefficient of $10^{-4}$ | Well-represented | Uses 1 of 14 frac bits | Well-represented |
| Uniform quantization | No (tapered) | Yes | No (floating-point) |

**Posits** excel when values cluster near $\pm 1$ -- exactly the regime
of biquad coefficients and normalized signals. **Fixed-point** excels when
uniform quantization matches the signal statistics and the dynamic range
is known a priori. **Cfloat/half** provides a familiar IEEE-like model with
moderate precision and wide availability in hardware.

## The practical workflow

1. **Design** the filter with `double` coefficients.
2. **Run** `stability_margin()` and `coefficient_sensitivity()` on the
   reference cascade.
3. **Project** onto the target coefficient type with `project_onto<T>()`.
4. **Re-run** the analysis on the projected cascade.
5. **Compare**: if the stability margin dropped by more than 50%, or the
   condition number exceeds 1000, choose a wider coefficient type.
6. **Sweep** the state type separately: process a test signal and measure
   SQNR at each candidate width.

This data-driven approach replaces guesswork with measurement, letting you
find the narrowest types that still meet your quality budget.
