---
title: The Three-Scalar Model
description: How sw::dsp parameterizes every algorithm on three independent scalar types for coefficients, state, and samples
---

The central abstraction in `sw::dsp` is the **three-scalar model**: every
filter and processing algorithm is parameterized on three independent
arithmetic types that correspond to three distinct roles in the computation.

## The three roles

### CoeffScalar -- design precision

Filter coefficients encode the mathematical design: pole and zero locations,
gain constants, window weights. A 4th-order Butterworth lowpass at
$f_c = 100\,\text{Hz}$, $f_s = 48\,\text{kHz}$ has a dominant pole at
radius:

$$
r = e^{-\pi f_c / f_s} \approx 0.9935
$$

If the coefficient type cannot distinguish $0.9935$ from $0.9940$, the
pole moves, the -3 dB frequency shifts, and the filter no longer meets
its specification. Coefficient precision sets the **design fidelity**.

Typical choices: `double`, `posit<32,2>`, `cfloat<32,8,…>`.

### StateScalar -- processing precision

The biquad difference equation accumulates products of coefficients and
samples:

$$
y[n] = \sum_{k=0}^{2} b_k\, x[n{-}k] \;-\; \sum_{k=1}^{2} a_k\, y[n{-}k]
$$

Each multiply-add introduces rounding error. Over $N$ samples, the noise
power in the output grows as $O(N)$ for uncorrelated errors. A narrow
state type that rounds aggressively after each accumulation degrades the
signal-to-quantization-noise ratio (SQNR) and can trigger limit cycles --
persistent low-level oscillations even with zero input.

The state type must also have enough **dynamic range** to hold the
intermediate sums without overflow. For an FIR filter with $L$ taps and
unit-amplitude input, the worst-case accumulator value is $L$ times the
largest coefficient.

Typical choices: `double`, `posit<64,3>`, `posit<32,2>`.

### SampleScalar -- streaming precision

Input samples come from a sensor with a fixed effective number of bits
(ENOB). Output samples go to a DAC, a display, or a downstream stage with
its own resolution. There is no point streaming 64-bit samples through a
memory bus if the sensor delivers 12 valid bits.

Narrowing the sample type is where the largest bandwidth and energy savings
come from, because samples are the high-volume data in any pipeline.

Typical choices: `float`, `half`, `posit<16,1>`, `cfloat<16,5,…>`.

## The template pattern

Every filter design in the library follows this pattern:

```cpp
#include <sw/dsp/dsp.hpp>
using namespace sw::dsp;

// Template parameters: Design<Order>, CoeffScalar, StateScalar, SampleScalar
SimpleFilter<ButterworthLowPass<4>, double, double, float> lp;
lp.setup(4, 48000.0, 1000.0);   // order, sample rate, cutoff

float y = lp.process(0.5f);     // filter one sample
```

The three scalar types are independent. You can mix native IEEE types with
Universal number types freely:

```cpp
#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

using half   = sw::universal::half;
using p32    = sw::universal::posit<32, 2>;

// double coefficients, posit<32,2> state, half samples
SimpleFilter<ButterworthLowPass<4>, double, p32, half> lp;
lp.setup(4, 48000.0, 1000.0);

half x{0.5};
half y = lp.process(x);
```

The same parameterization applies to all filter families: Chebyshev,
Elliptic, Bessel, and FIR designs.

## A concrete example: precision vs. energy

Consider a 4th-order Butterworth lowpass at 1 kHz, sampled at 48 kHz.
We compare three configurations against an all-`double` reference:

| Configuration | CoeffScalar | StateScalar | SampleScalar | Passband SQNR | Relative MAC energy |
|---------------|-------------|-------------|--------------|---------------|---------------------|
| Reference     | `double`    | `double`    | `double`     | $\infty$      | 1.0x                |
| Mixed-A       | `double`    | `posit<32,2>` | `half`     | 92 dB         | ~0.25x              |
| Mixed-B       | `float`     | `float`     | `half`       | 85 dB         | ~0.16x              |
| All-narrow    | `half`      | `half`      | `half`       | 31 dB         | ~0.06x              |

Mixed-A preserves 92 dB SQNR -- well above CD-quality (96 dB dynamic
range) -- while reducing MAC energy by roughly 4x. The `posit<32,2>`
state type provides 30 bits of precision near unity, which is where
biquad intermediate values concentrate. The `half` sample type matches
a 10-bit ENOB sensor and cuts streaming bandwidth in half compared to
`float`.

Mixed-B trades some coefficient fidelity for additional savings. The
quality is still adequate for speech or sensor-fusion pipelines. The
all-narrow configuration shows where the model breaks down: `half`
coefficients cannot represent the pole at $r = 0.9935$ with sufficient
accuracy, and the state accumulates rounding errors that degrade SQNR
by over 60 dB.

## Type projection and embedding

When moving data between precision domains, the library provides two
operations that make the direction of conversion explicit:

### `project_onto<Target>(source)` -- lossy narrowing

Projects a value (or an entire biquad cascade) from a wider type onto
a narrower one. This is inherently lossy -- precision is discarded.
The name comes from the linear-algebra analogy of projecting a vector
from $\mathbb{R}^n$ onto a lower-dimensional subspace.

```cpp
#include <sw/dsp/types/projection.hpp>
using namespace sw::dsp;
using fp16 = sw::universal::half;

// Design in double, then project onto half for deployment
SimpleFilter<ButterworthLowPass<4>, double, double, double> design;
design.setup(4, 48000.0, 1000.0);

auto narrow_cascade = project_onto<fp16>(design.cascade());
```

The compiler enforces the direction: `project_onto<T>` requires that
`Target` has fewer (or equal) significant digits than `Source`, checked
at compile time via the `ProjectableOnto` concept:

```cpp
template <typename Target, typename Source>
concept ProjectableOnto =
    std::numeric_limits<Target>::digits <= std::numeric_limits<Source>::digits;
```

### `embed_into<Target>(source)` -- lossless widening

Embeds a value from a narrower type into a wider one. This is lossless --
the original value is exactly representable in the target. The analogy is
embedding a low-dimensional object into a higher-dimensional space.

```cpp
// Bring narrow coefficients back to double for analysis
auto wide_cascade = embed_into<double>(narrow_cascade);
```

The `EmbeddableInto` concept enforces the direction at compile time.

### The design-deploy-verify workflow

These two operations enable a disciplined workflow:

1. **Design** coefficients in `double` (maximum accuracy).
2. **Project** onto the deployment type (`fixpnt<16,14>`, `posit<16,1>`, etc.).
3. **Analyze** the projected cascade with `stability_margin()` and
   `coefficient_sensitivity()` to verify that the quality loss is acceptable.
4. **Embed** back into `double` if you need to compare frequency responses
   or compute pole displacement.

```cpp
// Full workflow
auto ref_cascade = design.cascade();                  // double
auto deployed    = project_onto<fixpnt16>(ref_cascade); // narrow
double margin    = stability_margin(deployed);          // check

if (margin < 0.01) {
    // Poles too close to the unit circle -- use a wider type
}

auto recovered = embed_into<double>(deployed);        // compare
```

## When to use which type

| Role | Priority | Recommended types |
|------|----------|-------------------|
| CoeffScalar | Accuracy near 1.0, many significant digits | `double`, `posit<32,2>` |
| StateScalar | Dynamic range + precision for accumulation | `double`, `posit<32,2>`, `posit<64,3>` |
| SampleScalar | Bandwidth efficiency, matches sensor ENOB | `half`, `posit<16,1>`, `cfloat<16,5,…>`, `float` |

The three-scalar model does not dictate specific types. It provides the
**mechanism** to experiment and the analysis tools to **measure** the
consequences. The next page covers the numerical pitfalls that arise
when these choices go wrong.
