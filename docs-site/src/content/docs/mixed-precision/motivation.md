---
title: Why Mixed-Precision?
description: The energy, bandwidth, and accuracy arguments for using different arithmetic precisions at different stages of a DSP pipeline
---

Every digital signal processing system performs three kinds of arithmetic:
reading samples from a sensor, multiplying them by filter coefficients, and
accumulating partial results. Conventional practice uses a single type --
usually `float` or `double` -- for all three. This is safe but wasteful.
Mixed-precision arithmetic assigns a different numeric type to each role,
trading unnecessary precision for concrete savings in energy, bandwidth,
and silicon area.

## The energy cost of arithmetic

The dominant cost in a DSP pipeline is the multiply-accumulate (MAC)
operation. For an integer or fixed-point multiplier, gate count and
energy scale **quadratically** with operand width:

$$
E_{\text{mul}} \;\propto\; n^2
$$

where $n$ is the number of bits. A 16-bit multiplier uses roughly
$\frac{16^2}{32^2} = \frac{1}{4}$ the energy of a 32-bit one. A
8-bit multiplier drops to $\frac{1}{16}$. For floating-point units the
scaling is slightly better than quadratic for the mantissa multiplier
and linear for the exponent adder, but the overall trend holds:
**narrower operands cost less**.

| Multiplier width | Relative energy | Relative area |
|------------------|-----------------|---------------|
| 8-bit            | 1x              | 1x            |
| 16-bit           | ~4x             | ~4x           |
| 32-bit           | ~16x            | ~16x          |
| 64-bit           | ~64x            | ~64x          |

These ratios are order-of-magnitude estimates from published ISSCC and
VLSI-T survey data (Weste & Harris, *CMOS VLSI Design*, 4th ed.). Actual
numbers depend on technology node and microarchitecture, but the quadratic
trend is well established.

## The memory bandwidth bottleneck

In streaming DSP, memory bandwidth is often the binding constraint.
Consider a 4th-order IIR filter running at 48 kHz:

- **Samples**: 48,000 per second. At `double` that is 384 KB/s; at
  `half` it is 96 KB/s -- a 4x reduction.
- **Coefficients**: 10 values (5 per biquad section x 2 sections),
  loaded once. Precision matters for accuracy, not bandwidth.
- **State**: 4 delay elements, updated every sample. Wider is better
  for accumulation quality, but the state never leaves the ALU.

Narrowing the sample type from 64 bits to 16 bits saves 75% of the
streaming bandwidth without touching the coefficients or state that
actually need precision.

## Different roles need different precision

The key insight behind mixed-precision DSP is that the three arithmetic
roles in a filter have **different precision requirements**:

### Coefficients (design precision)

Filter coefficients encode the locations of poles and zeros in the
$z$-plane. A Butterworth lowpass with a cutoff at 100 Hz and sample
rate 48 kHz places poles at radius $r \approx 0.987$. Representing
this coefficient in a type that rounds it to $0.99$ or $0.98$ shifts
the pole, distorting the frequency response or even causing instability.
Coefficients need enough bits to preserve the designed pole/zero geometry.

### State (processing precision)

The accumulator in a biquad difference equation computes:

$$
y[n] = b_0 x[n] + b_1 x[n{-}1] + b_2 x[n{-}2] - a_1 y[n{-}1] - a_2 y[n{-}2]
$$

Each term is a product of a coefficient and a sample. The sum can be much
larger than any individual term, and rounding errors accumulate over
millions of samples. The state type must have enough dynamic range to
avoid overflow and enough precision to maintain signal-to-noise ratio
(SNR) during long processing runs.

### Samples (streaming precision)

Input samples come from an ADC with a fixed resolution -- typically
8 to 24 bits depending on the application. Output samples feed a DAC
or a downstream processor with its own resolution limit. There is no
benefit to streaming samples in 64-bit precision if the sensor only
delivers 12 valid bits.

## Real-world examples

| Domain | Sample source | Useful sample bits | Coefficient precision | Accumulator need |
|--------|---------------|--------------------|-----------------------|------------------|
| Audio (CD) | 16-bit PCM | 16 | High (narrow transition bands) | 32+ (long reverb tails) |
| Radar | 12-bit ADC | 10--12 | Very high (sharp nulls) | 40+ (pulse integration) |
| Data acquisition | 24-bit ADC | 18--20 (ENOB) | Moderate | 32--48 |
| Image processing | 8-bit sensor | 5--6 (ENOB) | Moderate | 16--24 |
| Telecommunications | 8-bit I/Q | 6--8 | High (channel equalization) | 24--32 |
| Vibration monitoring | 16-bit MEMS | 12--14 | Moderate | 32 |

In every case, the sample precision is bounded by the sensor, the
coefficient precision is set by the algorithm's sensitivity, and the
accumulator precision is driven by the dynamic range of intermediate
results. Using a single type for all three either wastes energy on
the samples or starves the accumulator.

## What the library provides

`sw::dsp` makes mixed-precision experimentation trivial. Every filter,
transform, and analysis function is parameterized on three independent
scalar types:

```cpp
#include <sw/dsp/dsp.hpp>
#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

using namespace sw::dsp;
using half = sw::universal::half;
using posit32 = sw::universal::posit<32, 2>;

// High-precision coefficients, posit accumulator, half-precision samples
SimpleFilter<ButterworthLowPass<4>, double, posit32, half> lp;
lp.setup(4, 48000.0, 1000.0);

half x{0.5};
half y = lp.process(x);
```

Changing a type is a one-line edit. The library's analysis module then
lets you measure the impact: `stability_margin()` checks pole proximity
to the unit circle, `coefficient_sensitivity()` quantifies how much
pole positions shift under quantization, and `cascade_condition_number()`
estimates frequency response fragility. This turns precision selection
from guesswork into engineering.

## The bottom line

Mixed-precision is not about using less precision everywhere. It is about
using the **right** precision at each stage -- no more, no less. The result
is a system that meets the same quality target with a fraction of the
energy, bandwidth, and silicon area. The pages that follow explain the
three-scalar model, the numerical pitfalls that precision choices expose,
and the sensor-noise argument that bounds the minimum useful precision
from below.
