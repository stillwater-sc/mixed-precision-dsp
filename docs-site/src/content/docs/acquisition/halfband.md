---
title: Half-Band FIR Filter
description: Optimized half-band FIR filter for efficient 2x decimation in high-rate acquisition pipelines
---

## History and Motivation

Half-band filters are a special case of FIR lowpass filters where the cutoff
frequency is exactly one quarter of the sampling rate ($f_s/4$). This
constraint yields a remarkable structural property: nearly half the filter
coefficients are exactly zero, cutting the computational cost roughly in half.

When combined with 2x decimation — where only every other output sample is
computed — the total savings approaches **4x** compared to a general FIR
filter followed by downsampling. This makes half-band filters the standard
building block between CIC decimation and final-stage processing in high-rate
data acquisition systems.

## Where Half-Band Fits in the Pipeline

```text
┌─────────┐    ┌─────────────┐    ┌───────────┐    ┌──────────────┐    ┌────────┐
│ ADC     │───>│ CIC         │───>│ CIC Comp  │───>│ Half-Band    │───>│ Final  │
│ 1 GSPS  │    │ Decimator   │    │ Filter    │    │ Decimator    │    │ FIR    │
│ 12-bit  │    │ ÷64         │    │ (droop)   │    │ ÷2           │    │ ÷4     │
└─────────┘    └─────────────┘    └───────────┘    └──────────────┘    └────────┘
  SampleT          StateT            CoeffT           CoeffT           CoeffT
  (narrow)         (wide)            /StateT          /StateT          /StateT
```

The CIC handles the bulk rate reduction at full clock rate (no multipliers).
The half-band filter then provides a clean 2x decimation stage with excellent
stopband rejection at modest computational cost.

## Theory

### Half-Band Property

A half-band filter satisfies the power complementary condition:

$$
H(\omega) + H(\pi - \omega) = 1
$$

This means the frequency response is symmetric about $\omega = \pi/2$
(normalized frequency 0.25). The passband and stopband ripples are equal.

### Coefficient Structure

For a Type I FIR filter of length $N = 4K + 3$ (odd, with center at
index $C = (N-1)/2$):

- **Center tap**: $h[C] = 0.5$
- **Even offsets from center**: $h[C \pm 2k] = 0$ for $k \geq 1$
- **Odd offsets**: $h[C \pm (2k+1)]$ are the non-zero design coefficients
- **Symmetry**: $h[C - j] = h[C + j]$ (linear phase)

### Computational Savings

| Property | General FIR | Half-Band | Half-Band + Decimate |
|----------|------------|-----------|---------------------|
| Multiplies per output | $N$ | $\sim N/2$ | $\sim N/4$ |
| Adds per output | $N-1$ | $\sim N/2$ | $\sim N/4$ |

The savings come from:
1. Skipping zero-valued taps (~half are zero)
2. Exploiting symmetry (fold left + right before multiply)
3. Computing only every other output (decimation)

### Valid Filter Lengths

The filter length must be of the form $N = 4K + 3$:

| K | N | Non-zero taps | Zero taps |
|---|---|--------------|-----------|
| 0 | 3 | 3 | 0 |
| 1 | 7 | 5 | 2 |
| 2 | 11 | 7 | 4 |
| 3 | 15 | 9 | 6 |
| 4 | 19 | 11 | 8 |
| 5 | 23 | 13 | 10 |

## API

### Design Function

```cpp
#include <sw/dsp/acquisition/halfband.hpp>

using namespace sw::dsp;

// Design an 11-tap half-band filter with transition width 0.1 (normalized)
// Passband: [0, 0.20], Stopband: [0.30, 0.50]
auto taps = design_halfband<double>(11, 0.1);

// Design a sharper 19-tap filter with narrower transition
auto sharp_taps = design_halfband<double>(19, 0.05);
```

The `transition_width` parameter controls the trade-off between filter
length and transition sharpness:
- Passband edge = $0.25 - \text{tw}/2$
- Stopband edge = $0.25 + \text{tw}/2$

### Half-Band Filter

```cpp
// Three-scalar parameterization: CoeffScalar, StateScalar, SampleScalar
HalfBandFilter<double> hb(taps);

// Full-rate filtering (one output per input)
double y = hb.process(x);

// Block processing
std::vector<double> output(input.size());
hb.process_block(std::span<const double>(input),
                 std::span<double>(output));
```

### Integrated 2x Decimation

```cpp
HalfBandFilter<double> hb(taps);

// Sample-by-sample decimation
auto [ready, y] = hb.process_decimate(x);
if (ready) {
    // Use decimated output y
}

// Block decimation (produces input.size()/2 outputs)
std::vector<double> output;
hb.process_block_decimate(std::span<const double>(input), output);
```

### Utility Methods

```cpp
hb.num_taps();           // Total tap count (including zeros)
hb.num_nonzero_taps();   // Non-zero taps (actual multiplications)
hb.order();              // Filter order (num_taps - 1)
hb.taps();               // Full tap vector (const reference)
hb.reset();              // Clear all delay-line state
```

## Example: CIC + Half-Band Cascade

```cpp
#include <sw/dsp/acquisition/cic.hpp>
#include <sw/dsp/acquisition/halfband.hpp>

using namespace sw::dsp;

// Stage 1: CIC decimation by 64 (no multiplications)
CICDecimator<double> cic(64, 4, 1);

// Stage 2: Half-band decimation by 2
auto hb_taps = design_halfband<double>(19, 0.08);
HalfBandFilter<double> hb(hb_taps);

// Process: 1 GSPS → 15.625 MSPS (CIC) → 7.8125 MSPS (half-band)
std::vector<double> adc_samples = /* ... */;
std::vector<double> cic_out;
cic.process_block(std::span<const double>(adc_samples), cic_out);

std::vector<double> hb_out;
hb.process_block_decimate(std::span<const double>(cic_out), hb_out);
```

## Precision Considerations

The half-band filter uses the three-scalar model:

- **CoeffScalar** stores the designed tap values. Since half-band taps are
  derived from Remez exchange, they are typically small fractions. Float
  precision is often sufficient for the coefficients themselves.

- **StateScalar** holds the delay-line accumulation. For high dynamic range
  signals, double or extended-precision accumulators prevent rounding
  error accumulation across the $\sim N/4$ multiply-accumulate operations.

- **SampleScalar** matches the data stream precision. In a cascade after
  CIC, this is the CIC output type (often wider than the original ADC samples).

```cpp
// Mixed precision: float coefficients, double state, float samples
auto taps_f = /* design and project to float */;
HalfBandFilter<float, double, float> hb(taps_f);
```

### Comparison with CIC

| Property | CIC | Half-Band |
|----------|-----|-----------|
| Multiplications | None | ~N/4 per output |
| Coefficients | None (unit) | Designed (Remez) |
| Passband shape | sinc droop | Equiripple (flat) |
| Typical decimation | 16x–256x | 2x |
| Precision model | Two-scalar | Three-scalar |
