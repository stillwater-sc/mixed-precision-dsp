---
title: CIC Filters
description: Cascaded Integrator-Comb decimation and interpolation filters for high-rate data acquisition
---

## History and Motivation

The Cascaded Integrator-Comb (CIC) filter was introduced by Eugene Hogenauer
in 1981 to address a fundamental problem in digital receiver design: how to
decimate a high-speed ADC stream by a large factor without multiplications.

At GHz sampling rates, even a single multiplier per clock cycle is expensive
in silicon area and power. Hogenauer showed that a cascade of integrators
(accumulators) followed by a cascade of combs (differentiators) — separated
by a rate change — implements a decimation or interpolation filter using
**only additions and subtractions**. This made CIC filters the universal
first stage in digital down-converters, software-defined radios, and
high-speed data acquisition systems.

## Where CIC Fits in the Pipeline

A typical high-rate acquisition chain processes an ADC output through several
stages of decimation before the signal reaches the application:

```text
┌─────────┐    ┌─────────────┐    ┌───────────┐    ┌──────────────┐    ┌────────┐
│ ADC     │───>│ CIC         │───>│ CIC Comp  │───>│ Half-Band    │───>│ Final  │
│ 1 GSPS  │    │ Decimator   │    │ Filter    │    │ Decimator    │    │ FIR    │
│ 12-bit  │    │ ÷64         │    │ (droop)   │    │ ÷2           │    │ ÷4     │
└─────────┘    └─────────────┘    └───────────┘    └──────────────┘    └────────┘
  SampleT          StateT            CoeffT            CoeffT           CoeffT
  (narrow)         (wide)            /StateT           /StateT          /StateT
```

The CIC decimator handles the bulk of the rate reduction (often 16× to 256×)
at the full ADC clock rate. Subsequent stages operate at the reduced rate
where multipliers are affordable.

## Theory

### Structure

A CIC decimation filter of order $M$ with decimation ratio $R$ and
differential delay $D$ consists of:

1. **$M$ integrator stages** running at the input (high) rate:

$$
H_I(z) = \frac{1}{1 - z^{-1}}
$$

2. **Downsampling by $R$**: keep every $R$-th sample

3. **$M$ comb stages** running at the output (decimated) rate:

$$
H_C(z) = 1 - z^{-D}
$$

The combined transfer function is:

$$
H(z) = \left(\frac{1 - z^{-RD}}{1 - z^{-1}}\right)^M
$$

### DC Gain

At DC ($z = 1$), the transfer function evaluates to:

$$
H(1) = (RD)^M
$$

For example, with $R=8$, $M=3$, $D=1$: DC gain $= 8^3 = 512$.

### Bit Growth

The CIC filter's DC gain determines the required accumulator width.
To prevent overflow, the state registers must accommodate:

$$
B_{\text{out}} = B_{\text{in}} + M \cdot \lceil\log_2(RD)\rceil
$$

additional bits beyond the input sample width. This formula directly
drives the `StateScalar` selection in the three-scalar model.

| Configuration | Bit Growth | DC Gain | Min StateScalar |
|--------------|-----------|---------|----------------|
| $R=4, M=1, D=1$ | 2 bits | 4 | 14-bit for 12-bit ADC |
| $R=8, M=3, D=1$ | 9 bits | 512 | 21-bit for 12-bit ADC |
| $R=16, M=4, D=1$ | 16 bits | 65536 | 28-bit for 12-bit ADC |
| $R=64, M=5, D=1$ | 30 bits | ~10⁹ | 42-bit for 12-bit ADC |

### Frequency Response

The magnitude response of a CIC filter is:

$$
|H(f)| = \left|\frac{\sin(\pi R D f)}{\sin(\pi f)}\right|^M
$$

This is a **sinc-like lowpass** with:
- Main lobe width proportional to $1/(RD)$
- Side lobes that decrease with $M$
- **Passband droop** that increases toward the band edge

The passband droop is the CIC's main limitation — it must be compensated
by a subsequent FIR filter (the "CIC compensation filter") whose response
is the inverse of the CIC's passband rolloff.

### Interpolation

A CIC interpolator reverses the structure:

1. **$M$ comb stages** at the low (input) rate
2. **Upsampling by $R$**: insert $R-1$ zeros between samples
3. **$M$ integrator stages** at the high (output) rate

The transfer function is identical. For each low-rate input sample,
$R$ high-rate output samples are produced.

## API

### CIC Decimator

```cpp
#include <sw/dsp/acquisition/cic.hpp>

using namespace sw::dsp;

// CICDecimator<StateScalar, SampleScalar>
// R=8 decimation, M=3 stages, D=1 differential delay
CICDecimator<double> cic(8, 3, 1);

// Process one sample at a time
bool ready = cic.push(sample);
if (ready) {
    double output = cic.output();
}

// Or process a block
std::vector<double> input = /* ... */;
std::vector<double> output;
cic.process_block(std::span<const double>(input), output);
```

### CIC Interpolator

```cpp
// CICInterpolator<StateScalar, SampleScalar>
CICInterpolator<double> interp(8, 3, 1);

// Feed one low-rate sample, get R high-rate samples
interp.push(sample);
for (int r = 0; r < 8; ++r) {
    double y = interp.output();
}

// Or process a block (produces input.size() * R outputs)
std::vector<double> input = /* ... */;
std::vector<double> output;
interp.process_block(std::span<const double>(input), output);
```

### Utility Functions

```cpp
// Compute required bit growth
int growth = cic_bit_growth(3, 8, 1);  // M=3, R=8, D=1 → 9

// Query filter properties
cic.decimation_ratio();    // R
cic.num_stages();          // M
cic.differential_delay();  // D
cic.dc_gain();             // (R*D)^M
cic.bit_growth();          // M * ceil(log2(R*D))
cic.reset();               // Clear all state
```

## Example: 12-bit ADC Decimation

```cpp
#include <sw/dsp/acquisition/cic.hpp>
#include <sw/universal/number/fixpnt/fixpnt.hpp>
#include <sw/universal/number/posit/posit.hpp>

using namespace sw::dsp;
using namespace sw::universal;

// 12-bit ADC → CIC decimate by 64, 4 stages
// Bit growth = 4 * ceil(log2(64)) = 24
// Need 12 + 24 = 36-bit accumulator minimum

// Fixed-point: 40-bit state, 12-bit samples (classic approach)
using state_fx = fixpnt<40, 36>;
using sample_fx = fixpnt<12, 11>;
CICDecimator<state_fx, sample_fx> cic_fixed(64, 4, 1);

// Posit: 32-bit state may suffice due to tapered precision
using state_p = posit<32, 2>;
using sample_p = posit<16, 2>;
CICDecimator<state_p, sample_p> cic_posit(64, 4, 1);

// Double: reference (no overflow concerns)
CICDecimator<double> cic_ref(64, 4, 1);
```

## Precision Considerations

The CIC filter is a natural showcase for the three-scalar model:

- **SampleScalar** matches the ADC bit depth (8-16 bits). Narrow types
  minimize memory bandwidth at the full sampling rate.

- **StateScalar** must accommodate the full bit growth. For a 5-stage
  CIC with $R=64$, this is 30 extra bits — a 12-bit ADC needs a 42-bit
  accumulator. Fixed-point types must be sized precisely; posit types
  may tolerate fewer bits due to tapered precision near zero.

- The CIC requires **no coefficients** (no `CoeffScalar`), since all
  operations are unit additions and subtractions. This is unique among
  DSP filter structures.

### Overflow Behavior

When `StateScalar` is too narrow, the integrators overflow. Unlike FIR
filters where coefficient errors cause gradual degradation, CIC overflow
is **catastrophic** — the output becomes meaningless. This makes the
bit-growth formula critical for correct design.

With wrapping arithmetic (as in hardware fixed-point), CIC filters can
exploit modular arithmetic: if the final output fits in the output word
width, intermediate overflow in the integrators is harmless because the
comb stages recover the correct result via modular subtraction. This
property holds for two's-complement integers and `fixpnt` types but
**not** for floating-point, posit, or other non-wrapping types.
