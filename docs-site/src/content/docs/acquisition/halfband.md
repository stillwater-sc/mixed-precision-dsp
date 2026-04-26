---
title: Half-Band FIR Filter
description: Optimized half-band FIR filter for efficient 2x decimation in high-rate acquisition pipelines
---

*Part of the [Multirate Signal Processing](../multirate/overview/) section — see the overview for the broader theory and the [pattern catalog](../multirate/patterns/) for problem→API mapping.*

## History and Motivation

### The Multirate Revolution

The story of half-band filters begins with the multirate signal processing
revolution of the 1970s and 1980s. Before multirate techniques, decimation
was done naively: filter the signal with a lowpass FIR to prevent aliasing,
then throw away samples. Engineers computed every output of the FIR filter,
even the samples that would immediately be discarded by the downsampler. For
a 2x decimation, half the computation was wasted.

Ronald Crochiere and Lawrence Rabiner at Bell Labs recognized this
inefficiency and formalized the **Noble identity** (1975–1983): filtering
and decimation can be interleaved so that only the surviving output samples
are ever computed. Their work on multirate filter banks, published in
*Multirate Digital Signal Processing* (1983), established the theoretical
framework that made efficient decimation chains practical.

### Why Half-Band?

Among all possible decimation-by-2 filters, half-band filters occupy a
uniquely efficient niche. The insight dates to the early work on quadrature
mirror filters (QMF) by Esteban and Galand (1977) and was further developed
by Vaidyanathan, Mintzer, and others through the 1980s.

A half-band filter is a lowpass FIR whose cutoff frequency is exactly
$f_s/4$ — one quarter of the sampling rate. This specific cutoff creates a
remarkable structural property: the frequency response satisfies the **power
complementary condition**:

$$
H(\omega) + H(\pi - \omega) = 1
$$

This symmetry about $\omega = \pi/2$ forces **nearly half the filter
coefficients to be exactly zero**. Combined with the linear-phase symmetry
of a Type I FIR, this means only about one quarter of the coefficients
are unique non-zero values. When you further combine this with polyphase
decimation (computing only the samples that survive downsampling), the total
savings reaches approximately **4x** compared to a general FIR
filter-then-decimate approach.

### From Telecom to Data Acquisition

Half-band filters became a standard building block in the 1990s as digital
receiver architectures matured. In the classic digital down-converter (DDC)
architecture — used in everything from cellular base stations to spectrum
analyzers — the signal processing chain after the ADC consists of:

1. **CIC filter** — bulk decimation at full clock rate, no multipliers
2. **CIC compensation filter** — corrects the CIC's passband droop
3. **Half-band filter(s)** — one or more stages of 2x decimation
4. **Programmable FIR** — final channel-select filter

The half-band stages sit in the sweet spot: the rate is low enough that
multipliers are affordable, but still high enough that the 4x computational
savings matters. Many DDC ASICs (e.g., the Graychip GC4016, Analog Devices
AD6620, Harris HSP50016) hardwired one or more half-band stages in their
decimation chains precisely because the fixed structure — zero taps known
at design time, symmetric — maps efficiently to silicon.

### The Remez Connection

Designing optimal half-band filters requires the Parks-McClellan (Remez
exchange) algorithm. James McClellan and Thomas Parks developed their
equiripple FIR design algorithm at Rice University in 1972, and it remains
the standard method for designing FIR filters with specified passband and
stopband characteristics.

For half-band design, the Remez algorithm is constrained to produce a
filter symmetric about $f_s/4$, with equal passband and stopband ripple.
The transition bandwidth is the primary design parameter: narrower
transitions require more taps but provide sharper roll-off and better
alias rejection.

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

In systems that need higher total decimation, multiple half-band stages can
be cascaded: each stage halves the rate, and because each subsequent stage
operates at half the rate of the previous one, the total cost converges
to roughly twice the cost of the first stage alone (a geometric series
argument). This is why DDC architectures often use 2–4 cascaded half-band
stages rather than a single higher-order decimation filter.

## Theory

### Half-Band Property

A half-band filter satisfies the power complementary condition:

$$
H(\omega) + H(\pi - \omega) = 1
$$

This means the frequency response is symmetric about $\omega = \pi/2$
(normalized frequency 0.25). The passband and stopband ripples are equal:
$\delta_p = \delta_s$. This is not a limitation for most applications —
the equal-ripple constraint is what enables the zero-tap structure that
makes half-band filters efficient.

### Coefficient Structure

For a Type I FIR filter of length $N = 4K + 3$ (odd, with center at
index $C = (N-1)/2$):

- **Center tap**: $h[C] = 0.5$
- **Even offsets from center**: $h[C \pm 2k] = 0$ for $k \geq 1$
- **Odd offsets**: $h[C \pm (2k+1)]$ are the non-zero design coefficients
- **Symmetry**: $h[C - j] = h[C + j]$ (linear phase)

The requirement $N = 4K + 3$ (i.e., $N \bmod 4 = 3$) ensures the correct
alignment of zero and non-zero taps. Choosing a length that does not
satisfy this form would break the half-band structure.

**Example: 11-tap half-band filter ($K = 2$)**

```text
Index:  0       1       2       3       4       5       6       7       8       9       10
Tap:    h[0]    0       h[2]    0       h[4]    0.5     h[4]    0       h[2]    0       h[0]
                                                 ↑ center
```

Only $h[0]$, $h[2]$, $h[4]$, and $0.5$ are non-zero — 7 non-zero values
out of 11 total, with only 3 unique values (plus the center) due to symmetry.

### Computational Savings

| Property | General FIR | Half-Band | Half-Band + Decimate |
|----------|------------|-----------|---------------------|
| Multiplies per output | $N$ | $\sim N/2$ | $\sim N/4$ |
| Adds per output | $N-1$ | $\sim N/2$ | $\sim N/4$ |

The savings come from three compounding optimizations:

1. **Skip zero-valued taps** — roughly half the taps are structurally zero
2. **Exploit symmetry** — fold the left and right delay-line samples
   before multiplying (one multiply instead of two)
3. **Compute only surviving outputs** — with 2x decimation, only every
   other output is needed

In hardware, the fixed zero-tap positions are known at design time, so
the multiplier resources simply aren't instantiated. In software, the
implementation uses a compact list of non-zero coefficient indices,
avoiding both the multiply and the memory fetch for zero taps.

### Valid Filter Lengths

The filter length must be of the form $N = 4K + 3$:

| K | N | Non-zero taps | Zero taps | Unique coefficients |
|---|---|--------------|-----------|---------------------|
| 0 | 3 | 3 | 0 | 1 + center |
| 1 | 7 | 5 | 2 | 2 + center |
| 2 | 11 | 7 | 4 | 3 + center |
| 3 | 15 | 9 | 6 | 4 + center |
| 4 | 19 | 11 | 8 | 5 + center |
| 5 | 23 | 13 | 10 | 6 + center |

### Transition Width and Stopband Rejection

The transition bandwidth parameter controls the trade-off between filter
length and selectivity:

- Passband edge = $0.25 - \text{tw}/2$
- Stopband edge = $0.25 + \text{tw}/2$

Typical values:

| Transition Width | Passband | Stopband | Typical N for 60 dB |
|-----------------|----------|----------|---------------------|
| 0.20 | [0, 0.15] | [0.35, 0.50] | 7 |
| 0.10 | [0, 0.20] | [0.30, 0.50] | 11–15 |
| 0.05 | [0, 0.225] | [0.275, 0.50] | 19–27 |
| 0.02 | [0, 0.24] | [0.26, 0.50] | 43–59 |

Narrower transitions require more taps, but the half-band structure ensures
that roughly half those additional taps are zero, so the computational
cost grows at half the rate of a general FIR.

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

The design function uses the Remez exchange algorithm internally, then
enforces exact half-band constraints: the center tap is set to exactly
0.5, even-offset taps are zeroed, symmetry is enforced, and the non-zero
taps are normalized so the filter has unity DC gain.

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

// Block decimation — returns dense_vector of decimated outputs
auto decimated = hb.process_block_decimate(
    std::span<const double>(input));
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

auto hb_out = hb.process_block_decimate(
    std::span<const double>(cic_out));
```

## Precision Considerations

The half-band filter uses the **three-scalar model**, and each scalar
plays a distinct role in maintaining signal fidelity:

### CoeffScalar — Design Precision

Stores the designed tap values. Since half-band taps are derived from
Remez exchange, they are typically small fractions (the largest non-zero
tap is typically around 0.3 for moderate-length filters). Float precision
is often sufficient for the coefficients themselves.

However, the design *process* should use higher precision. Our
`design_halfband<T>()` function performs all design-time arithmetic in
the template type `T`, so designing in `double` and then projecting to
`float` coefficients preserves the Remez optimality.

### StateScalar — Processing Precision

Holds the delay-line values and multiply-accumulate results. For high
dynamic range signals, double or extended-precision accumulators prevent
rounding error accumulation across the $\sim N/4$ multiply-accumulate
operations per output sample.

The denormal prevention system (`DenormalPrevention<StateScalar>`) is
active in the accumulator loop. For IEEE 754 types, a small alternating
signal keeps the accumulator above the denormal threshold. For posit and
fixed-point types, this is a compile-time no-op.

### SampleScalar — Streaming Precision

Matches the data stream precision. In a cascade after CIC, this is the
CIC output type — often wider than the original ADC samples due to the
CIC's bit growth. The half-band filter preserves this width through its
processing and delivers outputs at the same precision.

```cpp
// Mixed precision: float coefficients, double state, float samples
auto taps_d = design_halfband<double>(19, 0.08);

// Project double taps to float for coefficient storage
mtl::vec::dense_vector<float> taps_f(taps_d.size());
for (std::size_t i = 0; i < taps_d.size(); ++i)
    taps_f[i] = static_cast<float>(taps_d[i]);

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
| Why it's fast | Only add/subtract | Zero taps + symmetry |

## Historical References

- R. E. Crochiere and L. R. Rabiner, *Multirate Digital Signal Processing*,
  Prentice-Hall, 1983 — foundational multirate theory including Noble
  identities and polyphase decomposition
- J. H. McClellan, T. W. Parks, and L. R. Rabiner, "A Computer Program for
  Designing Optimum FIR Linear Phase Digital Filters," *IEEE Trans. Audio
  Electroacoustics*, vol. AU-21, 1973 — the Remez/Parks-McClellan algorithm
- P. P. Vaidyanathan, *Multirate Systems and Filter Banks*, Prentice-Hall,
  1993 — comprehensive treatment of half-band filters, QMF banks, and
  polyphase structures
- D. Esteban and C. Galand, "Application of Quadrature Mirror Filters to
  Split Band Voice Coding Schemes," *Proc. ICASSP*, 1977 — early QMF work
  that motivated half-band filter theory
- F. Mintzer, "On Half-Band, Third-Band, and Nth-Band FIR Filters and Their
  Design," *IEEE Trans. ASSP*, vol. 30, no. 5, 1982 — systematic treatment
  of the zero-tap structure in Mth-band filters
