---
title: Multi-Stage Decimation Chain
description: Cascading CIC, half-band, and polyphase filters to achieve large decimation ratios efficiently
---

*Part of the [Multirate Signal Processing](../multirate/overview/) section — see the overview for the broader theory and the [pattern catalog](../multirate/patterns/) for problem→API mapping.*

## History and Motivation

### Why Cascaded Filters?

A digital receiver sampling at 100 MHz that wants a 100 kHz channel
needs to decimate by 1000. Doing this in a single FIR filter is
impractical: the filter must reject everything above 50 kHz at the input
rate, which requires a very long impulse response (filter length grows
with the ratio of sample rate to transition bandwidth). For a 40 dB
stopband and 10 kHz transition width, that single filter needs
thousands of taps and a full multiply-accumulate per input sample.

The solution, developed across the 1970s and 1980s as multirate signal
processing matured, is to factor the large decimation into a cascade of
smaller, efficient stages. Each stage does part of the rate reduction
with a filter tuned to that stage's sample rate and bandwidth
requirements. The combined arithmetic cost is orders of magnitude below
the single-filter approach.

### Hogenauer's CIC and the First-Stage Problem

Eugene Hogenauer's 1981 paper *"An Economical Class of Digital Filters
for Decimation and Interpolation"* (IEEE Transactions on ASSP) solved
the first-stage problem: at GHz input rates, any filter needing
multiplications was too expensive. His Cascaded Integrator-Comb (CIC)
structure uses only additions and subtractions and runs efficiently at
the highest rate. The cost is a non-flat passband response (a
sinc-shaped droop) that later stages have to compensate.

### Harris and the Multirate Cascade

Fred Harris's *Multirate Signal Processing for Communication Systems*
(2004) formalized what the DDC ASIC designers had been doing for a
decade: split a large rate change `R = R1 × R2 × ... × Rn` across a
cascade of stages where each `Ri` is small (often 2) and each stage's
filter targets the narrower bandwidth at its input rate.

The canonical chain — replicated in the HSP50016, AD6620, GC4016 DDC
ICs — looks like:

```text
ADC (100 MHz)  →  CIC ↓64  →  HB ↓2  →  HB ↓2  →  FIR ↓2  →  baseband (~195 kHz)
```

- **CIC** handles bulk rate reduction efficiently at 100 MHz.
- **Half-bands** provide sharp stopband attenuation near 2x with about 75%
  zero taps (free decimation).
- **FIR** does final shaping at the lowest rate, where multiplications
  are cheapest.

### CIC Compensation

The CIC's passband droop — amplitude drops across the passband following
$\left|\dfrac{\sin(\pi f D)}{R D \sin(\pi f / R)}\right|^M$ (with $f$
normalized to the CIC output rate) — is predictable and
correctable. A short FIR filter with a rising magnitude response
(inverse-sinc$^M$) run at the CIC output rate flattens the combined
response. This is the origin of the "CIC compensator" or "droop
compensator" stage found in every DDC chain.

### Why Mixed-Precision Matters

Each stage in the chain has its own precision budget:
- The CIC accumulator must be wide enough for $M \lceil \log_2(RD)
  \rceil$ bits of growth — often 32 to 48 bits on silicon.
- The half-band state can be narrower because the signal bandwidth has
  been reduced and the filter has ~75% zero taps.
- The final FIR can use the narrowest state because it runs at the
  lowest rate and has the widest quantization headroom.

Allowing each stage to pick its own scalar type — IEEE float for one,
posit for another, fixpnt for a third — is a core design requirement
that this library supports via independent template parameters on
each stage.

## How to Use

### The `DecimationChain` Class

```cpp
#include <sw/dsp/acquisition/decimation_chain.hpp>
#include <sw/dsp/acquisition/cic.hpp>
#include <sw/dsp/acquisition/halfband.hpp>
#include <sw/dsp/filter/fir/polyphase.hpp>
#include <sw/dsp/filter/fir/fir_design.hpp>
#include <sw/dsp/windows/hamming.hpp>

using namespace sw::dsp;

double fs = 1'000'000.0;   // 1 MHz input

// Stage 1: CIC with R=64, M=3 (64x bulk reduction)
CICDecimator<double> cic(64, 3);

// Stage 2: half-band 2:1
auto hb_taps = design_halfband<double>(31, 0.1);
HalfBandFilter<double> hb(hb_taps);

// Stage 3: polyphase FIR 4:1
auto fir_window = hamming_window<double>(33);
auto fir_taps   = design_fir_lowpass<double>(33, 0.1, fir_window);
PolyphaseDecimator<double> pf(fir_taps, 4);

DecimationChain<double,
                CICDecimator<double>,
                HalfBandFilter<double>,
                PolyphaseDecimator<double>> chain(fs, cic, hb, pf);

// Total: 64 × 2 × 4 = 512x decimation, output at 1'953.125 Hz
std::cout << "total decimation: " << chain.total_decimation() << "\n";
std::cout << "output rate:      " << chain.output_rate()      << "\n";
```

### Streaming

```cpp
for (std::size_t n = 0; n < input_size; ++n) {
    auto [ready, y] = chain.process(input[n]);
    if (ready) sink(y);
}
```

### Block Processing

```cpp
auto output = chain.process_block(input);
// output.size() ≈ input.size() / chain.total_decimation()
```

### Per-Stage Rate Queries

```cpp
auto ratios = chain.stage_ratios();     // [64, 2, 4]
auto rates  = chain.stage_rates();      // [15625, 7812.5, 1953.125]
```

### Accessing Individual Stages

```cpp
auto& my_cic = chain.stage<0>();
std::cout << my_cic.dc_gain() << "\n";  // (64)^3 = 262144
```

## CIC Droop Compensation

The CIC's passband is not flat. For a CIC with ratio `R` and `M`
stages, the output-rate-normalized response drops by up to
$\sim M \times 3.9\,\text{dB}$ at the output Nyquist. A short FIR run
at the CIC output rate flattens this:

```cpp
// Design a 31-tap compensator for a CIC(R=16, M=3) with passband 0.2 * fs_out
auto comp_taps = design_cic_compensator<double>(31, 3, 16, 0.2);

// Run it after the CIC as an additional FIR stage
PolyphaseDecimator<double> compensator(comp_taps, 1);  // no further decimation
```

The compensator is designed via the frequency-sampling method: the
desired magnitude response is sampled at N uniform frequency points
(set to $1/|H_{\text{cic}}(f)|$ in the passband and rolled off toward
Nyquist), transformed to an impulse response via IDFT, and windowed
with a Hamming window. The result is normalized to unit DC gain.

### Flatness Improvement

For a typical configuration (R=16, M=3, passband=0.2):
- Uncompensated CIC passband ripple: ~1.7 dB
- Compensated passband ripple: ~0.06 dB (27x improvement)

## Mixed-Precision Use

Each stage's template parameters are independent. A typical
mixed-precision deployment might use `int32_t`-backed fixed-point in the
CIC (where bit growth matters most), `float` in the half-bands, and
`posit<32,2>` in the final FIR (where tapered precision near unit
magnitude pays off):

```cpp
CICDecimator<cfloat<48, 8>, int32_t>     cic(64, 3);
HalfBandFilter<float>                     hb(hb_taps);
PolyphaseDecimator<posit<32,2>>           pf(fir_taps, 4);

DecimationChain<float,
                CICDecimator<cfloat<48, 8>, int32_t>,
                HalfBandFilter<float>,
                PolyphaseDecimator<posit<32,2>>> chain(fs, cic, hb, pf);
```

> The `Sample` template parameter fixes the stream type flowing between
> stages. Each stage accepts and returns this type; internal
> computations may use wider/different scalars via the stage's own
> template parameters. Cross-precision stream transitions are a future
> extension.

## Architecture Diagrams

### A Typical DDC Chain

```text
  ADC  ─────▶  CIC ↓R1  ─────▶  HB ↓2  ─────▶  HB ↓2  ─────▶  FIR ↓Rn  ─────▶  I/Q
(100MHz)    (sinc^M         (sharp cut    (sharp cut      (channel      (~100 kHz)
             droop)          near 2x)      near 2x)        shaping)
                     ▲
                     │
                (compensator inserted between CIC and first HB)
```

### Per-Stage Bandwidth Budget

```text
  input BW ──┬────────────────── fs/2
             │
  after CIC ─┤───── fs/(2R1)
             │
  after HB ──┤─── fs/(4R1)
             │
  after HB ──┤── fs/(8R1)
             │
  after FIR ─┤── fs/(8R1*Rn)
             ▼
```

Each stage's filter only needs to be sharp enough to reject the band
beyond its own output Nyquist — not the full input Nyquist.

## Historical References

- E. B. Hogenauer, *"An Economical Class of Digital Filters for
  Decimation and Interpolation,"* IEEE Transactions on ASSP, vol. 29,
  no. 2, April 1981 — the CIC structure.
- R. E. Crochiere and L. R. Rabiner, *Multirate Digital Signal
  Processing*, Prentice Hall, 1983 — the multirate theory this chain
  rests on.
- F. J. Harris, *Multirate Signal Processing for Communication
  Systems*, Prentice Hall, 2004 — the modern treatment of cascaded
  decimation.
- Harris Semiconductor, *HSP50016 Digital Down Converter Datasheet*,
  1992 — the canonical DDC ASIC demonstrating the CIC → half-band →
  FIR cascade.
- A. Y. Kwentus, Z. Jiang, A. N. Willson, *"Application of Filter
  Sharpening to Cascaded Integrator-Comb Decimation Filters,"* IEEE
  Transactions on Signal Processing, 1997 — CIC droop compensation
  theory.
