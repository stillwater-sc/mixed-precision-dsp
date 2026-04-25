---
title: Multirate Pattern Catalog
description: Problem-to-API mapping for canonical multirate signal-processing patterns, with worked examples and links to per-component reference pages
---

This page maps the canonical multirate problems to the library's
APIs. For the theory behind these mappings — the Noble identity,
the polyphase decomposition theorem, the historical progression —
see the [Multirate Overview](./overview/).

## Quick reference

| Pattern | Use case | Library API | Header |
|---|---|---|---|
| [Integer decimation by M](#integer-decimation) | Lower the sample rate by an integer factor | `PolyphaseDecimator` | `filter/fir/polyphase.hpp` |
| [Integer interpolation by L](#integer-interpolation) | Raise the sample rate by an integer factor | `PolyphaseInterpolator` | `filter/fir/polyphase.hpp` |
| [Rational L/M conversion](#rational-lm-conversion) | e.g. 44.1 kHz ↔ 48 kHz audio | `RationalResampler` | `conditioning/src.hpp` |
| [Sharp 2:1 decimation](#sharp-21-decimation) | Tight stopband near $f_s/4$ | `HalfBandFilter` | `acquisition/halfband.hpp` |
| [Bulk rate reduction at high rate](#bulk-rate-reduction-cic) | First-stage decimation after a GHz ADC | `CICDecimator` | `acquisition/cic.hpp` |
| [IF → baseband](#if-to-baseband-ddc) | Pull a band of interest down to 0 Hz | `DDC` | `acquisition/ddc.hpp` |
| [Multi-stage decimation cascade](#multi-stage-cascade) | CIC → HB → polyphase chain | `DecimationChain` | `acquisition/decimation_chain.hpp` |
| [Channelizer (M-channel filter bank)](#channelizer) | Split a wideband signal into channels | **gap** — compose from `PolyphaseDecimator` + FFT |

Every row except the last has a first-class library API. The
channelizer is a documented composition gap — see that section for
the construction.

---

## Integer decimation

**Problem:** A signal sampled at $f_s$ contains content only up to
$f_s / (2M)$, and you want to lower the sample rate by an integer
factor $M$ to match.

**Naive solution:** Filter the signal with a lowpass at $f_s / (2M)$,
then keep every $M$th sample. Wastes $M-1$ out of every $M$ filter
multiplications.

**Library solution:** [`PolyphaseDecimator<CoeffScalar, StateScalar, SampleScalar>`](../../acquisition/polyphase-decimator/)
implements the polyphase decomposition. A length-$N$ filter
decimating by $M$ runs at $\sim N/M$ multiplies per *input* sample
(equivalently, $N$ multiplies per *output* sample) instead of $N$
per input sample.

```cpp
#include <sw/dsp/filter/fir/polyphase.hpp>
using namespace sw::dsp;

// Design an anti-alias lowpass for ↓4 (cutoff just below fs/8)
auto win = hamming_window<double>(64);
auto taps = design_fir_lowpass<double>(64, 0.45 / 4.0, win);

// Polyphase decimator with double-precision coefficients/state
PolyphaseDecimator<double, double, double> decim(taps, 4);

mtl::vec::dense_vector<double> input = ...;     // length-N input
auto output = decim.process_block(input);       // length-N/4
```

**See also:** [Polyphase Decimator reference](../../acquisition/polyphase-decimator/).

---

## Integer interpolation

**Problem:** A signal sampled at $f_s$ needs to be expressed at
$L \cdot f_s$, e.g., for D/A conversion at a higher oversampling
ratio.

**Naive solution:** Insert $L-1$ zeros between each input sample,
then run a lowpass at $L \cdot f_s$ to remove the resulting spectral
images. The filter sees mostly zeros — most of its multiplications
are wasted.

**Library solution:** `PolyphaseInterpolator<CoeffScalar, StateScalar, SampleScalar>`
runs the dual of polyphase decimation. The filter coefficients split
into $L$ sub-filters; each input sample feeds all $L$ sub-filters,
and the outputs are emitted in turn. Cost is $\sim N$ multiplies per
*input* sample, distributed evenly across the $L$ output samples.

```cpp
#include <sw/dsp/filter/fir/polyphase.hpp>

PolyphaseInterpolator<double, double, double> interp(taps, /*L=*/4);
auto output = interp.process_block(input);   // length 4N
```

The class lives in `sw/dsp/filter/fir/polyphase.hpp` alongside
`PolyphaseDecimator`. See the polyphase header for the full API.

---

## Rational L/M conversion

**Problem:** Convert from one rate to another where the ratio is a
non-trivial rational number, e.g., 44.1 kHz audio to 48 kHz
($L/M = 160/147$) or vice versa.

**Naive solution:** Interpolate by $L$ (with anti-image filter at
$L \cdot f_s$), then decimate by $M$ (with anti-alias filter). Two
big filters, both running at the unhelpful $L \cdot f_s$ rate.

**Library solution:** `RationalResampler<CoeffScalar, StateScalar, SampleScalar>`
(in `sw/dsp/conditioning/src.hpp`) combines polyphase interpolation
by $L$ and decimation by $M$ into a single time-register pipeline
that emits exactly the output samples that survive both stages. The
shared anti-alias / anti-image filter is auto-designed as a
Kaiser-windowed sinc of sufficient length to suppress aliases below
a configurable noise floor.

> **Docs gap:** `RationalResampler` doesn't yet have a dedicated
> reference page under `conditioning/`. The header comments in
> `sw/dsp/conditioning/src.hpp` are the source of truth until that
> page is filed as a follow-up.

```cpp
#include <sw/dsp/conditioning/src.hpp>
using namespace sw::dsp;

// 44.1 kHz -> 48 kHz: L=160, M=147
RationalResampler<double, double, double> rs(/*L=*/160, /*M=*/147);
auto out_48k = rs.process_block(in_44k1);
```

The class auto-designs the prototype filter from `(L, M, stopband_dB)`
parameters; you don't need to design taps yourself. See
`sw/dsp/conditioning/src.hpp` for the full constructor signature.

---

## Sharp 2:1 decimation

**Problem:** You need to decimate by 2 with a *sharp* transition
near $f_s / 4$, where a generic FIR would need a long impulse
response.

**Library solution:** [`HalfBandFilter`](../../acquisition/halfband/)
exploits the half-band identity (Mintzer 1982): in a properly
designed half-band filter, every other coefficient (except the
center tap) is **exactly zero**. So a length-$(4K+3)$ half-band has
only $K+2$ non-zero taps — the per-output cost is roughly half of a
generic FIR of the same length.

```cpp
#include <sw/dsp/acquisition/halfband.hpp>
using namespace sw::dsp;

// Design a length-31 half-band (must be of the form 4K+3 — 31 = 4·7+3).
// transition_width is normalized to fs (0.05 = 5% of the sample rate).
auto taps = design_halfband<double>(31, 0.05);

// Construct the filter from the designed taps
HalfBandFilter<double, double, double> hb(taps);

// Block-decimate-by-2: returns a length-N/2 dense_vector
auto output = hb.process_block_decimate(input);

// Per-sample streaming variant returns std::pair<bool, SampleScalar>:
//   auto [ready, y] = hb.process_decimate(x);
//   if (ready) { /* y is the next decimated output */ }
```

Cascade two or three half-bands for ↓4 or ↓8 with a transition that
gets progressively sharper at each stage's reduced rate. The
[Decimation Chain](#multi-stage-cascade) section shows this pattern.

**See also:** [Half-Band Filter reference](../../acquisition/halfband/).

---

## Bulk rate reduction (CIC)

**Problem:** Your input rate is so high (>>100 MHz) that any FIR
multiplier-based filter is too expensive for the first decimation
stage.

**Library solution:** [`CICDecimator`](../../acquisition/cic/)
implements Hogenauer's 1981 cascaded-integrator-comb structure: $M$
integrators at the input rate, downsample by $R$, $M$ comb stages at
the output rate. **No multipliers** — only adders and subtractors.

The cost is a $\text{sinc}^M$ passband droop that downstream stages
have to compensate (a half-band stage with a slight inverse-sinc
shape, or a dedicated [droop compensator](../../acquisition/decimation-chain/#cic-droop-compensation)).

```cpp
#include <sw/dsp/acquisition/cic.hpp>

// 4-stage CIC, decimating by 64
CICDecimator<int64_t, int32_t> cic(/*R=*/64, /*M=*/4);
auto coarse_baseband = cic.process_block(adc_samples);
```

**Important:** the CIC integrator structurally requires
two's-complement-wrapping state. `fixpnt` works; `float` and `posit`
accumulate uncorrectable drift on DC bias. See the
[SDR demo](../../acquisition/demo/) for an empirical demonstration
of this constraint.

**See also:** [CIC reference](../../acquisition/cic/).

---

## IF to baseband (DDC)

**Problem:** A real-valued ADC stream contains a band of interest
centered at some intermediate frequency $f_\text{IF}$, and you need
the complex baseband for demodulation.

**Library solution:** [`DDC`](../../acquisition/ddc/) composes:

1. An [`NCO`](../../acquisition/nco/) at $f_\text{IF} / f_s$ producing
   $\cos(\omega n) + j\sin(\omega n)$
2. A complex multiplier (mixer) that shifts the spectrum down by $f_\text{IF}$
3. A polyphase decimator that anti-alias-filters and reduces the rate

```cpp
#include <sw/dsp/acquisition/ddc.hpp>

DDC<double, double, double, PolyphaseDecimator<double, double, double>>
    ddc(/*center_frequency=*/0.1, /*sample_rate=*/1.0, decim);
auto iq_baseband = ddc.process_block(real_input);
```

The DDC is the primary front-end of any SDR receiver. See the
[SDR Receiver Front-End Overview](../../acquisition/overview/) for
how this composes into a full receiver chain.

**See also:** [DDC reference](../../acquisition/ddc/).

---

## Multi-stage cascade

**Problem:** A single huge decimation factor (say ↓1024 from 100 MHz
to 100 kHz) won't be efficient as a single filter. Each stage of a
multistage cascade can target its own bandwidth and arithmetic
constraints.

**Library solution:** [`DecimationChain<SampleScalar, Stage1, Stage2, ...>`](../../acquisition/decimation-chain/)
is a variadic-tuple wrapper that composes any sequence of decimators
into a single pipeline. Each stage operates at its own rate; samples
flow through the tuple in order.

```cpp
#include <sw/dsp/acquisition/decimation_chain.hpp>

// CIC ↓64 -> HalfBand ↓2 -> Polyphase ↓2 = total ↓256
using Chain = DecimationChain<double,
    CICDecimator<int64_t, int32_t>,
    HalfBandFilter<double, double, double>,
    PolyphaseDecimator<double, double, double>>;

Chain chain(/*sample_rate=*/100e6,
            CICDecimator<...>(64, 4),
            HalfBandFilter<...>(31),
            PolyphaseDecimator<...>(taps, 2));

auto output = chain.process_block(adc_samples);
```

`DecimationChain` accepts any tuple ordering; the ordering shown
above (CIC at the top to take advantage of multiplier-free bulk
reduction, half-band in the middle for sharp 2:1, polyphase at the
bottom for final shaping) is the canonical receiver-chain pattern,
but is not enforced.

**See also:** [Decimation Chain reference](../../acquisition/decimation-chain/).

---

## Channelizer

**Problem:** A wideband signal contains $M$ contiguous frequency
channels, and you want each channel separately at its own
(reduced) rate.

**Naive solution:** Run $M$ independent DDCs, each tuning to one
channel and decimating by $M$. Cost is $M$× a single DDC.

**Bellanger's solution:** A single polyphase filter followed by an
inverse FFT can produce all $M$ channel outputs simultaneously, at
roughly the cost of *one* polyphase filter plus one $M$-point IFFT
per output sample. This is the polyphase channelizer.

**Library status:** The library doesn't yet ship a dedicated
`Channelizer` class, but the construction is straightforward from
existing primitives:

```cpp
// Sketch — not yet a library class
// 1. Length-MN polyphase prototype filter, decomposed into M subfilters
//    each of length N
// 2. For each output frame:
//      - Push M new input samples through the M sub-filters in
//        parallel (each subfilter advances by 1 sample)
//      - Stack the M sub-filter outputs into a length-M vector
//      - Take the M-point IFFT of that vector
//      - Output is M parallel channels, each at fs/M
```

**This is a documented gap.** Filing a follow-up issue to add either
a thin `Channelizer` wrapper class or a worked-example demo would be
appropriate. The polyphase + FFT primitives needed for the
construction are already in the library
(`filter/fir/polyphase.hpp` + `spectral/fft.hpp`).

---

## Choosing among similar APIs

A few rules of thumb when more than one API could plausibly apply:

**Decimation by 2:** Use `HalfBandFilter`, not `PolyphaseDecimator`.
The half-band's ~75% zero coefficients make it cheaper than a
generic polyphase by 2.

**Decimation by a large factor at the highest rate:** Use
`CICDecimator` for the bulk reduction, then half-bands and a
polyphase for shaping. A single polyphase by, say, 64 would need a
~hundreds-of-taps prototype filter running at the output rate — the
CIC + cascade is much cheaper.

**Rational L/M with $L$ or $M$ very large:** `RationalResampler`
auto-designs the prototype filter, but if the ratio is very lopsided
(e.g., ×100 / ÷1) consider whether a multistage cascade would be
better. For the typical audio ratios (160/147 etc.) the single
`RationalResampler` stage is the right answer.

**Mixed integer + rational:** Cascade them. Use `CICDecimator` for
the first integer ↓R, then `RationalResampler` for the final L/M
shaping at the much lower rate.
