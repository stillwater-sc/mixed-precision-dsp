---
title: End-to-End Scope Demo
description: Capstone walkthrough wiring the calibration equalizer, trigger, ring buffer, peak-detect decimator, display envelope, and measurement primitives into a working digital oscilloscope simulator with a per-stage mixed-precision sweep
---

The library ships a runnable digital-oscilloscope demo at
[`applications/scope_demo/`](https://github.com/stillwater-sc/mixed-precision-dsp/tree/main/applications/scope_demo)
that exercises the entire `instrument/` module end-to-end and **sweeps
precision per stage**, not uniformly across the whole pipeline. A
simulated ADC produces a 50 MHz square wave with a 5 ns glitch buried
in it; the signal flows through an `EqualizerFilter` (calibration
correction), then trigger → ring buffer → peak-detect decimator →
display envelope; the captured segment is run through the seven
cursor-style measurement primitives. The output is a CSV trace plus a
console summary table over five **precision plans** that mix and match
types across the stages.

This page walks through the topology, the per-stage precision contract
that follows from how each stage is implemented, and the headline
mixed-precision finding the sweep produces.

## Pipeline

```text
   simulate_clean_source       <-- the SOURCE, what you'd see at an ideal
       (50 MHz sq wave             signal generator. Used as the "ground
        + 5 ns glitch)              truth" for the SNR-vs-source metric.
            |
            v
   forward calibration FIR     <-- analog-front-end MODEL (probe + amp +
       (31-tap inline,             sample-and-hold). Distorts the source
        sw::dsp design)            with the same profile the equalizer
            |                       inverts on the digital side.
            v
   AWGN + 12-bit ADC           <-- thermal noise added at the ADC input;
            |                       quantization to 12 bits.
            v
   EqualizerFilter<EqCoeff,    <-- the ONLY arithmetic stage on the
       EqState, EqSample>          streaming digital path. Inverts the
       (FIR, 31 taps)              forward profile to recover the source.
            |
            v cast to StorageScalar
   EdgeTrigger + AutoTrigger
            |
            v
   TriggerRingBuffer (pre + 1 + post)
            |
            v
   PeakDetectDecimator (preserves glitch)
            |
            v
   render_envelope (-> pixel columns)
            |
            v
   measurements (rise time, RMS, ... — always accumulate in double internally)
            |
            v
   scope_demo.csv + console summary
```

## Per-stage precision contract

Each stage's precision requirement follows from its arithmetic, not
from a uniform "let's run everything in `T`" choice:

| Stage             | What it does                       | Precision driver                                      |
|-------------------|-------------------------------------|-------------------------------------------------------|
| ADC               | Quantize input                      | Fixed by the ADC (12-bit here)                        |
| EqualizerFilter   | FIR multiply-accumulate per sample  | **Streaming arithmetic** — narrowing here costs SNR   |
| EdgeTrigger       | Compares against threshold          | Comparison-only — narrowing is free                   |
| TriggerRingBuffer | Stores captured samples             | **Storage bandwidth** — narrowing trades memory for nothing |
| PeakDetectDecimator | min/max over R samples            | Comparison + selection — narrowing is free            |
| render_envelope   | min/max per pixel column            | Same                                                  |
| measurements      | Sum, sum-of-squares, interpolation  | Always `double` internally regardless of input type   |

So a *real* mixed-precision plan picks two independent things:

1. **EqCoeff / EqState / EqSample** — the calibration FIR's three
   scalars, controlling the cost of the streaming arithmetic.
2. **StorageScalar** — the type used by the trigger / ring buffer /
   peak-detect / envelope, controlling the memory bandwidth
   downstream of the equalizer.

The demo's `run_pipeline<EqCoeff, EqState, EqSample, StorageScalar>`
template takes the four scalars independently. Each row of the sweep
table is a different (EQ-tuple, Storage) plan.

## Pre-distortion: making the equalizer's job realistic

The demo models the analog front end **explicitly** rather than
pretending the equalizer corrects a clean signal. In `simulate_adc()`:

1. `simulate_clean_source()` builds the SOURCE signal — what you'd see
   at the output of an ideal signal generator (50 MHz square + glitch,
   no noise, no quantization, no profile coloring).
2. The source is run through a **forward calibration FIR** designed
   from the same `CalibrationProfile` the equalizer inverts — but
   **without** the inversion step. This filter applies the profile's
   magnitude and phase response to the source, modeling what the
   probe + amplifier + sample-and-hold network does to the signal
   before it reaches the ADC.
3. AWGN is added (front-end thermal noise, added at the ADC input).
4. Samples are quantized to 12 bits.

The equalizer on the digital side then inverts the same profile,
recovering the source. The post-equalizer output is a delayed,
slightly noise-degraded copy of the source — which is exactly what
a real scope's calibration loop produces.

### Why this matters for the precision sweep

Without pre-distortion, the equalizer would be applying a tiny
correction to a clean signal — the FIR multiply-accumulate would
barely exercise its arithmetic precision, and posit16 vs. double
would look identical at the output. With pre-distortion, the
equalizer is doing **substantial** work — boosting +10 dB at Nyquist
to invert -10 dB attenuation — and the per-stage precision shows up
clearly in the SNR table.

### A note on FIR settling

Forward and inverse FIRs each have a 15-sample group delay (for the
31-tap design); their cascade has a combined 30-sample settling
transient at the start of the stream. The demo skips these samples
before feeding the trigger pipeline so the captured pre-trigger
window contains steady-state carrier, not FIR ringing. (If you forget
to skip them, the trigger fires inside the transient and the captured
segment is mostly garbage. The cost — discarding 30 samples — is
negligible vs the 8192-sample stream length.)

## Calibration profile

The synthetic profile models a realistic high-bandwidth scope front
end with a -3 dB corner near 100 MHz:

```cpp
freqs    = {0,  50e6, 100e6, 250e6, 500e6};   // up to Nyquist
gains_dB = {0, -0.5,  -3.0,  -6.0, -10.0};
phases   = {0, -0.10, -0.20, -0.40, -0.60};
```

In-band content (the 50 MHz carrier) is barely touched (-0.5 dB);
above the corner, attenuation grows progressively to -10 dB at
Nyquist. Phase walks roughly linearly with frequency, modeling the
front end's group delay.

The aggressiveness here is calibrated to what a 31-tap
Hamming-windowed FIR cascade can faithfully invert — too much
attenuation at Nyquist (e.g., -18 dB) and the inverse FIR runs into
its own bandwidth limit, leaving residual error in the equalized
signal. -10 dB is the deepest attenuation that still gives sample-
level SNR-vs-source > 30 dB on the reference plan.

## Precision plans

Five named plans, each picking each stage's type independently:

```cpp
// reference: all double
run_pipeline<double, double, double, double>(...);

// High-precision EQ + ADC-native fixpnt storage. The streaming
// arithmetic stays full-precision; storage drops 4x.
run_pipeline<double, double, double, fixpnt<16,12>>(...);

// Narrow EQ in posit32 + double storage. Isolates the cost of
// narrowing only the streaming arithmetic.
run_pipeline<posit<32,2>, posit<32,2>, posit<32,2>, double>(...);

// Narrow EQ in posit16 + double storage. The headline mixed-
// precision case: 16-bit streaming arithmetic vs. comparison-only
// downstream.
run_pipeline<posit<16,2>, posit<16,2>, posit<16,2>, double>(...);

// FPGA-pragmatic: float EQ + ADC-native fixpnt storage.
run_pipeline<float, float, float, fixpnt<16,12>>(...);
```

## Sweep result

```text
plan (EQ x storage)             B/samp glitch?    peak  rise  freq(MHz) SNRref SNRsrc
-------------------------------------------------------------------------------------
reference (double x double)          8    PASS   0.972  0.81     50.001    inf  33.49
eq_double_storage_fx16 (double..)    2    PASS   0.972  0.81     50.001  76.70  33.49
eq_posit32_storage_double (p32..)    8    PASS   0.972  0.81     50.001 162.10  33.49
eq_posit16_storage_double (p16..)    8    PASS   0.972  0.81     50.001  66.69  33.51
eq_float_storage_fx16 (float ..)     2    PASS   0.972  0.81     50.001  76.70  33.49
```

Now that the equalizer is undoing real distortion (not correcting a
clean signal), the carrier measurements recover the **source** values
within the 5% acceptance the v0.6 demo had to relax:

- Frequency: 50.001 MHz vs source 50.000 MHz (0.002% error)
- Rise time: 0.81 samples vs source 0.80 (1.3% error)
- Glitch peak: 0.972 vs source 0.95 (2.3% over — small overshoot from
  the FIR cascade's edge response, well within physical limits)

### What the SNR columns actually mean

Two SNR metrics, each answering a different question:

**`SNRref`** — vs the reference (all-double) plan's rendered envelope:

```text
SNRref(plan) = 10 * log10(sum(reference^2) / sum((reference - plan)^2))
```

`inf` for the reference plan (vs itself). Higher = "this plan's
pipeline output matches the all-double reference more closely" — i.e.,
less precision-induced drift. The classic apples-to-apples comparison
across plans of the same pipeline.

**`SNRsrc`** — vs the original clean source signal:

```text
SNRsrc(plan) = 10 * log10(sum(source^2) / sum((post_equalizer - source_delayed)^2))
```

(`source_delayed` accounts for the combined forward + inverse FIR
group delay of `eq_taps - 1` samples.) This metric answers the
**equalizer's reason for existing**: how well does it un-distort the
front-end-distorted ADC samples back to the source? All five plans
score ~33.5 dB — limited by the FIR cascade's edge-response error and
the ADC's 12-bit quantization noise, not by precision narrowing
within the equalizer.

## The mixed-precision finding

Two independent precision dimensions, with very different cost
profiles:

### Storage narrowing is cheap

`eq_double_storage_fx16` runs the equalizer in full-precision `double`
and stores everything downstream in `fixpnt<16,12>`. The result:
**4× memory reduction (8 → 2 bytes/sample)** at a cost of ~77 dB
SNRref — well below any practical noise floor.

That's a real-but-acceptable cost. The ADC produces 12-bit samples;
storing them in 16-bit fixpnt is essentially storing them at native
resolution. The only quantization is at the storage boundary, and
because every downstream stage is comparison-only, that quantization
doesn't compound.

### Streaming-arithmetic narrowing costs SNR proportionally

`eq_posit16_storage_double` keeps storage at `double` (no memory
reduction) but narrows the equalizer's streaming arithmetic to
`posit16`. The result: **~67 dB SNRref**, which is ~10 dB *worse*
than the storage-narrowing-only plan despite using 4× *more* memory.

Why? The equalizer is a 31-tap FIR multiply-accumulate, and now —
thanks to pre-distortion — it's doing **substantial** arithmetic
work to invert the calibration profile (boosting +10 dB at Nyquist).
At each tap, posit16's ~12-bit fraction precision rounds the partial
product. Those rounding errors accumulate across the 31 taps.
Repeated arithmetic in a 16-bit type is fundamentally noisier than
repeated copies of a 16-bit type.

The pre-distortion stage is what makes this finding meaningful: in
the v0.6 demo (clean source, no pre-distortion), the equalizer's
correction was so small that posit16 and double tied. With realistic
front-end distortion to invert, the precision gap opens up.

### The headline takeaway

> **Narrow your storage, not your arithmetic.**

When the streaming path forks into "things that compute on values"
and "things that just move them around", precision matters
disproportionately on the compute side. Memory bandwidth — usually
the dominant cost in a high-rate scope — narrows for free as long as
the compute stage maintains enough precision.

The `eq_float_storage_fx16` row is the FPGA-pragmatic version of this
lesson: float for the equalizer (cheap on most fabric) plus
fixpnt<16,12> for storage gives you 4× memory reduction *and* ~77 dB
SNRref — better than pure posit16 EQ.

### Why all plans tie on SNRsrc

The five plans show a 100 dB spread on SNRref (162 → 67 dB) but tie
at ~33.5 dB on SNRsrc. That isn't a contradiction:

- **SNRref** measures *precision-induced drift between plans* — it's
  bounded by the precision of the narrowest type in the pipeline.
- **SNRsrc** measures *distance from the source signal* — it's
  bounded by the FIR cascade's edge-response error and the ADC's
  12-bit quantization noise, neither of which the precision sweep
  changes.

So even posit16 EQ recovers the source within ~33.5 dB (the cascade-
limited ceiling); narrowing precision further than what the
non-precision noise sources allow doesn't move the SNRsrc needle.
That's the cleanest available demonstration that *the right
precision is the one that matches the surrounding noise floor* —
not the highest one available, not the lowest your hardware can
afford. Pick precision to match the system's other noise sources;
spending more is wasted, spending less degrades.

## Two non-obvious pitfalls (captured in code + docs)

The integration surfaced two issues worth recording so the next demo
author doesn't repeat them:

### Off-by-one trigger trap

A naive integration loop

```cpp
for (...) {
    if (auto_trig.process(x)) ring.push_trigger(x);
    else                       ring.push(x);
}
```

silently drops one sample per re-fire after the first trigger.
`push_trigger()` is a no-op when the ring is in `Capturing` state, but
the corresponding `push()` doesn't run either because the
`if/else` already chose the wrong branch. The captured segment then
has one missing sample per ~20 (every carrier rising edge), which
compresses the apparent period from 20 samples to ~19 and biases
measured frequency upward by ~5%.

The demo gates `push_trigger` with a `triggered` flag so only the first
fire takes the trigger path; everything afterwards goes through `push`.

### Square-wave FP boundary

`sin(2*pi*f*t) >= 0` is unstable at sample boundaries where
`sin(k*pi)` returns tiny noise that flips sign unpredictably
(`sin(6*pi)` came out positive on x86 in one build, shortening one
low half-cycle by a sample and biasing period measurements).
Replaced with an integer-phase counter:

```cpp
const std::size_t half_period_samples = round(0.5 * Fs / f);
const std::size_t phase_n = n % (2 * half_period_samples);
const double sq = (phase_n < half_period_samples) ? +amp : -amp;
```

## Per-stage timing and the 10 GSPS comparison

```text
=== Per-stage timing (reference plan) ===
  equalizer        ~12 ms total   ~15 us/sample    <-- dominant
  trigger+ring        66 us total      85 ns/sample
  peak_detect         79 us total     101 ns/sample
  render_envelope     34 us total      43 ns/sample
  measurements        46 us total      59 ns/sample
  TOTAL            ~12.1 ms total  ~15.5 us/sample
```

The equalizer dominates the per-stage cost — it's the only stage doing
arithmetic, and it does N\_taps multiplies + adds per sample. Real
10 GSPS scopes implement the equalizer in an ASIC pipeline with
hundreds of parallel taps; this CPU implementation is for
*understanding the gap*, not closing it.

## Out of scope (deferred)

The original v0.6 capstone ([#152](https://github.com/stillwater-sc/mixed-precision-dsp/issues/152))
explicitly deferred several items; the v0.7 follow-up
([#172](https://github.com/stillwater-sc/mixed-precision-dsp/issues/172))
landed the pre-distortion stage. What's still deferred:

- **Real ADC interfacing** (e.g. TI ADC12DJ5200RF). Simulated only.
- **Image rendering** of the envelope. CSV is the deliverable.
- **Multi-channel demonstration**. Tracked separately as
  [#173](https://github.com/stillwater-sc/mixed-precision-dsp/issues/173).
- **Public `design_fir_from_profile()` API.** The forward-FIR design
  helper is currently inline in `scope_demo.cpp` (per #172's
  out-of-scope note). Lifting it into `instrument/calibration.hpp`
  alongside the equalizer's inverse-FIR designer is a separate
  refactor.
- **Real (measured) calibration profiles.** Synthetic only.

## See also

- [Spectrum Analyzer Overview](./spectrum-analyzer-overview/) and the
  [End-to-End Spectrum Analyzer Demo](./spectrum-analyzer-demo/) — the
  companion analyzer-side capstone built on the same `instrument/`
  module. It demonstrates the same
  precision-of-storage-vs-precision-of-arithmetic tradeoff in a
  different topology, with the FFT (or the RBW filter cascade) playing
  the role the equalizer plays here.
