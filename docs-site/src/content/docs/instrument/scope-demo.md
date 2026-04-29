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
+---------------------------+    +-----------------------------------+
| simulate_adc(N_bits, Fs)  | -> | EqualizerFilter                   |
|   50 MHz square wave      |    |   <EqCoeff, EqState, EqSample>    |
|   +/- 0.5 amplitude       |    |   FIR inverse of calibration      |
|   5 ns +0.95 glitch       |    |   profile, 31 taps                |
|   AWGN sigma=0.005        |    |   <-- the ONLY arithmetic stage   |
|   12-bit uniform quant    |    |       on the streaming path       |
+---------------------------+    +-----------+-----------------------+
                                             |
                                             v cast to StorageScalar
                                 +-----------+-----------+
                                 | EdgeTrigger           |
                                 | + AutoTriggerWrapper  |
                                 +-----------+-----------+
                                             |
                                             v
                                 +-----------+-----------+
                                 | TriggerRingBuffer     |
                                 |   pre + 1 + post      |
                                 +-----------+-----------+
                                             |
                                             v
                                 +-----------+-----------+
                                 | PeakDetectDecimator   |
                                 |   preserves glitch    |
                                 +-----------+-----------+
                                             |
                                             v
                                 +-----------+-----------+
                                 | render_envelope       |
                                 |   -> N pixel columns  |
                                 +-----------+-----------+
                                             |
                                             v
                                 +-----------+-----------+
                                 | measurements (always  |
                                 |   accumulate in       |
                                 |   double internally)  |
                                 +-----------+-----------+
                                             |
                                             v
                                 scope_demo.csv + console
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

## Calibration profile

A synthetic mild rolloff:

```cpp
freqs    = {0,  50e6, 100e6, 250e6, 500e6};   // up to Nyquist
gains_dB = {0, -0.5,  -2.0,  -3.0,  -3.0};
phases   = {0, -0.10, -0.20, -0.30, -0.30};
```

This models a typical analog front end with a 100 MHz-ish corner.
The equalizer's job is to apply the inverse — a small, mostly-flat
boost — so the streaming output is closer to the source signal.

A more aggressive profile (e.g., -10 dB at Nyquist) would force the
equalizer to apply +10 dB at Nyquist, which would ring on every sharp
edge and turn the square wave into something that doesn't look like a
square wave anymore. That's a real tradeoff scope designers face. For
this demo we keep the corrections under +2 dB so the analytical
measurements remain interpretable across all configs and the
**precision-impact comparison** is the headline.

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
plan (EQ x storage)              B/samp glitch?    peak  rise  freq(MHz)  SNR(dB)
---------------------------------------------------------------------------------
reference (double x double)           8    PASS   1.459  9.36     58.507      inf
eq_double_storage_fx16 (double..)     2    PASS   1.458  9.36     58.510    78.75
eq_posit32_storage_double (p32..)     8    PASS   1.459  9.36     58.507   162.92
eq_posit16_storage_double (p16..)     8    PASS   1.458  9.36     58.507    66.32
eq_float_storage_fx16 (float ..)      2    PASS   1.458  9.36     58.510    78.75
```

The carrier measurements (rise time, frequency, glitch peak) reflect
the **equalized** signal, not the raw source — the equalizer reshapes
edges (its ~31-sample group delay + impulse-response width inflates
the apparent 10/90 rise time relative to the input's one-sample
edge). All five plans agree on the qualitative measurement, which is
the right consistency check.

### What the SNR column actually means

`SNR(dB) = inf` for the reference plan (vs itself). For every other
plan, SNR is computed against the reference plan's rendered envelope:

```text
SNR_dB(plan) = 10 * log10(sum(reference^2) / sum((reference - plan)^2))
```

So a higher number means "this plan's pipeline output matches the
all-double reference more closely" — i.e., less precision-induced
drift.

## The mixed-precision finding

Two independent precision dimensions, with very different cost
profiles:

### Storage narrowing is cheap

`eq_double_storage_fx16` runs the equalizer in full-precision `double`
and stores everything downstream in `fixpnt<16,12>`. The result:
**4× memory reduction (8 → 2 bytes/sample)** at a cost of ~80 dB SNR.

That's a real-but-acceptable cost. The ADC produces 12-bit samples;
storing them in 16-bit fixpnt is essentially storing them at native
resolution. The only quantization is at the storage boundary, and
because every downstream stage is comparison-only, that quantization
doesn't compound.

### Streaming-arithmetic narrowing costs SNR proportionally

`eq_posit16_storage_double` keeps storage at `double` (no memory
reduction) but narrows the equalizer's streaming arithmetic to
`posit16`. The result: **66 dB SNR**, which is ~12 dB *worse* than
the storage-narrowing-only plan despite using 4× *more* memory.

Why? The equalizer is a 31-tap FIR multiply-accumulate. At each tap,
posit16's ~12-bit fraction precision rounds the partial product. Those
rounding errors accumulate across the 31 taps. Repeated arithmetic in
a 16-bit type is fundamentally noisier than repeated copies of a
16-bit type.

### The headline takeaway

> **Narrow your storage, not your arithmetic.**

When the streaming path forks into "things that compute on values"
and "things that just move them around", precision matters
disproportionately on the compute side. Memory bandwidth — usually
the dominant cost in a high-rate scope — narrows for free as long as
the compute stage maintains enough precision.

The `eq_float_storage_fx16` row is the FPGA-pragmatic version of this
lesson: float for the equalizer (cheap on most fabric) plus
fixpnt<16,12> for storage gives you 4× memory reduction *and* 78 dB
SNR — better than pure posit16 EQ.

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

The issue ([#152](https://github.com/stillwater-sc/mixed-precision-dsp/issues/152))
explicitly defers:

- **Real ADC interfacing** (e.g. TI ADC12DJ5200RF). Simulated only.
- **Image rendering** of the envelope. CSV is the deliverable.
- **Multi-channel demonstration**. Single-channel for v0.6.
- **Pre-distortion of the input** with the calibration profile so the
  equalizer is undoing a real frequency-domain distortion. Currently
  the equalizer applies a small correction to the clean source. A
  follow-up could synthesize a profile-distorted signal and let the
  equalizer flatten it for an even more realistic demo.

## See also

- [Spectrum Analyzer Overview](./spectrum-analyzer-overview/) and the
  [End-to-End Spectrum Analyzer Demo](./spectrum-analyzer-demo/) — the
  companion analyzer-side capstone built on the same `instrument/`
  module. It demonstrates the same
  precision-of-storage-vs-precision-of-arithmetic tradeoff in a
  different topology, with the FFT (or the RBW filter cascade) playing
  the role the equalizer plays here.
