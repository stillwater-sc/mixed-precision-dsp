---
title: End-to-End Scope Demo
description: Capstone walkthrough wiring the trigger, ring buffer, peak-detect decimator, display envelope, and measurement primitives into a working digital oscilloscope simulator
---

The library ships a runnable digital-oscilloscope demo at
[`applications/scope_demo/`](https://github.com/stillwater-sc/mixed-precision-dsp/tree/main/applications/scope_demo)
that exercises the entire `instrument/` module end-to-end: a simulated
ADC produces a 50 MHz square wave with a 5 ns glitch buried in it, the
signal flows through trigger → ring buffer → peak-detect decimator →
display envelope, and the captured segment is run through the seven
cursor-style measurement primitives. The final output is a CSV trace
plus a console summary table swept across six precision configurations.

This page walks through the topology, the measurement contract, and the
mixed-precision finding the sweep produces (which is the opposite of
what the SDR demo shows — a feature, not a bug).

## Pipeline

```text
+----------------------------+    +-------------------------+
| simulate_adc(N_bits, Fs)   | -> | EdgeTrigger             |
|   50 MHz square wave       |    |   + AutoTriggerWrapper  |
|   +/- 0.5 amplitude        |    |   level=0, hyst=0.05,   |
|   5 ns +0.95 glitch buried |    |   slope=Rising          |
|   AWGN sigma=0.005         |    +-----------+-------------+
|   12-bit uniform quant     |                |
+----------------------------+                v
                                  +-------------------------+
                                  | TriggerRingBuffer       |
                                  |   pre=256 + 1 + post=768|
                                  +-----------+-------------+
                                              |
                                              v
                                  +-------------------------+
                                  | PeakDetectDecimator R=2 |
                                  |   preserves glitch peak |
                                  +-----------+-------------+
                                              |
                                              v
                                  +-------------------------+
                                  | render_envelope         |
                                  |   -> 200 pixel columns  |
                                  +-----------+-------------+
                                              |
                                              v
                                  +-------------------------+
                                  | measurements            |
                                  |   peak_to_peak, mean,   |
                                  |   rms, rise_time,       |
                                  |   period, frequency     |
                                  +-----------+-------------+
                                              |
                                              v
                                  scope_demo.csv + console
```

## Test signal

The synthesized waveform has three pieces:

- A 50 MHz square wave at +/- 0.5 amplitude. The square wave is
  generated with an integer-phase counter rather than
  `sin(2*pi*f*t) >= 0`, because `sin(k*pi)` for integer `k` returns
  numerically noisy values that flip sign unpredictably and bias the
  period measurement.
- A 5 ns positive glitch (peak amplitude 0.95) injected at 500 ns into
  the stream. The glitch *replaces* the carrier value during its
  window, modeling an EMI pulse hitting a probe — that way the glitch
  peak is well-defined regardless of what the carrier was doing
  underneath.
- AWGN at sigma=0.005, well below the carrier amplitude so that the
  trigger fires reliably without hysteresis-defeating chatter.

The ADC stage runs the noisy signal through a 12-bit uniform quantizer
(`floor(x / q_step)` clamped to `[-2^(N-1), 2^(N-1)-1]`), matching the
2's-complement code semantics in the SDR demo's `simulate_adc()`.

## Measurement window

The captured segment includes both the carrier and the glitch. Two
of the seven measurement primitives are sensitive to which sub-window
they read:

- **`rise_time_samples`**: bases its 10% / 90% thresholds on the
  segment's `peak_to_peak`, which the glitch lifts above the carrier.
  Reading rise time from the full segment would measure from the first
  carrier rising edge to the glitch's leading edge — a real number,
  but not the carrier rise time.
- **`period_samples` / `frequency_hz`**: detect zero-crossings, and
  the glitch's leading edge creates an extra rising crossing that
  biases the period average.

The demo sidesteps both by computing rise / period / frequency on a
glitch-free **measurement window** — the first `pre_glitch_window` (400)
samples of the captured segment, which contains several clean carrier
cycles and ends well before the glitch's 500 ns position. The full
segment is still used for `rms`, `mean`, and the rendered envelope (the
latter is where the headline glitch-survival metric lives).

This is a reusable pattern: aggregations want everything; transition-
based measurements want a representative quiet window.

## The off-by-one trigger trap

A subtlety surfaced during integration: the inner trigger fires at
*every* rising edge, not just the first one. The naive integration loop

```cpp
for (...) {
    if (auto_trig.process(x)) ring.push_trigger(x);
    else                       ring.push(x);
}
```

silently drops one sample per re-fire after the first trigger, because
`push_trigger()` is a no-op when the ring is in `Capturing` state and
the corresponding `push()` call doesn't happen on that iteration. The
resulting captured segment is missing one sample per ~20 (every carrier
rising edge), which compresses the apparent period from 20 samples to
~19 and biases the measured frequency upward by ~5%.

The demo gates `push_trigger` with a `triggered` flag so only the first
fire takes the trigger path; everything afterwards goes through `push`.
The fix is documented inline in `scope_demo.cpp` so the next demo
author sees it.

## Mixed-precision sweep

Six configs are run, all uniform (same scalar everywhere):

| Config            | Type                                    |
|-------------------|-----------------------------------------|
| uniform_double    | `double`                                |
| uniform_float     | `float`                                 |
| uniform_posit32   | `posit<32, 2>`                          |
| uniform_posit16   | `posit<16, 2>`                          |
| uniform_cfloat32  | `cfloat<32, 8, uint32_t, true,...>`     |
| uniform_fixpnt    | `fixpnt<32, 30>` (Q2.30)                |

The result on the demo signal:

```text
config               glitch?        peak  rise(samp)       rms   freq(MHz)     SNR(dB)
--------------------------------------------------------------------------------------
uniform_double          PASS       0.962        0.81     0.504      50.001         inf
uniform_float           PASS       0.962        0.81     0.504      50.001         inf
uniform_posit32         PASS       0.962        0.81     0.504      50.001         inf
uniform_posit16         PASS       0.962        0.81     0.504      50.001         inf
uniform_cfloat32        PASS       0.962        0.81     0.504      50.001         inf
uniform_fixpnt          PASS       0.962        0.81     0.504      50.001         inf
```

Every config produces **bit-identical** output — `SNR(dB) = inf`
against the uniform_double reference. This is the **headline finding**
of the scope demo and the structural opposite of what the SDR demo
produces.

### Why is scope DSP precision-insensitive?

The pipeline stages, from top to bottom:

| Stage                | Operations                                          |
|----------------------|-----------------------------------------------------|
| `EdgeTrigger`        | comparisons (`<`, `>`)                              |
| `AutoTriggerWrapper` | counter increment + comparison                      |
| `TriggerRingBuffer`  | array copy / index arithmetic                       |
| `PeakDetectDecimator`| min / max (comparisons + selection)                 |
| `render_envelope`    | min / max over per-pixel ranges (comparisons)       |
| `measurements`       | accumulate in `double` regardless of `SampleScalar` |

There is **no arithmetic on `SampleScalar` values** in the scope
streaming path. Every operation is either a comparison (which preserves
ordering across all the configured number systems for inputs in their
representable range), a copy (precision-preserving by definition), or
a min/max selection (which copies the surviving value bit-exactly).
The measurement primitives explicitly cast their inputs to `double`
for accumulation — see the `Mixed-precision contract` block in
[`measurements.hpp`](https://github.com/stillwater-sc/mixed-precision-dsp/blob/main/include/sw/dsp/instrument/measurements.hpp)
— so they don't introduce per-config drift either.

The contrast with the SDR demo is the lesson:

| Pipeline                       | Arithmetic in streaming path? | Per-config SNR? |
|--------------------------------|-------------------------------|-----------------|
| SDR / DDC                      | mixers + FIR + CIC            | Strongly varies |
| Scope                          | None — only compare / copy    | Bit-identical   |

A real scope's analog front-end is heavily dependent on amplifier and
ADC precision; the **digital** path is comparison-dominated and largely
type-blind. This demo verifies that the library's scope primitives
respect that.

### When *would* scope precision matter?

- **Very narrow types near saturation**: e.g. `posit8` would clamp the
  +0.95 glitch to its dynamic-range edge, attenuating the peak.
- **Asymmetric quantization grids**: `fixpnt<32, 30>` represents
  values in `[-2, +2)` exactly enough; `fixpnt<32, 16>` (Q16.16) would
  give the signal coarse stair-stepping that doesn't change the
  envelope much but quantizes RMS/mean visibly.
- **Calibration / equalizer stages**: a future version of this demo
  with an `EqualizerFilter` front-end would introduce arithmetic on
  the streaming path and produce the per-config SNR variation the SDR
  demo shows.

The current sweep is a **negative result by design**: the cleanly-
factored streaming path is comparison-only, so it should report
bit-identical envelopes — and does.

## Per-stage timing and the 10 GSPS comparison

The demo instruments each stage with `std::chrono::high_resolution_clock`
and reports `ns/sample` for the captured segment. Sample numbers from a
typical run:

```text
=== Per-stage timing (uniform_double reference) ===
  trigger+ring          71285 ns total       90.349 ns/sample
  peak_detect           89985 ns total      114.049 ns/sample
  render_envelope       33224 ns total       42.109 ns/sample
  measurements          49599 ns total       62.863 ns/sample
  TOTAL                244093 ns total      309.370 ns/sample

  10 GSPS budget: 0.100 ns/sample
  10 GSPS: NOT achievable on general-purpose CPU (would need 3093.7x speedup)
  Real 10 GSPS scopes use ASIC pipelines — this is an informational comparison.
```

10 GSPS = 100 ps / sample end-to-end. A general-purpose x86 core takes
roughly **3000x** that for the full chain. This isn't a defect — real
10 GSPS scopes use ASIC datapaths with hundreds-of-channel parallel
peak-detect blocks fed from the ADC's metal layer. The CPU comparison
exists to *quantify the gap*, not to close it.

If you want to push closer: the dominant stage is `peak_detect` at
~114 ns/sample. Vectorizing that loop (AVX2/AVX-512 min/max
intrinsics) would cut several factor-of-two off the total but not
close the 1000x gap.

## Output

A single `scope_demo.csv` is written, one row per (config, pixel)
pair. Columns:

```text
pipeline,config_name,coeff_type,state_type,sample_type,
pixel_index,envelope_min,envelope_max,
glitch_survived,glitch_peak,rise_time_samples,rms,mean,output_snr_db
```

The schema is a superset of the acquisition demo's CSV: the
`pixel_index / envelope_min / envelope_max` columns are scope-specific,
the rest are the standard precision-sweep columns so downstream
analysis tools that consume `acquisition_demo.csv` can also read
`scope_demo.csv` for the shared metrics.

## Out of scope (deferred)

The issue ([#152](https://github.com/stillwater-sc/mixed-precision-dsp/issues/152))
explicitly defers three pieces:

- **Real ADC interfacing** (e.g. TI ADC12DJ5200RF). Simulated only.
- **Image rendering** of the envelope. CSV is the deliverable; turning
  it into a PNG is a separate concern.
- **Multi-channel demonstration**. Single-channel for v0.6;
  multi-channel becomes natural once
  [`ChannelAligner`](https://github.com/stillwater-sc/mixed-precision-dsp/blob/main/include/sw/dsp/instrument/channel_aligner.hpp)
  is fully exercised.

Two further items are deferred from this PR but mentioned in the issue
as future quality-of-life work:

- **`EqualizerFilter` front-end** integration (calibration stage). The
  scope DSP primitives are quite separable from calibration; landing
  the calibration integration as a follow-up keeps the capstone
  scoped.
- **Selected mixed configurations** beyond the six uniform ones (e.g.
  `Coeff=double, State=double, Sample=posit16`). Useful once the
  pipeline is precision-sensitive — currently every combination
  produces the same envelope.
