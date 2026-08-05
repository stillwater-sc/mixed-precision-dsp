---
title: Audio Resampler Demo — 44.1 kHz ↔ 48 kHz
description: Mixed-precision rational sample-rate conversion demo — the canonical audio-industry rate conversion, measured across six numeric types with SNR, passband ripple, and stopband floor
---

The `audio_resampler` demonstrator (issue [#136](https://github.com/stillwater-sc/dsp/issues/136))
runs the **canonical audio-industry rate conversion** — 44.1 kHz to
48 kHz — through the library's `RationalResampler` across six numeric
types. It is the concrete case study behind the
[Rational L/M Conversion](./patterns/#rational-lm-conversion) entry in
the [Pattern Catalog](./patterns/).

## The 44.1 ↔ 48 kHz problem

44.1 kHz was the CD-audio sample rate (chosen for compatibility with
NTSC video timing). 48 kHz was the sample rate of the DAT format and
became the standard for video-industry audio (broadcast, film, VoIP,
DAWs). Every audio production toolchain that touches both worlds — a
DVD ripped to iTunes, a CD mixed against a video soundtrack, a
streaming service routing user-uploaded content — hits this
conversion. It is the single most-executed sample-rate conversion in
consumer audio.

The natural ratio is $48000 / 44100 = 320 / 294$. The `RationalResampler`
constructor immediately reduces this by $\gcd(48000, 44100) = 300$ to
$L / M = 160 / 147$: 160 upsampling insertion positions, 147 decimation
positions per output group. Everything else in the resampler runs at
this reduced ratio.

## Running the demo

```bash
cmake --preset ci
cmake --build build-ci --target audio_resampler -j4
./build-ci/applications/multirate_examples/audio_resampler/audio_resampler --csv=out.csv
```

By default the demo writes to `audio_resampler.csv` under the build
tree's `demo-output/` directory, so running it from the repository root
leaves nothing behind in the source tree. Passing `--csv=<path>`
overrides the destination. Every demo reports the path it actually
wrote.

## What the demo measures

The input is a **4-tone multitone** at $[100, 1000, 10000, 19000]$ Hz,
each at amplitude $0.25$ (sum peak = $1.0$). Tone placement is
deliberate: 100 Hz sits near the DC edge of the passband, 19 kHz sits
near the 20 kHz audio ceiling. Any passband tilt or transition-band
droop shows up as spread across the four tone levels.

For each numeric configuration the demo runs the input through
`RationalResampler`, then measures on a Kaiser-windowed FFT of the
steady-state output:

- **Tone level** (dB): peak magnitude in a ±20 Hz window around each
  test frequency. Deviation from the reference-config level reveals
  passband distortion.
- **Passband ripple** (dB): $\max(\text{tone levels}) - \min(\text{tone
  levels})$. Bounded from below by the anti-alias filter's own passband
  ripple.
- **Stopband floor** (dB): peak magnitude in $[24000, f_s / 2)$ Hz — i.e.,
  above the 24 kHz anti-alias corner and below the 48 kHz Nyquist. Any
  imaging that survives the anti-alias filter shows here.
- **In-band SNR** (dB): ratio of guarded-tone-band energy to non-tone
  energy in $[0, 20000)$ Hz. This is the demo's principal precision
  metric: it isolates arithmetic noise from filter shape.

The FFT analysis uses a Kaiser $\beta = 18$ window
(sidelobes ~-180 dB) and a $\pm 150$ Hz guard band around each tone,
so the reference SNR is bounded by the resampler's arithmetic noise
floor rather than by analysis-window leakage.

## The mixed-precision matrix

All six configurations use `RationalResampler<double, T, T>` — filter
design is fixed at `double` (Kaiser window and sinc-lowpass need
dynamic range that `fixpnt` doesn't offer), and the streaming
multiply-accumulate loop runs at `StateScalar = SampleScalar = T`.

| Config | `StateScalar / SampleScalar` |
|---|---|
| `reference` | `double` |
| `float`     | `float` |
| `posit32`   | `posit<32,2>` |
| `posit16`   | `posit<16,2>` |
| `cfloat32`  | `cfloat<32,8>` (IEEE-like) |
| `fixpnt32`  | `fixpnt<32,20>` (Q12.20) |

The `fixpnt` choice is worth a note: `RationalResampler` scales its
polyphase taps by $L$ for unity passband gain, so the coefficient
values reach ~160 at the center tap. Q8.24 (`fixpnt<32,24>`,
range $\pm 128$) overflows on that scale factor; Q12.20 (range
$\pm 2048$) leaves comfortable headroom while still delivering
~120 dB fractional resolution.

## Reference results

At `filter_half_length = 20`, Kaiser $\beta = 12$ (~-115 dB filter
stopband):

| Config | SNR (dB) | ripple (dB) | stopband (dB) |
|---|---:|---:|---:|
| `reference` (`double`)     | 82.62 | 0.32 | < -300 |
| `float`                     | 82.62 | 0.32 | < -300 |
| `posit32`                   | 82.62 | 0.32 | < -300 |
| `posit16`                   | 67.98 | 0.32 | < -300 |
| `cfloat32`                  | 82.62 | 0.32 | < -300 |
| `fixpnt32` (`Q12.20`)       | 82.61 | 0.32 | < -300 |

The 32-bit configurations are indistinguishable from `double` — the
SNR ceiling here is set by measurement setup (analysis window +
finite FFT), not by the arithmetic. Only `posit16` shows a clear
precision cost, giving up ~15 dB SNR for its 2× memory footprint
reduction.

The `< -300` stopband floor means the anti-alias filter is completely
attenuating aliased/imaged content: nothing leaks above 24 kHz that
this measurement can see. The 0.32 dB passband ripple comes from the
filter's own Kaiser-window ripple shape, not from precision loss.

## When to reach for this pattern

Reach for `RationalResampler` when both of the following hold:

- **The rate ratio is rational** — every case-of-interest reduces to
  small $L / M$ integers after GCD. If your input/output rates are
  irrational (drift, resample-to-track) you want an arbitrary
  resampler (Farrow / cubic interpolator) instead.
- **The rate ratio doesn't reduce to a small integer** — for pure
  integer rate changes, `PolyphaseInterpolator` or `PolyphaseDecimator`
  alone are simpler and slightly cheaper.

For the reduced $L / M = 160 / 147$ case here, `RationalResampler`
computes only the M/L output samples the polyphase commutator
selects, keeping the arithmetic budget bounded regardless of the ↑L
intermediate rate. See [`conditioning/src.hpp`][src] for the
implementation.

[src]: https://github.com/stillwater-sc/dsp/blob/main/include/sw/dsp/conditioning/src.hpp

## Source

- Application: `applications/multirate_examples/audio_resampler/audio_resampler.cpp`
- Build target: `audio_resampler` (default preset `ci` builds it)
- CSV schema: `pipeline, config_name, scalar_type, metric_kind, tone_hz, value_db`

## Related pages

- [Multirate Overview](./overview/) — the theory behind polyphase decomposition
- [Pattern Catalog](./patterns/) — problem→API mapping for all multirate patterns
- [Acquisition Demo](../acquisition/demo/) — end-to-end mixed-precision receiver chain that combines DDC + polyphase decimation
