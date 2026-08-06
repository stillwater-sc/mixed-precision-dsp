---
title: SDR Modulation & Demodulation Overview
description: Module overview, transmit/receive chain architecture, modulation types, and the three-scalar model applied to a digital link
---

This module implements the **modulation and demodulation** half of a
software-defined radio: everything between a bit stream and a complex
baseband waveform. It is the natural counterpart to the
[SDR receiver front-end](../../acquisition/overview/), which handles
the IF-to-baseband path (NCO, mixer, decimation) but deliberately stops
short of demodulation.

The module's umbrella header brings everything in:

```cpp
#include <sw/dsp/sdr/sdr.hpp>
```

Or include only what you use — every component lives behind a small
focused header (`constellation.hpp`, `rrc.hpp`, `agc.hpp`, ...).

## Scope: what this module is and isn't

**This module IS:**

- A constellation mapper and demapper (BPSK through 256-QAM) with Gray
  labelling, hard decisions and log-likelihood ratios
- Root-raised-cosine pulse shaping and the matched-filter design that
  goes with it
- Link-quality measurement: EVM, MER, BER, and structured I/Q
  impairment recovery
- The three synchronization loops a coherent receiver needs — AGC,
  symbol timing, carrier frequency and phase
- OFDM modulation and demodulation with pilot-based channel estimation
- A synthesis channelizer, completing the analysis bank from v0.8

**This module is NOT:**

- A forward-error-correction layer. The demapper produces LLRs in the
  form an LDPC or turbo decoder wants, but no decoder ships here.
- A frame synchronizer. Every loop in this module leaves a phase or
  index **ambiguity** that only a known preamble or differential
  encoding can resolve, and neither is provided.
- A channel model beyond AWGN and the static multipath the OFDM
  equalizer is tested against. No fading, no Doppler.
- An RF front end. Samples arrive already at complex baseband — that
  is what the [acquisition module](../../acquisition/overview/) is for.

## What lives in this module

| Component | Header | Role |
|---|---|---|
| [Constellation](./constellation/) | `constellation.hpp` | Gray-coded mapper and demapper, BPSK → 256-QAM |
| [Metrics](./constellation/#measuring-the-link) | `metrics.hpp` | EVM, MER, BER, theoretical AWGN curves, I/Q imbalance |
| [RRC pulse shaping](./constellation/#pulse-shaping) | `rrc.hpp` | Root-raised-cosine and raised-cosine designers, ISI measurement |
| [AGC](./synchronization/#automatic-gain-control) | `agc.hpp` | Log-domain gain loop with attack/decay asymmetry |
| [Timing recovery](./synchronization/#symbol-timing-recovery) | `timing_recovery.hpp` | Gardner / Mueller-Muller with a Farrow interpolator |
| [Carrier recovery](./synchronization/#carrier-recovery) | `carrier_recovery.hpp` | Costas loop with modulation-stripped AFC |
| [Loop filter](./synchronization/#the-shared-loop-filter) | `loop_filter.hpp` | The PI filter both loops share |
| [OFDM](./ofdm/) | `ofdm.hpp` | Modulator, demodulator, layout, channel estimation, PAPR |
| [Channelizer](./channelizer/) | `channelizer.hpp` | Oversampled analysis/synthesis pair with perfect reconstruction |
| [Precision analysis](./precision/) | `analysis/sdr_precision.hpp` | Per-block EVM attribution across number systems |

## Link architecture

A complete coherent link, transmitter through receiver:

```text
TRANSMITTER
   bits
     │
     ▼
   Constellation::map          ── Gray-coded, unit average power
     │
     ▼
   PolyphaseInterpolator ↑sps  ── RRC pulse shaping (rrc_filter taps)
     │
     ▼
   complex baseband waveform
     │
     ▼   (to the DUC / DAC — outside this module)

CHANNEL:  AWGN, and whatever the RF path adds — gain, frequency
          offset, phase offset, sample-clock error

RECEIVER
   complex baseband waveform    ── (from the DDC — the acquisition module)
     │
     ▼
   AutomaticGainControl        ── level the constellation for the demapper
     │
     ▼
   FIRFilter (matched)         ── the same RRC taps as the transmitter
     │
     ▼
   TimingRecovery              ── find the sampling instant, track clock drift
     │
     ▼
   CarrierRecovery             ── de-rotate: remove frequency and phase offset
     │
     ▼
   Constellation::demap_hard   ── or demap_llr_maxlog for a soft-decision FEC
     │
     ▼
   bits
```

**The order of the receive blocks is not arbitrary.** Each one depends
on what the block above it has already removed:

- The **AGC comes first** because both timing detectors form a product
  of two sample values, so their slope $K_p$ scales with signal power.
  A timing loop behind an un-settled AGC has a loop bandwidth that
  changes as the AGC converges.
- **Timing precedes carrier** when the Gardner detector is used, because
  Gardner is blind to carrier phase — it works on the signal's envelope
  transitions and does not care that the constellation is spinning.
- **Mueller-Muller reverses that dependency.** It is decision-directed,
  so it needs a roughly correct constellation and therefore belongs
  *after* carrier recovery. Choosing a detector is choosing an order.
- The **decision-directed carrier detector** likewise needs the loop
  to be close already; `CarrierDetector::qpsk` acquires, and the
  decision-directed form refines for anything denser than QPSK.

## Modulation types

| Scheme | `Modulation` | bits/symbol | $M$ | Geometry |
|---|---|---|---|---|
| BPSK | `bpsk` | 1 | 2 | 2-PSK, points $\pm 1$ |
| QPSK | `qpsk` | 2 | 4 | 4-PSK offset $\pi/4$ → $(\pm 1 \pm j)/\sqrt{2}$ |
| 8-PSK | `psk8` | 3 | 8 | 8-PSK, first point on the real axis |
| 16-QAM | `qam16` | 4 | 16 | square, 4-PAM per axis |
| 64-QAM | `qam64` | 6 | 64 | square, 8-PAM per axis |
| 256-QAM | `qam256` | 8 | 256 | square, 16-PAM per axis |

Every table is scaled to **unit average power**, $E[|s|^2] = 1$ over
equiprobable symbols, so a noise variance or an $E_b/N_0$ means the same
thing across schemes. The [constellation page](./constellation/) covers
the Gray construction and the normalization arithmetic.

## The three-scalar model in a digital link

Components in this module follow the library's
`(CoeffScalar, StateScalar, SampleScalar)` parameterization, but two of
them take a **two-scalar** form instead, and the reason is worth stating
because it is a design signal rather than an inconsistency:

| Component | Parameterization | Why |
|---|---|---|
| `Constellation<T>` | one scalar | A table of constants. There is no state and no separate coefficient set. |
| `rrc_filter<T>` | one scalar | A design function. It returns taps; the filtering is done by the multirate primitives, which take all three. |
| `AutomaticGainControl<StateScalar, SampleScalar>` | two | A gain and a level are real quantities whatever they multiply, and there are **no coefficients** — the loop constants are derived from time constants at construction. |
| `TimingRecovery<StateScalar, SampleScalar>` | two | Same: the Farrow interpolator's weights are computed per sample from $\mu$, so there is no stored coefficient set to carry its own precision. |
| `CarrierRecovery<StateScalar, SampleScalar>` | two | Same. |
| `OversampledChannelizer<Coeff, State, Sample>` | three | A real filter bank: the prototype is a stored tap set. |

Where a link differs most from a filter is that **the loops fail
differently from the datapath**. A datapath block loses precision
gracefully — the EVM rises smoothly as bits come off. A feedback loop
has a threshold: below some resolution the integrator's per-step
increment falls under the state's ULP and the loop stops moving
entirely, with no intermediate degradation. Two concrete cases, both
found by measurement and both fixed by changing what the state *holds*
rather than how wide it is:

- The timing loop's symbol clock was an absolute time accumulator.
  At 2 samples/symbol it reaches 16000 after 8000 symbols, where
  `posit<16,2>`'s ULP is about 8 — a step of ~2 cannot advance it at
  all, and the loop froze with the eye shut. Held as a position
  *relative* to the newest sample it stays in $[-1, 2]$ and every
  supported type has resolution to spare.
- The carrier NCO's phase has the same shape. It is **wrapped to
  $[-\pi, \pi)$ every sample**, not allowed to run as an unbounded sum.

The general rule this module contributes to the library: **an
unbounded accumulator is the recurring precision trap.** Keep loop
state relative or wrapped, and hold a *deviation from nominal* rather
than an absolute value — which is exactly what
[`PiLoopFilter`](./synchronization/#the-shared-loop-filter) enforces
for both loops.

## Reference numbers

Measured by the module's own tests at the current revision; reproduce
by running the binaries at this commit, since implementation changes
can shift them.

| Measurement | `double` | `posit<32,2>` | `float` / `cfloat<32,8>` | `posit<16,2>` |
|---|---|---|---|---|
| RRC peak ISI, span 256 (quantization-limited) | 3.5e-06 | 3.5e-06 | — | 8.2e-05 |
| Channelizer reconstruction error, $M=16$ | 1.5e-16 | 5.6e-09 | 7.9e-08 | 3.1e-04 |
| OFDM intercarrier interference, 64-QAM ideal channel | 3.4e-16 | 1.2e-08 | 1.6e-07 | 4.0e-04 |
| Timing jitter, $B_nT = 0.01$, 200 ppm clock error | 2.361e-01 | 2.361e-01 | 2.361e-01 | 2.363e-01 |
| Carrier residual EVM, $B_nT = 0.02$, $\Delta f = 0.03$ | 4.2806% | 4.2806% | 4.2806% | 4.2799% |

The pattern is worth reading carefully. The **datapath** rows (RRC,
channelizer, OFDM) separate the number systems by orders of magnitude,
because they measure arithmetic error directly. The **loop** rows do
not separate them at all — every type down to `posit<16,2>` produces
the same jitter and the same residual EVM, because those quantities
are set by the loop's *bandwidth* and the detector's *self-noise*, not
by its arithmetic. A synchronization loop is precision-insensitive
right up until it is catastrophically precision-sensitive, and the
transition is the ULP threshold described above rather than a gradual
slope.

For the systematic sweep across number systems and modulation orders —
including a result that inverts the expectation this epic started with
— see [SDR precision analysis](./precision/).

## See also

- [SDR Receiver Front-End](../../acquisition/overview/) — the IF-to-baseband
  path this module's receiver sits behind: NCO, mixer, CIC, half-band,
  polyphase decimation
- [Multirate Signal Processing](../../multirate/overview/) — the
  polyphase interpolator and decimator the pulse shaping is built on,
  and the [analysis channelizer](../../multirate/channelizer/)
- [DFT and FFT](../../spectral/dft-fft/) — the transform OFDM and the
  channelizer both run on
- [Filter Design Overview](../../filter/overview/) — the three-scalar
  model and the FIR machinery the matched filter uses
- [Analysis Overview](../../analysis/overview/) — the measurement
  primitives, including
  [acquisition precision](../../analysis/acquisition-precision/)
- Per-component pages: [Constellation](./constellation/),
  [Synchronization](./synchronization/), [OFDM](./ofdm/),
  [Channelizer](./channelizer/), [Precision](./precision/)
