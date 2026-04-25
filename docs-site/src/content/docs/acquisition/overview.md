---
title: SDR Receiver Front-End Overview
description: Module overview, pipeline architecture, and per-stage component selection for the SDR / IF-receiver front-end module
---

This module implements the digital receiver front-end of a
software-defined radio (SDR): the chain that takes a real-valued
high-rate ADC stream containing one or more modulated channels at an
intermediate frequency (IF), tunes the channel of interest down to
zero frequency, and decimates to a tractable baseband rate ready for
demodulation.

The module's umbrella header brings everything in:

```cpp
#include <sw/dsp/acquisition/acquisition.hpp>
```

Or include only what you use — every component lives behind a small
focused header (`cic.hpp`, `nco.hpp`, etc.).

## Scope: what this module is and isn't

**This module IS:**
- An SDR / IF-sampling receiver front end
- Multirate decimation primitives (CIC, half-band, polyphase) usable
  in any context that needs efficient rate reduction
- A numerically-controlled oscillator (NCO) and complex mixer for
  digital down-conversion
- A composable `DecimationChain` for cascading the above

**This module is NOT:**
- An instrument-style data-acquisition front end. If you're building a
  digital oscilloscope or a spectrum analyzer, you'll need primitives
  this module doesn't ship: edge/level/slope **trigger logic**,
  **pre/post-trigger ring buffers**, **peak-detect (min/max)
  decimation** for glitch capture, **segmented memory** capture,
  **multi-channel time alignment**, sweep-tuned local oscillators,
  RBW/VBW filters, detector modes (peak/sample/average/quasi-peak),
  trace averaging, and front-end calibration. Those primitives are
  tracked in their own roadmap (search the issue tracker for
  *"instrument data acquisition"*).
- An ADC driver. The samples are assumed to already be in memory.
- A demodulator. The complex baseband output of this chain is the
  *input* to a demodulator (FM, QAM, OFDM, GMSK, ...) which lives
  elsewhere.

The CIC, half-band, and polyphase decimators are the most reusable
piece — they're the same primitives an instrument front end would
use for bulk rate reduction. The NCO + mixer + DDC composition is
what makes this specifically a *receiver* rather than a general
acquisition chain.

## What lives in this module

| Component | Header | Role |
|---|---|---|
| [NCO](./nco/) | `nco.hpp` | Phase-accumulator complex sinusoid generator (the local oscillator) |
| [DDC](./ddc/) | `ddc.hpp` | NCO + complex mixer + first decimator (channel selection) |
| [CIC Decimator](./cic/) | `cic.hpp` | Multiplier-free bulk rate reduction at the highest sample rate |
| [Half-Band Filter](./halfband/) | `halfband.hpp` | Sharp ↓2 stage with ~75% zero taps |
| [Polyphase Decimator](./polyphase-decimator/) | `polyphase_decimator.hpp` | Final channel-shaping FIR decimation |
| [Decimation Chain](./decimation-chain/) | `decimation_chain.hpp` | Variadic-tuple multistage composition |
| [End-to-End Demo](./demo/) | `applications/acquisition_demo` | Full receiver, sweep across number systems |

The matching analysis primitives — SNR, ENOB, NCO SFDR, CIC bit-growth
verification — live in
[`analysis/acquisition_precision.hpp`](../../analysis/acquisition-precision/).

## Receiver architecture

A canonical IF-sampling SDR receiver looks like this:

```text
ADC (10s of MHz, real samples at IF)
   │
   ▼
NCO ──►  complex mixer  ──►  ↓K1  (DDC: tunes a channel down to 0 Hz)
                              │
                              ▼
                            CIC ↓K2     (bulk rate reduction; multiplier-free)
                              │
                              ▼
                       Half-Band ↓2     (sharp transition with free decimation)
                              │
                              ▼
                      Polyphase FIR ↓Kn (final channel shaping at the lowest rate)
                              │
                              ▼
                    I/Q baseband (kHz range, ready for demodulation)
```

The total decimation `K1 × K2 × 2 × Kn` typically ranges from $2^4$ to
$2^{10}$. Each stage targets the bandwidth and arithmetic constraints
of its own input rate, which is what makes the multistage approach an
order of magnitude cheaper than a single front-end FIR at the ADC rate.

The [decimation chain reference](./decimation-chain/) covers the
historical motivation (Hogenauer, Harris, the DDC-ASIC era) and the
math of why this factorization works.

## When to use each component

### Need a complex local oscillator? → [NCO](./nco/)

The phase-accumulator NCO produces $\cos(2\pi f_0 n / f_s) + j\sin(\ldots)$
at any frequency exactly representable as a phase increment. Use it
alone for signal generation, or as the local oscillator for a DDC.
Posit32 typically delivers >150 dB SFDR.

### Need to tune a channel down to baseband? → [DDC](./ddc/)

The DDC composes an NCO, a complex multiplier, and a polyphase
decimator into a single object: input real samples at the IF, output
complex baseband at $f_s / R$. This is the right entry point if you
just want to point at a frequency and get I/Q out.

### Need to cut the rate by 8× to 64× from a high-MHz input? → [CIC](./cic/)

CICs use only adders and subtractors, so they run efficiently at the
highest sample rate (above what FIR multipliers can sustain). They
introduce a $\text{sinc}^M$ passband droop that downstream stages
correct. **Important constraint:** the CIC integrator requires
two's-complement-wrapping state; floating-point and posit
state types accumulate uncorrectable drift on DC bias.

### Need a sharp ↓2 with low arithmetic cost? → [Half-Band](./halfband/)

Half-band FIR filters have ~75% zero coefficients (Mintzer 1982),
so the per-output cost is roughly half a normal FIR of the same
length. They produce a sharp transition at $f_s/4$, which is exactly
where ↓2 needs it. Cascade two or three for ↓4 or ↓8.

### Need the final channel-defining filter? → [Polyphase Decimator](./polyphase-decimator/)

Polyphase decomposition (Bellanger 1976) lets a length-$L$ FIR
decimating by $K$ run at $L/K$ multiplies per *output* sample
instead of per input sample. Use this for the last stage where the
channel-shaping requirements set the filter length.

### Need to compose multiple stages cleanly? → [Decimation Chain](./decimation-chain/)

`DecimationChain<SampleScalar, Stage1, Stage2, ...>` is a variadic
tuple wrapper that chains arbitrary heterogeneous decimator types,
threading samples through each stage. The end-to-end demo uses it to
build the post-DDC `CIC → HalfBand → Polyphase` cascade.

## The three-scalar model in an SDR front end

Every component in this module follows the same `(CoeffScalar,
StateScalar, SampleScalar)` parameterization as the filter and
spectral modules:

| Scalar | Role in this module | Typical choice |
|---|---|---|
| `CoeffScalar` | Filter taps, NCO twiddle constants | `double`, `posit<32,2>` |
| `StateScalar` | Accumulator state inside each stage | `double`, `fixpnt<32,28>` |
| `SampleScalar` | Inter-stage I/Q stream | `float`, `posit<16,1>`, `fixpnt<16,12>` |

Where this module differs from filter design is that **the scalar
choice is not uniform across the chain**. The CIC's accumulator has
to absorb $M \lceil \log_2(RD) \rceil$ bits of growth and (per the
constraint above) wants two's-complement wrap, so `fixpnt` is the
natural fit. The half-band and polyphase stages run at lower rates
on signals that already fit in a smaller envelope, so they can use
narrower types — possibly even narrower than the CIC's input
samples.

Picking the per-stage type is itself a design exercise. The
[end-to-end demo](./demo/) walks through three concrete decisions
(normalized rates, Q-format selection for fixpnt, and CIC state
precision) that come up the moment you try to instantiate the
pipeline at non-IEEE precision.

For empirical SNR/ENOB measurements across the chain, use the
[acquisition precision analysis](../../analysis/acquisition-precision/)
primitives. They reuse identifier columns from the
[`precision_sweep`](https://github.com/stillwater-sc/mixed-precision-dsp/tree/main/applications/precision_sweep)
CSV schema, so the same Python tools that visualize IIR sweeps work
on receiver-chain sweeps.

## Worked-example reference numbers

The library's regression tests measure these end-to-end SNR floors
for a 3-stage CIC → half-band → polyphase chain:

| Pipeline scalar (uniform) | Measured SNR | Measured ENOB |
|---|---|---|
| `double` | ref (clipped at 300 dB) | ref |
| `posit<32, 2>` | ~98 dB | ~16 |
| `fixpnt<32, 28>` (Q4.28) | ~153 dB | ~25 |
| `float` (IEEE) | ~54 dB | ~9 |
| `posit<16, 2>` | -30 dB (collapse) | — |

The `float` and `posit<32,2>` numbers look surprisingly low until you
realize they're not measuring arithmetic precision — they're
measuring CIC integrator drift on the DC component of the
post-mixer signal. `fixpnt` wins by ~100 dB because it provides the
two's-complement wrap that the CIC algorithm structurally requires.
The [end-to-end demo](./demo/) walks through this measurement and
the CIC reference covers the underlying math.

## See also

- [Filter Design Overview](../../filter/overview/) — the three-scalar
  model and FIR/IIR pipelines this module builds on
- [DFT and FFT](../../spectral/dft-fft/) — what you typically run
  *after* the receiver chain has reduced the rate
- [Acquisition Precision Analysis](../../analysis/acquisition-precision/)
  — measurement primitives (SNR, ENOB, NCO SFDR, CIC bit-growth)
- Per-component pages: [NCO](./nco/), [DDC](./ddc/),
  [CIC](./cic/), [Half-Band](./halfband/),
  [Polyphase Decimator](./polyphase-decimator/),
  [Decimation Chain](./decimation-chain/), [Demo](./demo/)
