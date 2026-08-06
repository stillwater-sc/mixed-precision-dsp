---
title: SDR Precision Analysis
description: Per-block EVM attribution, the implementation-loss metric, the full number-system sweep, and the measured result that inverts the usual expectation about tapered precision
---

```cpp
#include <sw/dsp/analysis/sdr_precision.hpp>   // run_link, analyze_blocks,
                                               // evm_budget, CSV writers
```

The per-module precision sweeps elsewhere in this library answer *"what
does this block cost?"*. This one answers the question a system
designer actually has: **given a link, which block should get the wider
arithmetic, and how narrow can the rest go before the constellation
stops closing.**

Two tools cover it, and the difference between them is important:

| | `analysis/sdr_precision.hpp` | [`applications/sdr_demo`](https://github.com/stillwater-sc/mixed-precision-dsp/tree/main/applications/sdr_demo) |
|---|---|---|
| What runs narrow | Coefficients are **projected** into the narrow type; the arithmetic runs in `double` | `PolyphaseInterpolator<Coeff,State,Sample>` and `FIRFilter<Coeff,State,Sample>` are **instantiated** on the type: coefficients, accumulators and samples all run narrow |
| Measures | Storage precision | Arithmetic precision |
| Good for | Attribution — isolating one block cleanly | Cost — what a real narrow implementation delivers |

Both views appear in the demo's report, clearly labelled. Neither
subsumes the other: projection tells you where the error *comes from*,
real narrow arithmetic tells you what it *costs*.

## The chain, and what it deliberately excludes

```text
bits → constellation map → RRC pulse shaping → AWGN
     → RRC matched filter → symbol sampling → demap → bits
```

**The timing and carrier loops are not in it.** Those converge
stochastically, so their residual would land on top of the arithmetic
residual being measured and there would be no way to tell the two
apart. Their precision behaviour is characterized in
[their own tests](./synchronization/), where the loop is the subject
rather than the noise floor. What remains here is the path whose error
is purely arithmetic — which is the only path an attribution can be
computed on.

## Per-block attribution

```cpp
using namespace sw::dsp::analysis;

SdrLinkConfig cfg;
cfg.modulation = Modulation::qam16;
cfg.ebn0_db    = 30.0;

auto rows = analyze_blocks<sw::universal::posit<16,2>>(cfg, "posit<16,2>", 16);
write_sdr_precision_csv("blocks.csv", rows);
```

`SdrBlock` names what runs narrow: `none` (the all-double reference),
`constellation`, `tx_shaping`, `rx_matched`, `whole_chain`.

Two design decisions make the result meaningful, and both are easy to
get wrong:

**One block at a time, everything else at `double`.** Measuring blocks
jointly gives a number that cannot be attributed; narrowing everything
at once gives a total that hides which block spent it. Only the
one-at-a-time form composes into a statement about a *block*.

**The reference is subtracted in power**, not compared raw:

$$
\mathrm{EVM}_{\mathrm{contribution}} =
\sqrt{\max\!\left(0,\;
\mathrm{EVM}_{\mathrm{narrow}}^2 - \mathrm{EVM}_{\mathrm{ref}}^2\right)}
$$

Every configuration carries a **common floor** — the pulse shaping's
truncation ISI, plus whatever channel noise the config asks for — and
that floor is routinely *larger* than the arithmetic under test. At
$E_b/N_0 = 30$ dB on a 10-symbol RRC span the floor measures
**1.07e-02** while `posit<16,2>`'s whole contribution sits below
**1e-04**: the raw EVMs agree to four digits and say nothing at all.
Removing the floor in power is what makes the remainder visible.

### The contributions do not add

Measured on 16-QAM:

| Type | Sum of parts | Measured `whole_chain` | Direction |
|---|---|---|---|
| `posit<16,2>` | 2.97e-04 | 3.41e-04 | more than the sum |
| `posit<8,2>` | 1.37e-02 | 9.47e-03 | less than the sum |

At 16 bits the errors are small and independent enough that narrowing
everything compounds slightly beyond the sum. At 8 bits they are large
enough to partly cancel. **A breakdown is an attribution, not a
budget**: it says where the error comes from, not what narrowing
everything will cost. Predict that by measuring `whole_chain` — which
is why `whole_chain` is measured and never summed.

### Reading a real breakdown

From a default `sdr_demo` run, 16-QAM:

| Type | Block | EVM | Contribution |
|---|---|---|---|
| `double` | reference | 7.26e-03 | 0 |
| `posit<16,2>` | constellation | 7.26e-03 | 1.35e-04 |
| `posit<16,2>` | tx_shaping | 7.26e-03 | 1.87e-04 |
| `posit<16,2>` | rx_matched | 7.26e-03 | 1.87e-04 |
| `posit<16,2>` | whole_chain | 7.26e-03 | 3.41e-04 |
| `posit<8,2>` | constellation | 1.37e-02 | **1.17e-02** |
| `posit<8,2>` | tx_shaping | 8.88e-03 | 5.13e-03 |
| `posit<8,2>` | rx_matched | 8.88e-03 | 5.13e-03 |
| `posit<8,2>` | whole_chain | 1.19e-02 | 9.47e-03 |
| `cfloat<8,4>` | constellation | 1.37e-02 | **1.17e-02** |
| `cfloat<8,4>` | tx_shaping | 8.63e-03 | 4.67e-03 |
| `cfloat<8,4>` | rx_matched | 8.63e-03 | 4.67e-03 |
| `cfloat<8,4>` | whole_chain | 9.17e-03 | 5.61e-03 |

The **constellation table is the single largest contributor at 8
bits**, larger than either filter. That is not obvious in advance — the
table is a handful of constants and the filters do thousands of
multiply-accumulates — but the table's error is *systematic*: every
symbol carrying that label lands at the same wrong place, so the errors
do not average down the way filter rounding does.

Note also that `posit<8,2>` and `cfloat<8,4>` give **exactly the same
1.17e-02** constellation contribution. Two very different formats
producing identical numbers points at a shared bottleneck rather than
at each format's own quantization; that observation is what
[issue #209](https://github.com/stillwater-sc/mixed-precision-dsp/issues/209)
exists to resolve.

**A contribution of 0 is a resolution limit, not an exact result.** At
16 bits the arithmetic sits an order of magnitude below the truncation
floor and the power subtraction has little left to resolve. The 8-bit
rows, which clear the floor, are where the breakdown has margin.

## Implementation loss: the usability metric

Raw EVM is not the right axis for a go/no-go decision, because it mixes
the arithmetic under test with the channel noise the link was going to
have anyway. The demo therefore reports **implementation loss**:

$$
L_{\mathrm{dB}} = 10\log_{10}
\frac{\mathrm{EVM}_{\mathrm{narrow}}^2}{\mathrm{EVM}_{\mathrm{double}}^2}
$$

measured at the $E_b/N_0$ that puts the **`double` chain** at BER
$10^{-3}$, with **1 dB** the usability threshold. That is the number a
link budget has a slot for: how much extra SNR this arithmetic costs.

Two details make it reliable:

- The reference is re-measured at the sweep's **own symbol count and
  seed**, so the narrow chain sees the *same noise realization* and most
  of the sampling variance cancels in the ratio.
- The operating point is found by bisection on
  `theoretical_ber_awgn`, so it moves with the modulation rather than
  being a fixed SNR that means different things at different orders.

`evm_budget(m)` gives the complementary ceiling — half the distance
from a constellation point to its nearest neighbour, the error that
would put a *noiseless* symbol exactly on a decision boundary:

| Modulation | EVM budget |
|---|---|
| QPSK | 0.7071 |
| 16-QAM | 0.3162 |
| 64-QAM | 0.1543 |
| 256-QAM | 0.0767 |

It is a ceiling, not a target. A practical link needs a wide margin
below it, which is what the 1 dB implementation-loss criterion supplies.

## The sweep

```bash
cmake --preset ci && cmake --build build-ci -j4
./build-ci/applications/sdr_demo/sdr_demo
```

15 number-system configurations × 4 modulations (QPSK, 16-QAM, 64-QAM,
256-QAM), across IEEE, cfloat, posit and fixpnt at 8, 12, 16 and 32
bits. Five CSVs: the sweep, a Pareto frontier (bits of arithmetic
against delivered bits/symbol), constellation clouds for plotting, the
per-block breakdown, and the input-backoff table. Runs in ~40 s at
defaults.

### Highest usable modulation, by family and width

| Family | 8-bit | 12-bit | 16-bit | 32-bit |
|---|---|---|---|---|
| IEEE | — | — | — | 256-QAM |
| cfloat | QPSK | 256-QAM | 256-QAM | 256-QAM |
| posit | QPSK | 256-QAM | 256-QAM | 256-QAM |
| fixpnt | **16-QAM** | 256-QAM | 256-QAM | 256-QAM |

256-QAM is the top of this sweep, not a ceiling of any format.
`cfloat<32,8>` and `cfloat<16,5>` *are* binary32 and binary16, so the
cfloat row continues the IEEE one below 32 bits.

## The result that inverts the expectation

The SDR epic began from the premise that **posit reaches a higher
modulation order than its rivals at equal bit width**. At full scale,
the measurement says the opposite. 16-QAM at 8 bits, at the BER-$10^{-3}$
operating point:

| Config | EVM arith | Implementation loss | Usable |
|---|---|---|---|
| `fixpnt<8,5>` | 6.66e-02 | **0.81 dB** | **yes** |
| `posit<8,2>` | 9.08e-02 | 1.27 dB | no |
| `cfloat<8,4>` | 9.22e-02 | 1.27 dB | no |
| `posit<8,0>` | 9.02e-02 | 1.46 dB | no |

Fixed-point carries 16-QAM at 8 bits where posit and cfloat carry only
QPSK.

**Why, stated plainly.** The link is amplitude-normalized end to end:
unit-average-power constellation, unit-energy RRC, receiver gain
restored before measurement. The whole waveform therefore lives inside
roughly **one octave** — and EVM is an *absolute* error metric. Under
those two conditions a uniform absolute step is optimal by
construction, and a tapered format spends bits on dynamic range the
signal never uses. Posit's tapered precision buys nothing when nothing
needs the dynamic range.

This is not a disappointing result to be explained away. It is a
sharper statement of when each format wins, and it identifies the
condition — amplitude normalization — that the original premise
silently assumed away.

## Where the tapered format does win: dynamic range

The demo therefore also measures the axis the families genuinely differ
on. The **input-backoff sweep** attenuates signal *and* noise together,
so $E_b/N_0$ is unchanged, and restores the level before measuring EVM.
Every column is the same link; only where the waveform sits inside the
number format changes. 16-QAM, quiet channel:

| Number system | bits | 0 dB | −12 dB | −24 dB | −36 dB | −48 dB |
|---|---|---|---|---|---|---|
| `double` | 64 | 7.26e-03 | 7.26e-03 | 7.26e-03 | 7.26e-03 | 7.26e-03 |
| `cfloat<16,5>` | 16 | 7.31e-03 | 7.31e-03 | 7.31e-03 | 7.29e-03 | 7.30e-03 |
| `posit<16,2>` | 16 | 7.26e-03 | 7.26e-03 | 7.31e-03 | 7.31e-03 | 7.41e-03 |
| `fixpnt<16,13>` | 16 | 7.31e-03 | 7.41e-03 | 8.33e-03 | 2.02e-02 | 6.43e-02 |
| `cfloat<8,4>` | 8 | 9.25e-02 | 9.11e-02 | 1.03e-01 | 2.11e-01 | 4.06e-01 |
| `posit<8,2>` | 8 | 9.11e-02 | 8.87e-02 | 1.26e-01 | 1.23e-01 | 2.19e-01 |
| `fixpnt<8,5>` | 8 | **6.70e-02** | 1.94e-01 | 4.06e-01 | **1.00e+00** | **1.00e+00** |

`fixpnt<8,5>` is the **best** format at full scale and **completely
lost** by −36 dB. `posit<8,2>` holds. Expressed as the backoff each
format tolerates before its EVM doubles:

| Format | Usable backoff range |
|---|---|
| `double` | 48 dB |
| `cfloat<16,5>` | 48 dB |
| `posit<16,2>` | 48 dB |
| `posit<8,2>` | 36 dB |
| `cfloat<8,4>` | 24 dB |
| `fixpnt<16,13>` | 24 dB |
| `fixpnt<8,5>` | **0 dB** |

**Dynamic range, not precision at full scale, is what separates the
number systems in a digital link.** A normalized full-scale link is the
best case for fixed-point and the worst case for posit; move the signal
off full scale — an un-settled AGC, a weak carrier, a system sized for
the strong signal, or an [OFDM waveform backed off for its
PAPR](./ofdm/#peak-to-average-power-ratio) — and the uniform grid runs
out while the tapered one keeps its relative precision.

The demo derives this conclusion **from its own data at run time**
rather than printing a stored claim, so changing the sweep parameters
changes the finding instead of leaving a stale statement in the output.

## Restated design guidance

> In an amplitude-normalized modem the arithmetic format's advantage is
> not precision at full scale — a uniform grid wins there — but the
> **range of input levels over which the modem holds its EVM**.

Practically:

- **If your signal is reliably at full scale** and you control the
  scaling, fixed-point at 8 bits is competitive and will beat an 8-bit
  float or posit. Verify the binary point against the measured peak;
  the demo prints a `clip%` column for exactly that check.
- **If your signal level is uncertain** — AGC still settling, variable
  path loss, high-PAPR waveform — the tapered formats hold EVM across
  24 to 48 dB more input range at the same width, and that is worth
  more than the fraction of a dB fixed-point wins at full scale.
- **At 12 bits and above the question mostly disappears** for a
  single-carrier link: every family carries 256-QAM, and the choice
  should be made on the loops and the transforms instead — where, as
  the [channelizer](./channelizer/#precision-behaviour) and
  [OFDM](./ofdm/#fft-precision-and-subcarrier-orthogonality) pages show,
  `posit<32,2>` is 14–22 dB better than `float` at equal width.

## An open question

The 8-bit ordering above is **measured but not yet mechanistically
explained**, and two things in the data are unresolved:

- `posit<8,2>` and `cfloat<8,4>` land on the same 1.27 dB to three
  digits, and on the same 1.17e-02 constellation contribution. Very
  different formats producing identical numbers suggests a shared
  bottleneck rather than each format's own quantization.
- `fixpnt<8,5>` gets a hand-picked binary point; posit and cfloat have
  no equivalent tuning knob. The measured waveform peak is ~0.8 with
  `clip%` at 0.00, so its two integer bits are not earning their keep,
  and `fixpnt<8,6>` may be better still.

[Issue #209](https://github.com/stillwater-sc/mixed-precision-dsp/issues/209)
tracks a Q-point and `es` sweep across the whole 8-bit design space, a
three-scalar factorial to localize the loss to `CoeffScalar` /
`StateScalar` / `SampleScalar`, and loss-versus-$E_b/N_0$ curves.
Treat the 8-bit row as a measurement awaiting a mechanism, not a
settled property of the number systems.

## CSV schema

`write_sdr_precision_csv` emits one row per measurement, schema-compatible
at the identifier columns with `precision_sweep.csv` and
`acquisition_demo.csv`, so the same Python tooling reads all three:

```text
pipeline,block,scalar_type,bit_width,modulation,ebn0_db,
evm_rms,evm_db,mer_db,evm_contribution,bit_errors,total_bits,ber
```

`write_constellation_csv` emits received symbols one per row
(`block,scalar_type,modulation,i,q`) for constellation plots.

## See also

- [SDR Overview](./overview/) — the link these measurements run on
- [Constellation and Metrics](./constellation/) — EVM normalization,
  the BER curves the operating point is derived from, and the RRC
  truncation floor that is subtracted out here
- [Synchronization](./synchronization/) — the loops deliberately
  excluded from this chain, and why loop precision behaves completely
  differently
- [OFDM](./ofdm/) and [Channelizer](./channelizer/) — the transform
  measurements that complete the picture
- [Analysis Overview](../../analysis/overview/) and
  [Acquisition Precision](../../analysis/acquisition-precision/) — the
  sibling measurement modules
