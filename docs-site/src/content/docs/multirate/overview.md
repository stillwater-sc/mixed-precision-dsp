---
title: Multirate Signal Processing Overview
description: The library's multirate primitives — decimation, interpolation, rational rate change, and filter banks — with the historical context and pattern-to-API mapping
---

Multirate DSP is the body of theory that lets a digital system process
samples at *different* rates inside the same pipeline — decimation,
interpolation, rational rate change, channelization. The library
ships the primitives; this section is the conceptual layer that
connects them.

If you have a concrete problem in mind ("I need to convert 44.1 kHz
to 48 kHz audio" or "what's the right pattern for a channelizer?"),
jump straight to the [Pattern Catalog](./patterns/) — it maps each
canonical multirate problem to the library's API. This page covers
the *why* behind those mappings.

## What "multirate" means

Single-rate DSP runs every block of the pipeline at the same sample
rate. Multirate DSP changes the sample rate between blocks — usually
to match the bandwidth requirements of each stage. The two atomic
operations are:

- **Decimation by M** — keep every Mth sample, throwing away the
  intermediate ones. Output rate = input rate / M.
- **Interpolation by L** — insert L-1 zero samples between each
  input sample, then filter to remove the resulting spectral images.
  Output rate = input rate × L.

Both must be paired with an anti-alias / anti-image filter to avoid
spectral folding. Decimation needs a lowpass with cutoff at
$f_s / (2M)$ before downsampling; interpolation needs the same
lowpass after the upsampling, run at the *higher* rate.

A naive implementation runs the filter at the high rate and then
discards (decimation) or runs it after zero-stuffing (interpolation).
Both waste compute on samples that don't matter. The whole point of
multirate theory is to avoid that waste.

## The historical progression

Multirate DSP came together from a sequence of independent insights,
each driven by a hardware constraint of its era:

### 1960s–70s: the resampling problem

The earliest multirate work was driven by audio applications: digital
audio systems needed to convert between 44.1 kHz, 48 kHz, and the
various intermediate rates that hardware happened to use. The basic
upsample/filter/downsample structure was understood, but the
arithmetic cost was prohibitive on the hardware of the day.

### 1976: Bellanger and the polyphase decomposition

Maurice Bellanger's 1976 paper *"Digital Filtering by Polyphase
Network: Application to Sample-Rate Alteration and Filter Banks"*
showed that an L-band decimating FIR filter can be decomposed into L
sub-filters, each operating at the *output* rate, with a commutator
selecting which sub-filter contributes to each output sample. The
arithmetic cost drops by a factor of L because no multiplications
are wasted on samples that will be discarded.

The same decomposition runs in reverse for interpolation, and
extends naturally to filter banks (see channelization below).
Polyphase is the foundational structural insight of multirate DSP.

### 1981: Hogenauer's CIC

Eugene Hogenauer's 1981 paper *"An Economical Class of Digital
Filters for Decimation and Interpolation"* tackled a different
constraint: at GHz input rates, **any** filter that needed
multipliers was too expensive. His Cascaded Integrator-Comb (CIC)
structure uses only adders and subtractors and runs efficiently at
the highest sample rate. The cost is a non-flat passband response
(a $\text{sinc}^M$ droop) that downstream stages have to compensate.

CIC made the modern digital receiver chain possible: a CIC handles
the front-end bulk decimation at the ADC rate, and cheaper-per-sample
multiplier-based filters take over at lower rates.

### 1982: Mintzer and the half-band filter

Fred Mintzer's 1982 paper *"On Half-Band, Third-Band, and Nth-Band
FIR Filters and Their Design"* characterized a class of FIR filters
where roughly $1 - 1/M$ of the coefficients are exactly zero. For
$M = 2$ — the half-band filter — that's ~75% zero taps, so a length-N
half-band runs at roughly the cost of a length-$(N/2)$ generic FIR.
Half-bands are the workhorse 2:1 stage between a CIC and a final
shaping FIR.

Vaidyanathan's 1993 textbook *Multirate Systems and Filter Banks*
formalized the M-th band design space and connected half-bands to
the broader theory of perfect-reconstruction filter banks.

### 1990s: DDC ASICs

Once polyphase + CIC + half-bands were on the table, dedicated
silicon followed. The Harris HSP50016, Graychip GC4016, and Analog
Devices AD6620 *Digital Down-Converter* chips of the early 1990s
implemented the canonical receiver chain — NCO + complex mixer +
CIC + half-bands + polyphase FIR — in fixed-function hardware. Fred
Harris's 2004 textbook *Multirate Signal Processing for Communication
Systems* documented and generalized what those ASIC designers had
been doing.

### 2000s: SDR and software-only chains

USRP hardware and the GNU Radio framework (early 2000s) moved the
DDC chain into software. The same polyphase / CIC / half-band
primitives now ran on general-purpose CPUs and GPUs. The rate-
reduction stack remained essentially unchanged from the ASIC era;
only the implementation moved.

### Today: mixed-precision multirate

The library you're reading docs for is in this era. Each multirate
primitive is parameterized on three independent scalar types:

- `CoeffScalar` — filter coefficients (design precision)
- `StateScalar` — accumulator state (processing precision)
- `SampleScalar` — input/output samples (streaming precision)

The compelling cases for non-IEEE types in multirate work:

- **CIC integrators** structurally require two's-complement-wrapping
  state — `fixpnt` works, `float`/`posit` accumulate uncorrectable
  drift on DC bias. The [SDR demo](../../acquisition/demo/) measures
  this empirically.
- **Posits** concentrate precision near unity, which is exactly where
  baseband signals live after a DDC mix. Across the band, posit
  delivers more usable precision per bit than IEEE float.
- **Half-band and polyphase** stages run at lower rates on
  bandwidth-reduced signals, so they tolerate narrower types than
  the front-end CIC.

## The Noble identity

The Noble identity is the algebraic fact that lets all the
multirate optimizations work. Stated for decimation:

$$
\text{Filter } H(z) \text{ followed by ↓M} \;\equiv\; ↓M \text{ followed by filter } H(z^M)
$$

The *picture* is: instead of running a filter at the high rate and
then throwing samples away, you can throw the samples away first and
run an "expanded" filter at the low rate. The expanded filter
$H(z^M)$ has zeros stuffed between its taps; combined with polyphase
decomposition (next section), this becomes a filter that runs only
at the output rate.

The dual identity for interpolation:

$$
↑L \text{ followed by filter } H(z) \;\equiv\; \text{Filter } H(z^L) \text{ followed by ↑L}
$$

is what makes polyphase interpolation efficient: the filter operates
at the *input* rate (before upsampling), and the upsampling is just
a commutator that picks which sub-filter's output goes where.

Combined, the Noble identities are the reason every multirate
primitive in this library runs at $O(N)$ multiplies per *output*
sample rather than $O(N \cdot \text{rate change})$.

## The polyphase decomposition theorem

Given a length-$N$ FIR filter $H(z) = \sum_{n=0}^{N-1} h[n] z^{-n}$
and an integer $M$, define the polyphase components

$$
E_k(z) = \sum_{n} h[nM + k] z^{-n}, \quad k = 0, 1, \ldots, M-1
$$

Then

$$
H(z) = \sum_{k=0}^{M-1} z^{-k} E_k(z^M)
$$

In words: $H(z)$ factors into $M$ sub-filters, each addressing every
$M$th tap of the original.

Combine that factorization with the Noble identity for decimation
and you get the canonical polyphase decimator: the input commutates
through the $M$ sub-filters round-robin, each running at the output
rate. The total multiplier budget is $N$ per output sample (vs.
$N \cdot M$ for filter-then-downsample). The library's
[`PolyphaseDecimator`](../../acquisition/polyphase-decimator/) is
exactly this construction.

For interpolation, the dual: $L$ sub-filters each run at the input
rate, and a commutator at the output picks which sub-filter's output
to emit. `PolyphaseInterpolator` is this version.

For rational $L/M$ rate change, both are combined: $L$ sub-filters
at the slow rate, with a time-register that advances by $M$ per
output and wraps modulo $L$. The library's `RationalResampler` (in
`sw/dsp/conditioning/src.hpp`) implements this directly.

### Channelization (filter banks)

Bellanger's 1976 paper also showed that an $M$-channel uniform
filter bank — splitting a wideband signal into $M$ contiguous
channels — can be implemented as a single polyphase filter followed
by an inverse FFT. The filter does the rate reduction; the FFT does
the frequency-domain channel separation.

The library doesn't yet ship a dedicated channelizer class, but the
construction is straightforward to compose from `PolyphaseDecimator`
plus the existing `dft` / `fft` primitives. See the
[Pattern Catalog](./patterns/) for where this is a gap.

## Pattern-to-API mapping

For the full mapping of multirate problems to library APIs, see the
[Pattern Catalog](./patterns/). The very short version:

| You want to... | Use |
|---|---|
| Decimate by integer $M$ | `PolyphaseDecimator` |
| Interpolate by integer $L$ | `PolyphaseInterpolator` |
| Convert by rational $L/M$ | `RationalResampler` |
| Decimate by 2 with sharp transition | `HalfBandFilter` |
| Decimate by a large factor at the highest rate | `CICDecimator` |
| Tune a band down to baseband | `DDC` |
| Cascade multiple decimators | `DecimationChain` |

Each row is covered in detail on the [pattern catalog](./patterns/)
page, with worked examples and links to the per-component reference
pages.

## Where to go next

- [Pattern Catalog](./patterns/) — problem→API table with worked examples
- [Polyphase Decimator](../acquisition/polyphase-decimator/) — the foundational decimating-FIR primitive
- [CIC](../acquisition/cic/) — Hogenauer's bulk-rate-reduction filter
- [Half-Band](../acquisition/halfband/) — Mintzer's ↓2 stage with ~75% zero taps
- [Decimation Chain](../acquisition/decimation-chain/) — composing the above
- [DDC](../acquisition/ddc/) — receiver-front-end composition
- [SDR Receiver Front-End Overview](../acquisition/overview/) — the SDR-specific application of these primitives

## References

- M. Bellanger, G. Bonnerot, M. Coudreuse, *"Digital Filtering by
  Polyphase Network: Application to Sample-Rate Alteration and
  Filter Banks,"* IEEE Trans. ASSP, 1976
- E. B. Hogenauer, *"An Economical Class of Digital Filters for
  Decimation and Interpolation,"* IEEE Trans. ASSP, 1981
- F. Mintzer, *"On Half-Band, Third-Band, and Nth-Band FIR Filters
  and Their Design,"* IEEE Trans. ASSP, 1982
- R. E. Crochiere, L. R. Rabiner, *Multirate Digital Signal
  Processing*, Prentice Hall, 1983
- P. P. Vaidyanathan, *Multirate Systems and Filter Banks*,
  Prentice Hall, 1993
- F. Harris, *Multirate Signal Processing for Communication
  Systems*, Prentice Hall, 2004
