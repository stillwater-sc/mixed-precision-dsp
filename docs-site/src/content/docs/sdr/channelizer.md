---
title: Polyphase Channelizer
description: The oversampled analysis/synthesis filter-bank pair — weighted overlap-add, perfect reconstruction, and why the prototype spans exactly one transform
---

```cpp
#include <sw/dsp/sdr/channelizer.hpp>   // OversampledChannelizer,
                                        // OversampledSynthesizer, Channelizer
```

A channelizer splits one wideband stream into $M$ narrowband channels;
a synthesizer puts them back. Together they let a receiver work on one
channel at a time — filter it, null it, re-modulate it — and rebuild
the composite.

The **analysis** half already existed as
[`sw::dsp::multirate::Channelizer`](../../multirate/channelizer/),
added in v0.8: a Bellanger polyphase bank that splits a wideband stream
into $M$ channels at $f_s/M$. This header adds the missing **synthesis**
half and re-exports the analysis one, so a caller building a
channelized link reaches for one include rather than two.

## Two different banks, two different jobs

This header does not simply add an inverse to the v0.8 bank. It ships a
**different pair**, and choosing between them is the first decision:

| | `multirate::Channelizer` | `OversampledChannelizer` / `Synthesizer` |
|---|---|---|
| Decimation | Maximally decimated, $\downarrow M$ | Oversampled 2×, hop $= M/2$ |
| Prototype | `taps_per_phase` configurable | Exactly one transform, $N = M$ |
| Channel selectivity | Sharp — a long prototype is allowed | Modest — one transform of taps |
| Reconstruction | **None** | **Perfect**, to 1e-16 |
| Use when | Analysis only: you look at channels, you never rebuild | You must rebuild the composite |

**A maximally-decimated DFT filter bank has no perfect-reconstruction
property with a plain prototype.** Each channel is decimated by $M$,
the channel responses overlap, and the aliasing does not cancel.
Measured on the maximally-decimated pair, the best any combination of
FFT direction and commutator order achieved was a **17% relative
residual** — not a bug to be found, a structural limit. Oversampling
removes the constraint entirely.

So: if you only need to *look* at channels, the maximally-decimated
bank is the right tool and buys you a sharper channel response. If you
need to reconstruct, take the oversampled pair and accept the modest
selectivity.

## Weighted overlap-add

The oversampled pair runs the bank at $\mathrm{hop} = M/2$ — two channel
sample sets per $M$ input samples. Analysis windows the last $N$ samples
with the prototype, folds them into $M$ bins, and transforms:

$$
z[k] = \sum_p x[n-N+1+pM+k]\, h[pM+k],
\qquad k = 0 \ldots M-1
$$

$$
\text{channels} = \mathrm{FFT}(z)
$$

Synthesis is the exact inverse — transform back, unfold against the
same window, overlap-add at the same hop:

$$
z' = \mathrm{IFFT}(\text{channels}),
\qquad
y[pM+k] \mathrel{+}= z'[k]\, h[pM+k]
$$

```cpp
using namespace sw::dsp::sdr;

OversampledChannelizer<double> analysis(16);      // M = 16, hop = 8
OversampledSynthesizer<double>  synthesis(16);

auto channels = analysis.process(block);          // block.size() == hop()
auto wideband = synthesis.process(channels);      // channels.size() == M
```

Both halves must be built with the same $M$ and Kaiser $\beta$. They
share one prototype through `Channelizer::prototype_bank()`, so there
is no second copy of the design to drift out of step.

## The prototype spans exactly one transform

$N = M$. That is **not a simplification — it is the condition for
reconstruction**, and it is why `taps_per_phase` is not offered as a
parameter here.

A longer prototype ($N = MK$, $K > 1$) has to be *folded* into $M$ bins
before the transform, and that fold **aliases in time**: it sums input
samples $M$ apart, which an $M$-point transform cannot separate again.
Measured, driving white noise through the pair:

| $K$ | Relative reconstruction residual |
|---|---|
| 1 | **1.1e-16** (exact, at every $M$ tested) |
| 2 | 4.5e-02 |
| 4 | 2.2e-01 |
| 8 | 2.0e-01 |

The jump from $K=1$ to $K=2$ is fourteen orders of magnitude. This is
not a quality knob with a trade-off curve; it is a cliff, and the API
declines to put a foot near it. Getting both selectivity *and*
reconstruction needs a transform as long as the prototype, or a
designed paraunitary bank — neither is what this pair is.

## Prototype normalization

The prototype is normalized so its **squared** overlap-add sums to one:

$$
\sum_m h^2[n - m \cdot \mathrm{hop}] = 1 \quad\text{for every } n
$$

With that, analysis followed by synthesis reproduces the input
*exactly* rather than approximately — the window contributions at each
output sample add to unity **by construction instead of by luck**. It
is a property of the hop, so it is computed once at construction for
the configured $M$.

## Structure: analysis and synthesis are transposes

In the literal signal-flow sense — every step runs in reverse order and
in the opposite direction:

| Analysis | Synthesis |
|---|---|
| Commutate $M$ inputs into the $M$ sub-filters | FFT the $M$ channel values |
| Run each sub-filter | Feed FFT output $k$ to sub-filter $k$ |
| IFFT the sub-filter outputs | Commutate the $M$ sub-filter outputs into $M$ wideband samples |

The analysis bank's IFFT becomes a **forward** FFT here, and its input
commutator becomes an output commutator with the same $M-1-k$ ordering,
so the two agree on which channel index means which frequency. Getting
either reversal wrong produces a bank that channelizes correctly and
reconstructs to garbage.

## Delay and reconstruction quality

The cascade delay is exact and reported rather than silently absorbed:

```cpp
std::size_t d = OversampledSynthesizer<double>::cascade_delay(M);   // M - M/2
```

Measured by `tests/test_sdr_channelizer.cpp`:

| $M$ | hop | Delay | Reconstruction error |
|---|---|---|---|
| 4 | 2 | 2 | 1.810e-16 |
| 8 | 4 | 4 | 1.060e-16 |
| 16 | 8 | 8 | 1.432e-16 |
| 32 | 16 | 16 | 2.023e-16 |
| 64 | 32 | 32 | 2.090e-16 |

Machine precision at every size, and the delay confirmed against
least-squares alignment.

## Channel isolation

The price of a one-transform prototype:

| Tone in channel | Peak channel | Adjacent rejection |
|---|---|---|
| 3 | 3 | 22.7 dB |
| 5 | 5 | 22.1 dB |

**~22 dB is modest**, and it is the direct consequence of the $N = M$
constraint above — a longer prototype would sharpen this considerably
and destroy reconstruction. If your application needs 60 dB of adjacent
rejection and does not need to rebuild the composite, use
[`multirate::Channelizer`](../../multirate/channelizer/) with a longer
`taps_per_phase`.

Channels are nonetheless independently manipulable: nulling 1 of 16
changes the reconstructed output by **26.4%**, confirming that each
channel genuinely carries its own share of the signal.

## Precision behaviour

Reconstruction error against the bank's precision, $M = 16$:

| Bank precision | Reconstruction error |
|---|---|
| `double` | 1.463e-16 |
| `posit<32,2>` | 5.599e-09 |
| `float` | 7.852e-08 |
| `cfloat<32,8>` | 7.852e-08 |
| `posit<16,2>` | 3.075e-04 |

The same pattern as the [OFDM transform](./ofdm/#fft-precision-and-subcarrier-orthogonality),
and for the same reason: **`posit<32,2>` is 14× better than `float`**
at equal width, because the windowed and normalized values the bank
works on sit near unity, where posit puts its extra mantissa bits.
`cfloat<32,8>` matches `float` exactly — it is binary32.

This is a datapath measurement, so it separates the number systems
cleanly across four orders of magnitude, unlike the
[synchronization loops](./synchronization/) where every type produced
the same answer. When you are deciding where to spend width in an SDR,
that distinction is the one that matters: **filter banks and transforms
reward precision, feedback loops reward the right state
representation.**

## See also

- [Analysis Channelizer](../../multirate/channelizer/) — the
  maximally-decimated bank, when you need selectivity and not
  reconstruction
- [SDR Overview](./overview/) — where the channelizer sits in a
  multi-channel receiver
- [OFDM](./ofdm/) — the other multi-carrier structure here; OFDM makes
  subcarriers orthogonal by construction, a channelizer makes channels
  separable by filtering
- [DFT and FFT](../../spectral/dft-fft/) — the transform both halves
  run on
- [Multirate Signal Processing](../../multirate/overview/) — polyphase
  decomposition and the Noble identities behind the structure
