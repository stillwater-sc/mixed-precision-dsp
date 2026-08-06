---
title: Polyphase M-Channel Channelizer Demo
description: Bellanger's 1976 polyphase channelizer — a single polyphase filter followed by an M-point IFFT emits all M channel outputs simultaneously at cost O(K + M log M) per input sample instead of O(M*K)
---

The `channelizer` demonstrator (issue [#137](https://github.com/stillwater-sc/dsp/issues/137))
implements the **Bellanger 1976 M-channel analysis channelizer** — a
single polyphase filter followed by an M-point inverse FFT that
emits M parallel channel outputs at rate $f_s / M$, at roughly $1/M$
the arithmetic cost of the naive M-parallel-DDC design.

This closes the "channelizer" gap flagged in the
[Pattern Catalog](./patterns/) — every multirate primitive there had a
first-class API except the channelizer, which was described as "compose
`PolyphaseDecimator` + FFT." This demo introduces
`sw::dsp::multirate::Channelizer` as the promised composition.

## The Bellanger construction

Consider a naive M-channel filter bank: for each channel
$c = 0, \ldots, M-1$, mix the input down by $\exp(-j 2 \pi c n / M)$,
lowpass with a length-$MK$ prototype $h[n]$, and decimate by $M$.
Total cost: $M \cdot MK / M = MK$ multiplies per input sample (each of
M parallel filters costs $K$ per output at rate $f_s/M$).

Bellanger's insight is that **the M mixers plus M lowpass filters
collapse into a single polyphase filter followed by one M-point IFFT**:

1. Decompose the prototype into $M$ polyphase phases:
   $$E_k[p] = h[pM + k], \qquad k = 0, \ldots, M-1,\ p = 0, \ldots, K-1$$

2. Distribute each input block $x[qM+0], x[qM+1], \ldots, x[qM+M-1]$
   through the $M$ sub-filters via a rotating commutator: sub-filter
   $k$ receives the sample at block position $M-1-k$ so that its
   "newest input" at output time $q$ is exactly $x[qM - k]$.

3. Collect the sub-filter outputs into an $M$-vector and take its
   $M$-point IFFT. Output index $c$ is channel-$c$'s complex output
   at time $q$.

Cost per input sample: $K$ multiplies for the polyphase FIR plus
$\frac{M \log M}{2M}$ multiplies (amortized) for the IFFT — that is,
$O(K + \log M)$ instead of $O(MK)$.

### Why the IFFT works

The channel-$c$ bandpass filter is $h[n] \exp(j 2 \pi c n / M)$
(prototype modulated to channel-$c$ center frequency). Decomposing the
modulated filter into polyphase phases:

$$
h[pM + k] \cdot \exp(j 2 \pi c (pM + k) / M) = E_k[p] \cdot \exp(j 2 \pi c k / M)
$$

(the $\exp(j 2 \pi c p)$ factor is unity for integer $c, p$). Summing
channel-$c$'s output across the $M$ sub-filters is exactly an $M$-point
IDFT indexed by $k$:

$$
Y_c[q] = \sum_k \exp(j 2 \pi c k / M) \cdot \left( \sum_p E_k[p]\ x_k[q-p] \right)
$$

where $x_k[q]$ is the sub-sampled input stream feeding sub-filter $k$.
The bracketed sum is one sub-filter output; the outer sum-with-twiddle
is one M-point IDFT.

## Running the demo

```bash
cmake --preset ci
cmake --build build-ci --target channelizer -j4
./build-ci/applications/multirate_examples/channelizer/channelizer --csv=out.csv
```

## What the demo measures

**Test signal.** A real multitone with tones at $\{6, 12, 18\}$ kHz
against $f_s = 48$ kHz and $M = 8$. Each tone lands at the center of a
distinct channel (channels 1, 2, 3 at $c \cdot f_s / M$). Because the
input is real, channels 5, 6, 7 (the mod-$M$ conjugates of 3, 2, 1)
also carry mirrored energy. Channels 0 (DC) and 4 (Nyquist) receive
nothing.

**Metric 1: in-channel SNR.** For a tone at channel-$c$'s center, the
channel-$c$ output is a complex DC constant. The FFT of the channel
output should have all its energy in bin 0; anything in bins $k > 0$
is arithmetic noise or leakage. SNR = $\lvert \text{bin}[0] \rvert^2 /
\sum_{k>0} \lvert \text{bin}[k] \rvert^2$.

**Metric 2: cross-channel rejection.** For a quiet channel (0 or 4),
its output time-series should be near zero. The peak time-domain
magnitude relative to the loud channels' peak reports how much of
neighboring channels' energy leaks through the polyphase prototype's
stopband.

The FFT window is sized to a **power-of-two multiple of the trimmed
output length** — no zero-padding — so a constant-DC channel signal
reads cleanly at bin 0 without the sinc-of-rectangular-window sidelobe
artifact that shows up when you zero-pad a constant.

## Reference results

At $M = 8$, $K = 24$ taps per phase (192 total prototype taps),
Kaiser $\beta = 12$ (~-115 dB prototype stopband):

**Loud channels (in-channel SNR):**

| Channel | Reference SNR | Rounded to sentinel |
|---:|---:|---|
| 1 | 246.5 dB | *(finite)* |
| 2 | 244.4 dB | *(finite)* |
| 3 | 246.4 dB | *(finite)* |
| 5 | 246.4 dB | *(finite)* |
| 6 | 244.4 dB | *(finite)* |
| 7 | 246.5 dB | *(finite)* |

Worst case is 244 dB — over 180 dB clear of the 60 dB acceptance floor.
The floor at these values is essentially the FFT numerical noise
(double-precision N=1024 FFT has ~-320 dB noise floor per bin, so
integrated across 1023 bins the noise pool is ~-290 dB total power,
matching what we measure).

**Quiet channels (cross-channel rejection):**

| Channel | Reference peak magnitude |
|---:|---:|
| 0 (DC)     | -133.2 dB |
| 4 (Nyquist) | -133.2 dB |

Well below the -60 dB acceptance ceiling. The floor here is Kaiser
$\beta = 12$'s prototype stopband depth combined with double-precision
arithmetic noise.

**Cross-precision sweep:**

| Config | Worst loud-channel SNR | Worst quiet-channel rejection |
|---|---:|---:|
| `reference` (double)         | 244.4 dB | -133.2 dB |
| `float`                       | 245.7 dB | -132.1 dB |
| `posit<32,2>`                 | 245.7 dB | -133.0 dB |
| `posit<16,2>`                 | 245.9 dB |  -64.3 dB |
| `cfloat<32,8>`                | 245.7 dB | -132.1 dB |
| `fixpnt<32,24>`               | 300+ dB  | -128.0 dB |

`posit<16,2>` sits right at the -64 dB rejection floor — 4 dB clear of
the acceptance ceiling but visibly less headroom than the 32-bit
configs. Every 32-bit type — IEEE float, posit, cfloat, fixpnt Q8.24 —
gives essentially the same performance as double, indicating that a
well-designed polyphase prototype makes the channelizer's arithmetic
demands modest.

## When to reach for this pattern

Use `multirate::Channelizer` when you need **all M channel outputs
simultaneously** and $M$ is a power of two. The cost advantage over M
parallel DDCs grows linearly with M — for M = 16 the polyphase
channelizer is roughly 16× faster; for M = 64 it's 64× faster.

Prefer a single `DDC` (see [Acquisition](../acquisition/ddc/)) when
you only need one channel at an arbitrary tune frequency (channelizer
frequencies are locked to $c \cdot f_s / M$).

Prefer a `RationalResampler` (see [Audio Resampler](./audio-resampler/))
when you need one channel at a rate that isn't $f_s / M$.

## Design knobs

- **$M$ (channels)**: must be a power of two. Larger $M$ = finer
  channelization (narrower per-channel bandwidth $f_s / M$) and larger
  IFFT — but per-input-sample compute stays $O(K + \log M)$.
- **$K$ (taps per phase)**: sets prototype filter length and channel
  edge sharpness. $K \approx 16$ to $K \approx 32$ is typical; deeper
  values give sharper channel skirts at proportional compute cost.
- **Kaiser $\beta$**: controls prototype stopband depth (crosstalk
  between channels). $\beta = 8$ gives $\sim -58$ dB stopband;
  $\beta = 12$ gives $\sim -115$ dB. The demo defaults to $\beta = 12$
  for the tight cross-channel rejection numbers reported above.

## Source

- Header: `include/sw/dsp/multirate/channelizer.hpp`
- Application: `applications/multirate_examples/channelizer/channelizer.cpp`
- Build target: `channelizer` (default preset `ci` builds it)
- CSV schema: `pipeline, config, scalar_type, channel, expected_tone_channel, kind, value_db`

## Related pages

- [SDR Polyphase Channelizer](../sdr/channelizer/) — the synthesis half. This bank is maximally decimated and therefore has **no perfect-reconstruction property**: recombining its channels bottoms out at a ~17% residual. If you need to rebuild the wideband stream, use the oversampled analysis/synthesis pair there and accept a shorter prototype
- [Multirate Overview](./overview/) — polyphase decomposition + Noble identity theory, including the historical context of Bellanger's 1976 paper
- [Pattern Catalog](./patterns/) — full multirate problem→API mapping
- [DDC](../acquisition/ddc/) — single-channel tune-and-decimate alternative
- [Audio Resampler](./audio-resampler/) — rational sample-rate conversion sibling demo
- [Fractional Delay](./fractional-delay/) — polyphase filter-bank sibling primitive
