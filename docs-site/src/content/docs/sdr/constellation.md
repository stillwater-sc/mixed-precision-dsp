---
title: Constellation, Pulse Shaping and Metrics
description: QAM/PSK geometry, Gray labelling, unit-power normalization, hard and soft demapping, RRC pulse shaping, and the EVM/MER/BER measurement primitives
---

```cpp
#include <sw/dsp/sdr/constellation.hpp>   // Constellation, Modulation
#include <sw/dsp/sdr/rrc.hpp>             // rrc_filter, raised_cosine_filter, peak_isi
#include <sw/dsp/sdr/metrics.hpp>         // evm, mer_db, ber, theoretical_ber_awgn
```

These three headers are the datapath of a digital link: the table that
turns bits into points, the pulse that carries them through a
band-limited channel, and the measurements that say how well it worked.
Unlike the [synchronization loops](./synchronization/), nothing here
holds state between calls — `Constellation` is a table, the RRC
functions are designers, and the metrics are batch measurements. A
single `Constellation` instance is safe to share across streams.

## Constellation geometry

`Constellation<T>` is an immutable table of $M$ complex points, indexed
by bit pattern:

```cpp
using namespace sw::dsp::sdr;

Constellation<double> qam16(Modulation::qam16);

qam16.order();            // 16
qam16.bits_per_symbol();  // 4
qam16.average_power();    // 1.0 by construction
```

### Gray labelling

Constellation-adjacent points differ in **exactly one bit**, which is
what makes a symbol error usually cost a single bit error rather than
half the symbol. The construction leans on the defining property of the
reflected binary code: $\mathrm{gray}(k)$ and $\mathrm{gray}(k+1)$
differ in one bit, where

$$
\mathrm{gray}(k) = k \oplus (k \gg 1).
$$

Position $k$ — on the circle for PSK, on an axis for square QAM — is
therefore *labelled* $\mathrm{gray}(k)$, and recovering the position
from a label is the inverse map. Writing the table by label rather than
by position means indexing it with a bit pattern needs no further
decoding: `points()[index]` is already the right point.

The two families are labelled differently, and it matters:

- **PSK** Gray-codes around the circle. All $k$ bits participate in one
  cyclic sequence.
- **Square QAM** Gray-codes **each axis independently**: the leading
  $k/2$ bits drive I, the trailing $k/2$ drive Q. That is what makes
  Gray-coded square QAM decompose into two independent PAM channels,
  which is why its BER expression is a per-axis formula rather than a
  circle-geometry one.

### Unit-average-power normalization

Every constellation is scaled so $E[|s|^2] = 1$ over equiprobable
symbols. This is not cosmetic: it is what lets an SNR, a noise variance
or an $E_b/N_0$ mean the same thing across schemes, so a sweep can
change modulation order without re-deriving the channel.

PSK points already sit on the unit circle. Square $M$-QAM on the
odd-integer grid $\pm 1, \pm 3, \ldots, \pm(L-1)$ per axis has

$$
E[|s|^2] = \frac{2(M-1)}{3},
\qquad\text{so}\qquad
\text{scale} = \sqrt{\frac{3}{2(M-1)}},
$$

the familiar $1/\sqrt{10}$, $1/\sqrt{42}$, $1/\sqrt{170}$ for 16-, 64-
and 256-QAM.

### Three conventions that quietly ruin a receiver

1. **Bits are MSB-first**, one bit per `std::uint8_t`, values 0 or 1.
   For square QAM the leading half drives I, the trailing half drives Q.
2. **A positive LLR means bit 0 is more likely.** The returned quantity
   is $\ln\!\big(P(b=0\mid r) / P(b=1\mid r)\big)$. Decoders differ on
   this sign; check before wiring one up.
3. **`noise_variance` is $E[|n|^2]$ for the complex noise**, not the
   per-dimension variance. Complex AWGN of total variance $N_0$ carries
   $N_0/2$ in each of I and Q.

## Mapping and demapping

```cpp
Constellation<double> c(Modulation::qam16);

// bits -> symbol
std::uint8_t bits[4] = {1, 0, 1, 1};
auto s = c.map(bits);                    // complex_for_t<double>

// symbol -> nearest point -> bits
std::uint8_t out[4];
c.demap_hard_bits(s, out);               // exact inverse in the absence of noise
```

`demap_hard` is exhaustive over $M$ points. That is exact for every
scheme including the PSK ones, where a per-axis slicer would be wrong.
Square QAM does admit an $O(1)$ slicer; making that substitution is a
speed optimization to reach for once a profile demands it, not a
correctness one.

### Soft decisions

Two forms, both writing `bits_per_symbol()` LLRs:

```cpp
double llr[4];
c.demap_llr(r, n0, llr);           // exact
c.demap_llr_maxlog(r, n0, llr);    // max-log approximation
```

The exact form evaluates

$$
\mathrm{LLR}_k = \ln
\frac{\sum_{s\,:\,b_k=0} e^{-|r-s|^2/N_0}}
     {\sum_{s\,:\,b_k=1} e^{-|r-s|^2/N_0}}
$$

using the **max-subtraction form of log-sum-exp**, so the exponentials
stay bounded no matter how small $N_0$ is. Without that shift, a
high-SNR link — small $N_0$, so large negative exponents — underflows
every term to zero and the ratio becomes $0/0$.

The max-log approximation,

$$
\mathrm{LLR}_k \approx
\frac{\min_{s\,:\,b_k=1}|r-s|^2 - \min_{s\,:\,b_k=0}|r-s|^2}{N_0},
$$

keeps the sign and the ordering of the exact LLR, costs **no
transcendentals**, and is therefore usable with any `DspField` — a
fixed-point or narrow-posit type that has no `exp`/`log` can still
produce soft decisions. It is accurate to a fraction of a dB at the
SNRs where a coded link operates, and is what a real receiver uses.

`demap_llr` is a template member, so it is only instantiated when
called: a scalar type without transcendentals can use everything else
on the class without a compile error.

## Pulse shaping

The root-raised-cosine pulse splits the Nyquist filter evenly between
transmitter and receiver. An RRC at each end convolves to a full raised
cosine, whose zero crossings fall at every non-zero multiple of the
symbol period — **zero intersymbol interference at the sampling
instants**, while the receive filter stays matched to the transmit
pulse, which is what maximizes SNR.

```cpp
using namespace sw::dsp::sdr;

const std::size_t sps  = 4;
const std::size_t span = 10;                       // symbols
auto h = rrc_filter<double>(span * sps + 1, sps, 0.35);
```

`num_taps` **must be odd**, and the designer throws if it is not. An odd
length puts a tap exactly on $t = 0$ — the symbol instant the zero-ISI
property is defined at. `span * samples_per_symbol + 1` is the idiom.

### The formula and its two singularities

With $x = t/T$ and rolloff $\alpha$:

$$
h_{\mathrm{rrc}}(x) =
\frac{\sin\!\big(\pi x (1-\alpha)\big) + 4\alpha x \cos\!\big(\pi x (1+\alpha)\big)}
     {\pi x \big(1 - (4\alpha x)^2\big)}
$$

which has two **removable** singularities that must be evaluated as
limits rather than computed:

$$
x = 0:\quad h = 1 + \alpha\left(\tfrac{4}{\pi} - 1\right)
$$

$$
|4\alpha x| = 1:\quad
h = \frac{\alpha}{\sqrt{2}}
\left[\left(1 + \tfrac{2}{\pi}\right)\sin\frac{\pi}{4\alpha}
    + \left(1 - \tfrac{2}{\pi}\right)\cos\frac{\pi}{4\alpha}\right]
$$

The second exists only for $\alpha > 0$ and lands on an **actual tap**
whenever $\mathrm{sps}/(4\alpha)$ is an integer — $\alpha = 1$ with
`sps = 4`, for instance. Evaluating the general form there produces
$0/0$. Since the singularities sit at exact rational positions, a tap
either lands on one or is a full grid step away, so the tolerance only
has to absorb the rounding in computing $x$.

### Normalization

| `PulseNormalization` | Property | Use |
|---|---|---|
| `unit_energy` *(default)* | $\sum h^2 = 1$ | Matched pairs. The composite then peaks at exactly 1 on the symbol instant, so zero-ISI reads directly off the composite samples. Matches MATLAB `rcosdesign`. |
| `unit_dc_gain` | $\sum h = 1$ | The pulse used as a plain interpolation filter rather than half of a matched pair — passes DC at unity gain. |

Unit energy has a second, less obvious benefit that the
[precision analysis](./precision/) depends on: a unit-energy RRC pair
passes the channel's noise variance through **unchanged**, with no
samples-per-symbol factor. Getting that wrong is worth about 6 dB of
apparent SNR.

### RRC is symmetric

"Convolve with the time-reversed RRC" — the matched filter — is the
same tap set. No reversal step is needed, and none is provided; a
reversal helper would be an inert copy.

### Using the taps

The designers return taps and hold no state; the shaping reuses the
library's multirate primitives:

```cpp
// TX: upsample and shape in one polyphase pass
PolyphaseInterpolator<Coeff, State, Sample> shaper(h, sps);
auto waveform = shaper.process_block(symbols);

// RX: matched filter
FIRFilter<Coeff, State, Sample> matched(h);
```

### Measuring residual ISI

```cpp
double isi = peak_isi(composite, sps);
```

`composite` is the full TX-RX response. The peak tap is taken as the
symbol instant; every other tap an exact multiple of
`samples_per_symbol` away from it should be zero, and the largest of
those, relative to the peak, is what is returned. An untruncated raised
cosine gives 0; a real design gives the **truncation floor**.

That floor is the number worth tracking when sweeping coefficient
precision, because it says how much of the eye a given tap set costs
before any channel noise is involved. Measured by
`tests/test_sdr_rrc.cpp`:

| Number system | span 16 (truncation-limited) | span 256 (quantization-limited) |
|---|---|---|
| `double` | 1.289e-03 | 3.533e-06 |
| `float` | 1.289e-03 | — |
| `posit<32,2>` | 1.289e-03 | 3.533e-06 |
| `cfloat<32,8>` | 1.289e-03 | — |
| `posit<16,2>` | 1.287e-03 | 8.191e-05 |
| `posit<8,2>` | 4.880e-03 | 4.904e-03 |

The two columns separate the two error sources, and that separation is
the point. At **span 16** every type from `double` to `posit<16,2>`
agrees to three digits: the truncation dominates, and coefficient
precision is invisible beneath it. Only `posit<8,2>` is bad enough to
show through. At **span 256** the truncation floor has dropped by two
and a half orders of magnitude and the number systems separate cleanly
— `posit<16,2>` is now 23× worse than `double`, and `posit<8,2>` has
not improved at all, because it was quantization-limited even at span
16.

The design lesson: **lengthening a filter buys nothing once its
coefficient precision is the binding constraint.** Measure both
regimes before spending taps.

## Measuring the link

`metrics.hpp` holds no state and takes batches, so the arithmetic it
reports on is the caller's, not its own. Every metric is accumulated in
`double` regardless of the scalar type the symbols arrive in — which is
what lets a `posit<16,2>` link and a `double` reference be compared on
the same axis. That mirrors the convention in
[`<sw/dsp/analysis/>`](../../analysis/overview/).

### EVM and MER

```cpp
EvmResult e = evm<Complex>(reference, received);
e.rms;          // fraction, e.g. 0.05
e.rms_percent;  // 5.0
e.rms_db;       // 20*log10(rms)
e.peak;         // max|r-s| / sqrt(mean|s|^2)

double m = mer_db<Complex>(reference, received);   // == -e.rms_db
```

$$
\mathrm{EVM}_{\mathrm{rms}} =
\sqrt{\frac{\overline{|r-s|^2}}{\overline{|s|^2}}},
\qquad
\mathrm{MER}_{\mathrm{dB}} = 10\log_{10}
\frac{\overline{|s|^2}}{\overline{|r-s|^2}}
$$

Both `rms` and `peak` are normalized by the **same** RMS reference
amplitude, so they are directly comparable and their ratio is the crest
factor of the error vector.

**EVM normalization is the classic trap.** The same cloud yields
different EVM numbers depending on the reference power used. Everything
here normalizes by the **mean reference symbol power**, the convention
3GPP uses. Standards that normalize by the peak constellation magnitude
report *smaller* figures for the same signal, so cross-standard
comparisons need care.

MER and RMS EVM are two faces of one measurement under this
normalization — `mer_db()` is literally `-evm().rms_db`. Both are
provided because both appear in specifications, and offering only one
invites a sign error at the call site.

### BER and the theoretical curves

```cpp
BerResult b = ber(transmitted_bits, received_bits);
double theory = theoretical_ber_awgn(Modulation::qam16, 10.0);
```

BPSK and QPSK are **exact**: Gray-coded QPSK is two independent BPSK
channels in quadrature, so they share a curve,
$P_b = Q(\sqrt{2E_b/N_0})$. The higher orders are the standard
nearest-neighbour approximations — they count only the dominant error
events, so they are slightly optimistic at low SNR and tighten as
$E_b/N_0$ rises. These are the curves quoted in the literature.

$Q(x)$ is written through `erfc` rather than as $1 - \Phi(x)$, so it
stays accurate in the far tail — exactly where BER curves live.

Measured against theory by `tests/test_sdr_metrics.cpp`:

| Scheme | $E_b/N_0$ | Measured | Theory | Errors |
|---|---|---|---|---|
| BPSK | 4.0 dB | 1.201e-02 | 1.250e-02 | 2403 |
| BPSK | 7.0 dB | 7.625e-04 | 7.727e-04 | 305 |
| QPSK | 4.0 dB | 1.236e-02 | 1.250e-02 | 2472 |
| QPSK | 7.0 dB | 7.775e-04 | 7.727e-04 | 311 |
| 16-QAM | 10.0 dB | 1.796e-03 | 1.754e-03 | 431 |
| 16-QAM | 12.0 dB | 1.360e-04 | 1.387e-04 | 136 |

The error counts are printed alongside deliberately: a BER measurement
is a Poisson count, and its relative standard deviation is
$1/\sqrt{N_{\mathrm{err}}}$. At 136 errors that is 8.6%, so a
measured-to-theory ratio of 0.98 says nothing on its own. Quoting a BER
without its error count is quoting a number without its uncertainty.

Two conversions are provided so $E_s/N_0$ and $E_b/N_0$ never get
confused:

```cpp
double esn0 = esn0_db_from_ebn0_db(Modulation::qam64, ebn0);  // + 10*log10(6)
double ebn0 = ebn0_db_from_esn0_db(Modulation::qam64, esn0);
```

### Separating structured error from noise

```cpp
IqImbalance iq = iq_imbalance<Complex>(reference, received);
```

This is the tool to reach for **when EVM is worse than the arithmetic
alone can explain.** Quantization and thermal noise scatter symbols
isotropically; a gain error, DC offset, common rotation or quadrature
error moves them *systematically*. `iq_imbalance` tells the two apart
by least-squares fitting the general real-linear map plus a DC term:

$$
\begin{bmatrix}\mathrm{Re}(r)\\ \mathrm{Im}(r)\end{bmatrix}
=
\begin{bmatrix}a & b\\ c & d\end{bmatrix}
\begin{bmatrix}\mathrm{Re}(s)\\ \mathrm{Im}(s)\end{bmatrix}
+
\begin{bmatrix}i_{\mathrm{off}}\\ q_{\mathrm{off}}\end{bmatrix}
$$

which is enough to express all four impairments and to tell them apart.
The reported `residual_evm` is what survives removing the entire fitted
model — the part of the error that is **not** structured. A clean link
returns unit gains, zero offsets, and a residual equal to the raw EVM.

**BPSK cannot be measured this way.** The fit needs a reference that
exercises **both axes**. BPSK symbols are all real, which leaves the
normal equations rank-deficient, and `iq_imbalance` throws rather than
returning a confident guess: no measurement of Q gain or quadrature
error exists in that data. Use QPSK or higher.

## Mixed-precision notes

`Constellation<T>` quantizes the table itself, and that turns out to be
the single largest arithmetic contributor at 8 bits — see the
[per-block attribution](./precision/#per-block-attribution). Measured
by `tests/test_sdr_metrics.cpp`, a 64-QAM table built in `posit<16,2>`
sits **0.0042% EVM (−87.5 dB)** from its `double` counterpart, which is
two orders of magnitude below any practical link's floor. At 8 bits it
is the dominant term.

The RRC designers are the opposite case. They compute the pulse in
`double` and project to `T` at the very end:

> the design is a one-off cost and the closed form is far better
> conditioned in `double` than in a narrow type, which keeps a
> `posit<16,2>` tap set as close to the ideal pulse as its own
> precision allows rather than compounding design error on top of
> representation error.

That is a deliberate split, and it is the same one the IIR module makes
between design precision and processing precision. **Design wide,
store narrow** — the design runs once, the processing runs forever.

## See also

- [SDR Overview](./overview/) — where these blocks sit in the link
- [Synchronization](./synchronization/) — the loops that deliver
  correctly-sampled, de-rotated symbols to the demapper
- [SDR Precision Analysis](./precision/) — which block's precision
  actually costs EVM, measured
- [Multirate Signal Processing](../../multirate/overview/) — the
  polyphase interpolator and decimator the shaping runs on
- [Filter Design Overview](../../filter/overview/) — the FIR machinery
  behind the matched filter
