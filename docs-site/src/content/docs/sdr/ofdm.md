---
title: OFDM
description: Orthogonal frequency-division multiplexing — cyclic prefix, subcarrier layout, pilot-based channel estimation, PAPR, and the FFT precision the subcarrier orthogonality depends on
---

```cpp
#include <sw/dsp/sdr/ofdm.hpp>   // OfdmConfig, OfdmLayout, OfdmModulator,
                                 // OfdmDemodulator, papr_db
```

OFDM carries data on $N$ orthogonal subcarriers by treating the
**frequency domain as the thing being transmitted**: load the
subcarriers, inverse transform to get a time-domain symbol, send it.
The receiver transforms back and reads the subcarriers off directly.

That inversion of the usual picture is the whole idea, and every
property below follows from it — including the two that make OFDM
awkward: a high peak-to-average ratio, and a sensitivity to anything
that breaks subcarrier orthogonality.

## The cyclic prefix is the whole trick

Copying the tail of each time-domain symbol to its front makes the
channel's **linear** convolution look **circular** over the retained
window. Circular convolution is per-bin multiplication in the frequency
domain, so a multipath channel that would otherwise smear symbols into
each other becomes **one complex gain per subcarrier**, which a single
division undoes.

$$
Y[k] = H[k] \cdot X[k] + N[k]
\qquad\Longrightarrow\qquad
\hat{X}[k] = \frac{Y[k]}{H[k]}
$$

That is why equalizing OFDM is a divide rather than an adaptive filter,
and it is the reason OFDM won for wideband channels.

**The prefix must be at least as long as the channel's impulse
response.** This header cannot check that — it never sees the channel —
so it is a caller obligation. A prefix shorter than the delay spread
reintroduces inter-symbol interference *and* destroys the circularity,
so the equalizer's model becomes wrong at the same moment its input
becomes contaminated. The failure is not graceful.

## Subcarrier layout

```cpp
using namespace sw::dsp::sdr;

OfdmConfig cfg;
cfg.fft_size          = 64;   // power of two, >= 8
cfg.cyclic_prefix     = 16;   // samples; >= channel impulse length
cfg.guard_subcarriers = 8;    // nulled band centred on index N/2
cfg.pilot_spacing     = 8;    // every Nth active subcarrier

OfdmLayout layout(cfg);
layout.active().size();      // 54
layout.num_data();           // 40
layout.num_pilots();         // 14
layout.symbol_length();      // 80 = fft_size + cyclic_prefix
```

The allocation rules:

| Index | Role | Why |
|---|---|---|
| 0 | **Always nulled** (DC) | It sits on the receiver's LO leakage and DC offset — the one bin guaranteed to be contaminated |
| within `guard_subcarriers/2` of $N/2$ | Nulled (guard band) | $N/2$ is the spectral folding edge; the guard gives the analog filters room to roll off |
| every `pilot_spacing`-th active | Pilot | Carries a known reference instead of data |
| the rest | Data | |

`OfdmLayout` is computed **once from a config and shared by both
halves**, so the modulator and demodulator cannot disagree about which
subcarrier means what. That is not a convenience — a layout mismatch
between transmitter and receiver produces a plausible-looking
constellation made entirely of the wrong bins.

The constructor rejects configurations that cannot work, with reasons
rather than error codes: a guard band that leaves no active
subcarriers, fewer than two pilots (you cannot interpolate a channel
estimate from one), no data subcarriers, a non-power-of-two FFT, a
prefix longer than the symbol, or `pilot_spacing < 2` — "otherwise
every subcarrier is a pilot and none carries data".

## Modulation and demodulation

```cpp
OfdmModulator<double>   mod(cfg);
OfdmDemodulator<double> demod(cfg);

// num_data() constellation points in, symbol_length() time samples out
auto tx = mod.modulate(data_symbols);

// one symbol in, num_data() equalized points out
auto rx = demod.demodulate(channel_output);
```

The modulator loads the grid, runs `fft_inverse`, and prepends the last
`cyclic_prefix` samples. The demodulator **drops the prefix** — its job
is done once the channel has convolved — runs `fft_forward`, estimates
the channel, and divides.

## Channel estimation

Least squares at the pilots, $H = Y/P$ (for a unit-magnitude pilot,
just $Y$ times the conjugate), then **linear interpolation** across the
data subcarriers between them.

Outside the outermost pilots the nearest estimate is **held, not
extrapolated**. Extrapolating a channel estimate is how an edge
subcarrier acquires confident nonsense: the fit has no data beyond the
last pilot, so an extrapolated slope is pure model, and it lands on the
subcarriers that are already the most marginal.

```cpp
const auto& H = demod.channel_estimate();   // one entry per subcarrier, zero on nulls
```

The estimate is refreshed from **every** symbol, so a slowly varying
channel is tracked without any explicit tracking loop.

### Pilot spacing against multipath

The estimate is only as good as the interpolation between pilots, and
how far apart pilots can be depends on how frequency-selective the
channel is. Measured by `tests/test_sdr_ofdm.cpp` against a 2-tap and a
3-tap multipath channel:

| Pilot spacing | 2-tap EVM | 3-tap EVM |
|---|---|---|
| 2 | 4.378e-03 | 1.835e-02 |
| 4 | 9.645e-03 | 3.427e-02 |
| 8 | 3.595e-02 | 1.242e-01 |

Both columns degrade roughly **linearly in the spacing**, and the
3-tap channel is consistently ~3.5× worse at every spacing. That ratio
is the point: a more frequency-selective channel does not merely need
*more* pilots, it needs them at a spacing scaled to its coherence
bandwidth. Doubling the spacing on the 3-tap channel costs more EVM
than the 2-tap channel has in total.

The overhead is real — at `pilot_spacing = 2` half the active
subcarriers carry no data — so this table is a rate-versus-robustness
choice, not a tuning knob with a right answer.

## Peak-to-average power ratio

```cpp
double papr = papr_db<Complex>(time_domain_block);
```

OFDM's characteristic weakness. Summing many independent subcarriers
gives a near-Gaussian time-domain signal whose peaks run far above its
mean power, which is what forces a linear power amplifier to be backed
off — directly costing transmit efficiency. It grows roughly as
$10\log_{10}(N)$ in the worst case. Measured, mean over 200 symbols:

| FFT size | Mean PAPR |
|---|---|
| 16 | 5.26 dB |
| 64 | 7.72 dB |
| 256 | 12.03 dB |

Every doubling of $N$ costs a little over 2 dB of amplifier headroom,
and it compounds: going from a 16-point to a 256-point transform — a
perfectly ordinary design change for a wider channel — costs **6.8 dB**
of backoff.

This connects directly to the [precision analysis](./precision/): a
signal that must be backed off is a signal that does *not* sit at full
scale, and a uniform quantization grid loses resolution exactly in
proportion to that backoff while a tapered format does not. OFDM is the
case where posit's dynamic range earns its keep.

## FFT precision and subcarrier orthogonality

Subcarrier orthogonality is a property of the **transform**, so
arithmetic error in the FFT appears as **intercarrier interference** —
energy from one subcarrier landing in another's bin. It is
indistinguishable from a channel impairment at the demapper, which is
why it must be measured separately.

Measured on 64-QAM through an ideal channel, so the entire residual is
transform arithmetic:

| FFT precision | EVM | dB |
|---|---|---|
| `double` | 3.437e-16 | −309.3 |
| `posit<32,2>` | 1.234e-08 | −158.2 |
| `float` | 1.544e-07 | −136.2 |
| `cfloat<32,8>` | 1.603e-07 | −135.9 |
| `posit<16,2>` | 3.999e-04 | −68.0 |

Two readings worth taking:

- **`posit<32,2>` is 22 dB better than `float`** at the same width.
  The FFT's butterflies work on values near unity after the
  normalization, which is exactly where posit's tapered precision puts
  its extra mantissa bits. `cfloat<32,8>` tracks `float` to three
  digits, as it should — it *is* binary32.
- **`posit<16,2>` at −68 dB is still comfortably below any practical
  link floor.** Even 256-QAM's decision distance corresponds to about
  −22 dB EVM, and a real link operates far above its own noise. A
  16-bit transform is not the constraint in an OFDM receiver; the
  channel is.

That second point is the useful design conclusion. It is tempting to
read the 240 dB spread in this table as a reason to spend width on the
FFT, but the column that matters is *whether the number is below the
link's floor* — and at 16 bits it already is.

## Verified round trips

`tests/test_sdr_ofdm.cpp` checks the chain at three levels, and the
progression is deliberate:

| Test | What it pins down |
|---|---|
| `ideal_round_trip` | Layout and transform agree — data out equals data in |
| `flat_channel_is_exact` | The equalizer is not introducing error of its own |
| `multipath_equalization` | The cyclic prefix is doing its job (the table above) |
| `bits_through_awgn` | End to end: **0 errors in 16000 bits** |

Each isolates one failure. A single end-to-end test that passed would
say nothing about *which* stage was right, and one that failed would
say nothing about which stage was wrong.

## See also

- [SDR Overview](./overview/) — where OFDM sits relative to the
  single-carrier chain
- [Constellation and Metrics](./constellation/) — the mapper feeding
  the subcarriers and the EVM measurement used throughout this page
- [DFT and FFT](../../spectral/dft-fft/) — the transform this module
  runs on, and its own precision characterization
- [SDR Precision Analysis](./precision/) — the input-backoff
  measurement that the PAPR section points at
- [Channelizer](./channelizer/) — the other multi-carrier structure in
  this module, and how it differs
