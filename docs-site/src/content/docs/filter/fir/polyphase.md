---
title: Polyphase Decomposition
description: Efficient multirate filtering via polyphase interpolation and decimation in sw::dsp
---

**Polyphase decomposition** splits a single FIR filter into a bank of smaller
sub-filters, enabling multirate operations (interpolation and decimation) at
dramatically lower computational cost.

## The idea

Consider an FIR filter $H(z)$ with $M$ taps. The impulse response $h[n]$ can
be partitioned into $L$ sub-sequences (phases):

$$
E_k(z) = \sum_{m=0}^{\lfloor M/L \rfloor} h[mL + k]\, z^{-m}, \quad k = 0, 1, \ldots, L-1
$$

The original filter is then:

$$
H(z) = \sum_{k=0}^{L-1} z^{-k}\, E_k(z^L)
$$

Each sub-filter $E_k$ operates at $1/L$ the original rate.

## Decimation by $M$

When downsampling by a factor $M$, only every $M$th output sample is kept.
Computing all samples and discarding $M-1$ out of $M$ wastes computation.
The polyphase decimator restructures the filter so that each sub-filter
processes at the **output rate** $f_s / M$:

$$
y[n] = \sum_{k=0}^{M-1} E_k(z)\, x[nM - k]
$$

**Savings**: instead of $N_{\text{taps}}$ multiplications per input sample,
only $N_{\text{taps}} / M$ multiplications per output sample.

## Interpolation by $L$

Upsampling by $L$ inserts $L-1$ zeros between each input sample, then filters.
The polyphase interpolator avoids multiplying by zero by computing each output
phase from the original (non-zero-stuffed) input:

$$
y[nL + k] = E_k(z)\, x[n], \quad k = 0, 1, \ldots, L-1
$$

**Savings**: $N_{\text{taps}} / L$ multiplications per output sample instead
of $N_{\text{taps}}$.

## Noble identities

The Noble identities justify moving downsamplers and upsamplers across filters
in the signal flow graph:

1. A downsampler by $M$ followed by $E(z)$ equals $E(z^M)$ followed by a
   downsampler by $M$.
2. $E(z)$ followed by an upsampler by $L$ equals an upsampler by $L$ followed
   by $E(z^L)$.

These identities are the formal basis for polyphase decomposition. They
guarantee that the restructured system produces identical output to the
naive approach.

## Library API

### Polyphase decimator

```cpp
#include <sw/dsp/filter/fir/polyphase.hpp>
using namespace sw::dsp;

// Design a 128-tap anti-aliasing filter, then decimate by 4
auto h = windowed_sinc_lowpass<double>(128, 48000.0, 6000.0,
                                       WindowType::Kaiser, 7.0);

PolyphaseDecimator<double> decimator(h, 4);

// Feed samples one at a time; output is produced every 4th call
std::optional<double> out;
for (auto x : input_buffer) {
    out = decimator.process(x);
    if (out) {
        // Store or forward the decimated sample *out
    }
}
```

### Polyphase interpolator

```cpp
#include <sw/dsp/filter/fir/polyphase.hpp>
using namespace sw::dsp;

// Interpolate by 3: 16 kHz -> 48 kHz
auto h = windowed_sinc_lowpass<double>(96, 48000.0, 8000.0,
                                       WindowType::Kaiser, 7.0);

PolyphaseInterpolator<double> interpolator(h, 3);

// Each input sample produces 3 output samples
std::array<double, 3> out;
for (auto x : input_buffer) {
    interpolator.process(x, out);
    // out[0], out[1], out[2] are the interpolated samples
}
```

### Rational rate conversion

Combining interpolation by $L$ and decimation by $M$ gives rational rate
conversion by $L/M$. The library supports this with a single fused
polyphase structure:

```cpp
// Convert 44100 Hz -> 48000 Hz (ratio 160/147)
auto h = windowed_sinc_lowpass<double>(
    160 * 12, 48000.0 * 160, 22050.0, WindowType::Kaiser, 10.0);

PolyphaseInterpolator<double> up(h, 160);
PolyphaseDecimator<double> down(h, 147);
```

## Computational comparison

For a signal at $f_s = 48\,\text{kHz}$ decimated by $M = 4$ with a 128-tap
filter:

| Method | Multiplications per output sample |
|---|---|
| Filter then downsample | $128 \times 4 = 512$ (wasteful) |
| Polyphase decimation | $128$ |
| Savings | $4\times$ |

For interpolation by $L = 3$ with a 96-tap filter:

| Method | Multiplications per output sample |
|---|---|
| Zero-stuff then filter | $96$ (includes multiply-by-zero) |
| Polyphase interpolation | $32$ |
| Savings | $3\times$ |

## Mixed-precision considerations

Polyphase sub-filters have shorter impulse responses, which reduces
accumulator dynamic range requirements. This makes polyphase decomposition
particularly attractive for narrow arithmetic types like `posit<16,1>`: the
shorter accumulation chains produce less round-off buildup than a single
long convolution.
