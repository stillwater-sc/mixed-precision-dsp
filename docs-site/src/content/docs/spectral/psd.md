---
title: Power Spectral Density
description: Periodogram, Welch's method, and PSD estimation in mixed-precision DSP
---

Power Spectral Density (PSD) describes how the power of a signal is
distributed across frequency. Estimating the PSD from a finite-length
record is a core task in spectral analysis.

## Periodogram

The simplest PSD estimator is the periodogram. Given a length-$M$
signal $x[n]$ and its DFT $X[k]$, the periodogram is:

$$
P[k] = \frac{|X[k]|^2}{M}, \quad k = 0, 1, \dots, M-1
$$

While straightforward, the periodogram has high variance -- its estimate
at each frequency bin does not improve as the record length grows.

### One-sided PSD

For real-valued signals the spectrum is symmetric. The one-sided PSD
folds the negative-frequency energy into the positive side:

$$
P_{\text{one}}[k] =
\begin{cases}
P[k],   & k = 0 \text{ or } k = M/2 \\
2P[k],  & 1 \le k < M/2
\end{cases}
$$

This representation is standard for plotting and comparing real-world
measurements.

## Window power normalization

When a window function $w[n]$ is applied before the DFT, the periodogram
must be normalized by the window's power to preserve the correct PSD
level:

$$
S = \frac{1}{M} \sum_{n=0}^{M-1} |w[n]|^2
$$

The normalized periodogram becomes:

$$
P_w[k] = \frac{|X_w[k]|^2}{M \cdot S}
$$

where $X_w[k]$ is the DFT of the windowed signal $x[n] \cdot w[n]$.

## Welch's method

Welch's method reduces variance by averaging periodograms computed from
overlapping segments of the signal:

1. Divide the signal into $L$ segments of length $M$ with overlap $D$
   (commonly 50%).
2. Apply a window $w[n]$ to each segment.
3. Compute the periodogram of each windowed segment.
4. Average the $L$ periodograms.

The variance reduction factor is approximately $L$ (depending on the
overlap and window shape), at the cost of reduced frequency resolution
since each segment is shorter than the full record.

$$
P_{\text{Welch}}[k] = \frac{1}{L} \sum_{i=0}^{L-1} P_w^{(i)}[k]
$$

## Library API

### Basic periodogram

```cpp
#include <sw/dsp/spectral/psd.hpp>

using namespace sw::dsp;

mtl::vec::dense_vector<double> signal(4096);
// ... fill signal ...

// Raw periodogram (no windowing)
auto P = spectral::periodogram(signal);
```

### Welch's method

```cpp
// Welch PSD with Hann window, 256-sample segments, 50% overlap
auto psd = spectral::welch(signal, 256, 0.5, window::hann);
```

### Decibel conversion

The `psd_db()` utility converts a linear PSD to decibels relative to a
reference level:

```cpp
// Convert to dB (reference = 1.0 by default)
auto psd_log = spectral::psd_db(psd);

// Custom reference level
auto psd_log_ref = spectral::psd_db(psd, 1e-12);
```

The conversion applies the standard formula:

$$
P_{\text{dB}}[k] = 10 \log_{10}\!\left(\frac{P[k]}{P_{\text{ref}}}\right)
$$

### Mixed-precision PSD

Accumulation during the FFT and the averaging step benefit from extra
precision even when the input samples are narrow:

```cpp
using Sample = sw::universal::posit<16, 2>;
using State  = double;

mtl::vec::dense_vector<Sample> signal(4096);
// ...

auto psd = spectral::welch<State, Sample>(signal, 256, 0.5, window::hann);
```

## Choosing parameters

| Parameter       | Effect of increasing             |
|-----------------|----------------------------------|
| Segment length  | Better frequency resolution      |
| Overlap         | More segments, lower variance    |
| Window sidelobe | Less spectral leakage            |

A common starting point is 50% overlap with a Hann window and segment
length chosen so that the frequency resolution $\Delta f = f_s / M$
meets your application requirements.
