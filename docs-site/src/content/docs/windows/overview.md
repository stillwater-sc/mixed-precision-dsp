---
title: Window Functions
description: Spectral leakage mitigation through windowing with standard window definitions
---

## Why Windows Are Needed

A discrete Fourier transform (DFT) operates on a finite block of $N$
samples. Extracting this block from a longer signal is equivalent to
multiplying by a rectangular window. The DFT of the rectangular window
is a **sinc-like function** with significant side lobes, causing energy
from one frequency bin to leak into neighbouring bins -- a phenomenon
called **spectral leakage**.

Applying a tapered window function $w[n]$ before the DFT smoothly
reduces the signal toward zero at the block edges, suppressing side lobes
at the cost of widening the main lobe.

### Windowed DFT

The DFT of a windowed signal is the convolution of the signal spectrum
$X(k)$ with the window spectrum $W(k)$:

$$
Y(k) = \sum_{n=0}^{N-1} x[n] \, w[n] \, e^{-j 2\pi k n / N}
= (X * W)(k)
$$

## Window Properties

Every window is characterized by a set of trade-off metrics:

| Property | Definition |
|----------|-----------|
| **Main lobe width** | Width of the central peak in frequency bins; determines frequency resolution |
| **Peak side lobe level** | Height of the largest side lobe relative to the main lobe (dB) |
| **Side lobe rolloff** | Rate at which side lobes decay away from the main lobe (dB/octave) |
| **Scalloping loss** | Maximum reduction in detected amplitude when a tone falls between two bins |
| **Processing gain** | Coherent gain relative to the rectangular window |

Narrower main lobes resolve closely spaced frequencies but let more
leakage through the side lobes. The choice of window is always a
compromise between these two goals.

## Standard Window Definitions

All windows below are defined for $0 \leq n \leq N-1$.

### Rectangular

$$
w[n] = 1
$$

Best frequency resolution (narrowest main lobe) but worst side lobe
level ($-13\,\text{dB}$). Suitable only when the signal is exactly
periodic in the analysis frame.

### Hamming

$$
w[n] = 0.54 - 0.46\cos\!\left(\frac{2\pi n}{N-1}\right)
$$

Optimized to **cancel the first side lobe**, yielding a peak side lobe
of $-43\,\text{dB}$ with a main lobe width of approximately 4 bins.

### Hanning (Hann)

$$
w[n] = 0.5\left(1 - \cos\!\left(\frac{2\pi n}{N-1}\right)\right)
$$

Also known as the raised cosine window. Side lobes roll off at
$-18\,\text{dB/octave}$, making it a good general-purpose choice.
Peak side lobe is $-32\,\text{dB}$.

### Blackman

$$
w[n] = 0.42 - 0.5\cos\!\left(\frac{2\pi n}{N-1}\right) + 0.08\cos\!\left(\frac{4\pi n}{N-1}\right)
$$

Three-term cosine sum with peak side lobes at $-58\,\text{dB}$.
The wider main lobe (6 bins) trades resolution for excellent leakage
rejection.

### Kaiser

$$
w[n] = \frac{I_0\!\left(\beta\sqrt{1 - \left(\frac{2n}{N-1} - 1\right)^2}\right)}{I_0(\beta)}
$$

where $I_0$ is the zeroth-order modified Bessel function of the first kind.
The parameter $\beta$ continuously controls the main lobe / side lobe
trade-off:

| $\beta$ | Peak side lobe | Approximate equivalent |
|---------|---------------|----------------------|
| 0 | $-13$ dB | Rectangular |
| 5 | $-36$ dB | Hamming |
| 8.6 | $-60$ dB | Blackman-class |
| 14 | $-90$ dB | High-dynamic-range |

### Flat-Top

$$
w[n] = a_0 - a_1\cos\!\left(\frac{2\pi n}{N-1}\right) + a_2\cos\!\left(\frac{4\pi n}{N-1}\right) - a_3\cos\!\left(\frac{6\pi n}{N-1}\right) + a_4\cos\!\left(\frac{8\pi n}{N-1}\right)
$$

with coefficients $a_0 = 0.2156$, $a_1 = 0.4160$, $a_2 = 0.2781$,
$a_3 = 0.0836$, $a_4 = 0.0069$.

Designed for **amplitude accuracy**: the scalloping loss is nearly zero
($< 0.01\,\text{dB}$), making it ideal for calibration and instrument
measurement at the expense of a very wide main lobe.

### Tukey (Cosine-Tapered)

$$
w[n] = \begin{cases}
\frac{1}{2}\left[1 + \cos\!\left(\frac{2\pi}{\alpha}\left(\frac{n}{N-1} - \frac{\alpha}{2}\right)\right)\right] & 0 \leq \frac{n}{N-1} < \frac{\alpha}{2} \\
1 & \frac{\alpha}{2} \leq \frac{n}{N-1} \leq 1 - \frac{\alpha}{2} \\
\frac{1}{2}\left[1 + \cos\!\left(\frac{2\pi}{\alpha}\left(\frac{n}{N-1} - 1 + \frac{\alpha}{2}\right)\right)\right] & 1 - \frac{\alpha}{2} < \frac{n}{N-1} \leq 1
\end{cases}
$$

The parameter $\alpha \in [0, 1]$ controls the fraction of the window
that is tapered. $\alpha = 0$ gives the rectangular window; $\alpha = 1$
gives the Hanning window. Useful when a flat passband with controlled
edge tapering is needed.

### Gaussian

$$
w[n] = \exp\!\left(-\frac{1}{2}\left(\frac{n - (N\!-\!1)/2}{\sigma\,(N\!-\!1)/2}\right)^2\right)
$$

The parameter $\sigma$ controls the width: smaller values give narrower
windows with more attenuation at the edges. The Gaussian window has no
side lobes in the continuous case but achieves only modest side lobe
suppression in the discrete DFT.

### Dolph-Chebyshev

Designed so that all side lobes are exactly equal at a specified
attenuation level. This is optimal in the minimax sense: for a given
side lobe level, no other window achieves a narrower main lobe.

The window is constructed in the frequency domain using Chebyshev
polynomials and transformed to the time domain via inverse DFT.
Parameter: `attenuation_db` (default 100 dB).

### Bartlett-Hann

$$
w[n] = 0.62 - 0.48\left|\frac{n}{N-1} - 0.5\right| + 0.38\cos\!\left(2\pi\left(\frac{n}{N-1} - 0.5\right)\right)
$$

A hybrid between the Bartlett (triangular) and Hann windows. Provides
moderate side lobe rejection with good side lobe rolloff.

## Library API

Each window function returns a `mtl::vec::dense_vector<T>` of length $N$:

```cpp
#include <sw/dsp/windows/windows.hpp>

using Scalar = double;
constexpr std::size_t N = 1024;

auto rect    = sw::dsp::rectangular_window<Scalar>(N);
auto ham     = sw::dsp::hamming_window<Scalar>(N);
auto hann    = sw::dsp::hanning_window<Scalar>(N);
auto black   = sw::dsp::blackman_window<Scalar>(N);
auto kai     = sw::dsp::kaiser_window<Scalar>(N, 8.6);
auto flat    = sw::dsp::flat_top_window<Scalar>(N);
auto tuk     = sw::dsp::tukey_window<Scalar>(N, 0.5);
auto gauss   = sw::dsp::gaussian_window<Scalar>(N, 0.4);
auto cheby   = sw::dsp::dolph_chebyshev_window<Scalar>(N, 100.0);
auto bh      = sw::dsp::bartlett_hann_window<Scalar>(N);
```

### Applying a Window

```cpp
mtl::vec::dense_vector<Scalar> signal = /* ... */;
auto windowed = sw::dsp::apply_window(signal, ham);
```

The `apply_window` function performs element-wise multiplication and
returns a new vector of the same length.

## Precision Considerations

Window coefficients are typically computed once and reused. Because the
cosine sums involve subtractions of nearly equal values, computing them
in `double` or a high-precision posit avoids coefficient errors that
would raise the effective side lobe floor.
