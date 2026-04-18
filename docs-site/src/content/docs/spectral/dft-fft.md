---
title: DFT and FFT
description: Discrete Fourier Transform and the Fast Fourier Transform algorithm in mixed-precision DSP
---

The Discrete Fourier Transform (DFT) converts a finite-length time-domain
sequence into its frequency-domain representation. The Fast Fourier Transform
(FFT) is an algorithm that computes the DFT in $O(N \log N)$ operations
instead of the naive $O(N^2)$.

## DFT definition

Given a length-$N$ sequence $x[n]$, the DFT is defined as:

$$
X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-j2\pi kn/N}, \quad k = 0, 1, \dots, N-1
$$

Each output bin $X[k]$ is the inner product of the input with a complex
sinusoid at frequency $f_k = k \cdot f_s / N$, where $f_s$ is the sampling
rate.

### Inverse DFT

The original sequence is recovered by the inverse DFT (IDFT):

$$
x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k]\, e^{j2\pi kn/N}, \quad n = 0, 1, \dots, N-1
$$

The factor $1/N$ normalizes the round-trip so that $\text{IDFT}(\text{DFT}(x)) = x$.

## Cooley-Tukey radix-2 FFT

The Cooley-Tukey algorithm exploits the symmetry and periodicity of the
complex exponential $W_N = e^{-j2\pi/N}$. For a power-of-two length $N$
the DFT splits recursively into two half-length DFTs:

$$
X[k] = \underbrace{\sum_{m=0}^{N/2-1} x[2m]\, W_{N/2}^{mk}}_{\text{even samples}}
      + W_N^k \underbrace{\sum_{m=0}^{N/2-1} x[2m+1]\, W_{N/2}^{mk}}_{\text{odd samples}}
$$

This decomposition reduces the operation count from $O(N^2)$ to
$O(N \log_2 N)$, a dramatic improvement for large transforms.

## Zero-padding for frequency resolution

Appending zeros to a signal before computing the DFT increases the number
of frequency bins without adding new information. This interpolates the
spectrum, making peaks easier to locate visually. The intrinsic frequency
resolution remains $\Delta f = f_s / N_{\text{original}}$, but the bin
spacing becomes $f_s / N_{\text{padded}}$.

## Library API

The `sw::dsp` namespace provides both the direct DFT and the FFT:

```cpp
#include <sw/dsp/spectral/fft.hpp>

using namespace sw::dsp;

// Create a real-valued test signal
mtl::vec::dense_vector<double> signal(1024);
// ... fill signal with samples ...

// Compute the FFT (returns complex spectrum)
auto X = spectral::fft(signal);

// Compute the inverse FFT to recover the time-domain signal
auto recovered = spectral::ifft(X);
```

For non-power-of-two lengths, or when you want the reference
implementation, use the direct DFT:

```cpp
// Direct DFT -- O(N^2) but works for any length
auto X_direct = spectral::dft(signal);
auto x_back   = spectral::idft(X_direct);
```

### Mixed-precision transforms

The three-scalar parameterization applies to spectral analysis as well.
Twiddle factors (the $W_N^k$ constants) can be stored in a high-precision
coefficient type while the input samples use a narrower streaming type:

```cpp
#include <sw/universal/number/posit/posit.hpp>

using Coeff  = double;                                // twiddle factors
using Sample = sw::universal::posit<16, 2>;           // input samples
using State  = double;                                // accumulator

auto X = spectral::fft<Coeff, State, Sample>(signal);
```

This keeps accumulation accurate while reducing memory bandwidth for the
input data path.

### Interpreting the output

The FFT of a real signal of length $N$ produces $N$ complex bins. Due to
Hermitian symmetry, only the first $N/2 + 1$ bins carry unique information.
Bin $k$ corresponds to frequency:

$$
f_k = \frac{k \cdot f_s}{N}
$$

To obtain a magnitude spectrum in decibels:

```cpp
auto mag_db = spectral::magnitude_db(X);
```

## Performance notes

| Length $N$ | DFT multiplies | FFT multiplies |
|-----------|---------------|----------------|
| 256       | 65 536        | 1 024          |
| 1 024     | 1 048 576     | 5 120          |
| 4 096     | 16 777 216    | 24 576         |

For real-time applications the FFT is the only practical choice. The
library's implementation uses in-place butterfly operations and
bit-reversal permutation for cache-friendly access.
