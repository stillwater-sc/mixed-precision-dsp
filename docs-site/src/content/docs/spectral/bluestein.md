---
title: Arbitrary-Length FFT (Bluestein)
description: Bluestein chirp-z algorithm for O(N log N) DFT of any length
---

The Cooley-Tukey radix-2 FFT requires the input length to be a power of
two. In practice, signal lengths are dictated by hardware constraints,
protocol frame sizes, or measurement durations that rarely align to
powers of two. Zero-padding to the next power of two wastes memory and
computation; truncating discards data.

**Bluestein's algorithm** (1970) computes the DFT of any length $N$ in
$O(N \log N)$ time by reformulating it as a circular convolution, which
is then computed using three power-of-two FFTs.

## The chirp-z identity

The key insight is an algebraic identity on the DFT kernel exponent:

$$
kn = \frac{1}{2}\bigl[k^2 + n^2 - (k - n)^2\bigr]
$$

Substituting into the DFT definition:

$$
X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-j2\pi kn/N}
     = e^{-j\pi k^2/N} \sum_{n=0}^{N-1}
       \bigl[x[n]\, e^{-j\pi n^2/N}\bigr]\,
       e^{j\pi(k-n)^2/N}
$$

This has the form of a **convolution** of the chirp-modulated input with
a chirp sequence, followed by a chirp modulation of the output. The
convolution can be computed via FFT of a zero-padded, power-of-two
length.

## Algorithm steps

Given input $x[n]$ of length $N$:

1. **Chirp sequence.** Compute $w[n] = e^{-j\pi n^2/N}$ for
   $n = 0, \ldots, N-1$.

2. **Modulate input.** Form $a[n] = x[n] \cdot w[n]$, zero-padded to
   length $M = 2^{\lceil\log_2(2N-1)\rceil}$.

3. **Convolution kernel.** Form $b[n] = \overline{w[n]}$ with
   wrap-around: $b[M-n] = \overline{w[n]}$ for $n = 1, \ldots, N-1$.

4. **Circular convolution.** Compute $C = \text{IFFT}(\text{FFT}(a)
   \cdot \text{FFT}(b))$.

5. **Demodulate.** Extract $X[k] = w[k] \cdot C[k]$ for
   $k = 0, \ldots, N-1$.

The inverse DFT uses the conjugate-transform-conjugate-scale approach:
conjugate the input, apply the forward Bluestein transform, conjugate
the result, and scale by $1/N$.

## API

### Auto-dispatching (recommended)

```cpp
#include <sw/dsp/spectral/bluestein.hpp>

// Automatically selects radix-2 FFT or Bluestein based on length
auto X = sw::dsp::spectral::dft_forward<double>(x);
auto x = sw::dsp::spectral::dft_inverse<double>(X);
```

The `dft_forward` and `dft_inverse` functions check whether the input
length is a power of two. If so, they use the in-place radix-2 FFT. For
all other lengths, they dispatch to Bluestein's algorithm.

### Direct Bluestein

```cpp
auto X = sw::dsp::spectral::bluestein_forward<double>(x);
auto x = sw::dsp::spectral::bluestein_inverse<double>(X);
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` / `X` | `dense_vector<complex_for_t<T>>` | Input (time or frequency domain) |
| **Returns** | `dense_vector<complex_for_t<T>>` | Transformed output |

The template parameter `T` must satisfy `DspField` and must not be
integral.

## Performance

| Length $N$ | Method | FFT size $M$ | Cost |
|------------|--------|-------------|------|
| 1024 | Radix-2 | 1024 | $N \log N$ |
| 1000 | Bluestein | 2048 | $\sim 3 M \log M$ |
| 997 (prime) | Bluestein | 2048 | $\sim 3 M \log M$ |

Bluestein is roughly 3x slower than a radix-2 FFT of the same padded
length due to the three FFT operations. However, it is still
$O(N \log N)$ -- dramatically better than the $O(N^2)$ naive DFT.

## Example: spectrum of a 1000-sample signal

```cpp
#include <sw/dsp/spectral/bluestein.hpp>
#include <sw/dsp/signals/generators.hpp>

auto signal = sw::dsp::sine<double>(1000, 50.0, 1000.0);

// Convert to complex
using complex_t = sw::dsp::complex_for_t<double>;
mtl::vec::dense_vector<complex_t> x(1000);
for (std::size_t i = 0; i < 1000; ++i)
    x[i] = complex_t(signal[i], 0.0);

// dft_forward auto-dispatches to Bluestein for N=1000
auto X = sw::dsp::spectral::dft_forward<double>(x);
```

## Precision considerations

Bluestein's algorithm involves three FFTs and several complex
multiplications with chirp sequences. Each operation introduces
rounding error. For long transforms, the accumulated error grows as
$O(\sqrt{N \log N})$ in floating-point arithmetic.

The chirp sequence $e^{-j\pi n^2/N}$ evaluates trigonometric functions
at angles that grow quadratically. For large $N$, the argument
$\pi n^2 / N$ can be very large, and standard `sin`/`cos` lose
precision for large arguments. Using a high-precision type for the chirp
computation can improve transform accuracy.
