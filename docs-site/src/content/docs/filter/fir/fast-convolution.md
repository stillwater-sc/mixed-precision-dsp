---
title: Fast Convolution
description: Overlap-add and overlap-save FFT-based convolution methods in sw::dsp
---

Direct FIR convolution costs $O(NM)$ operations to filter an $N$-sample
signal with an $M$-tap filter. For long signals and long filters, **FFT-based
convolution** reduces this to $O(N \log N)$ by exploiting the convolution
theorem:

$$
y = h * x \;\;\Longleftrightarrow\;\; Y(k) = H(k) \cdot X(k)
$$

Multiply in the frequency domain, then inverse-transform to get the time-domain
result.

## Block processing challenge

The FFT operates on fixed-length blocks, but real-time signals arrive as
continuous streams. Two classical methods handle this: **overlap-add** and
**overlap-save** (also called overlap-scrap).

## Overlap-add

### Algorithm

Given filter length $M$ and block size $L$:

1. Partition the input into non-overlapping blocks of length $L$.
2. Zero-pad each block and the filter to length $N \geq L + M - 1$.
3. Compute the $N$-point FFT of each block and of the filter.
4. Multiply spectra pointwise.
5. Inverse FFT to get a length-$N$ result.
6. **Add** the overlapping tails of adjacent blocks (the last $M - 1$ samples
   of each result overlap with the first $M - 1$ samples of the next).

### Why it works

Each block convolution produces a result of length $L + M - 1$. The overlap
region contains the transient from the previous block's tail, and adding
the contributions reconstructs the correct linear convolution.

### Diagram

```
Block k:    [----L----][0...0]     (zero-padded to N)
Block k+1:            [----L----][0...0]
Result k:   [----L----|--M-1--]
Result k+1:           [----L----|--M-1--]
                       ^^^^^^^^
                       overlap-add region
```

## Overlap-save

### Algorithm

1. Read blocks of length $N$ with an overlap of $M - 1$ samples from the
   previous block.
2. Compute the $N$-point FFT of each block and the filter.
3. Multiply spectra pointwise.
4. Inverse FFT.
5. **Discard** the first $M - 1$ samples of each result (they suffer from
   circular convolution wrap-around). The remaining $L = N - M + 1$ samples
   are correct.

### Why it works

The overlapping input ensures that the circular convolution's wrap-around
region only corrupts samples that came from the previous block. By discarding
those samples and taking them from the previous block's valid output instead,
we recover the correct linear convolution.

## Computational comparison

For a signal of length $N_{\text{sig}}$ and a filter of length $M$, using
FFT block size $N = 2^p$:

| Method | Operations |
|---|---|
| Direct convolution | $O(N_{\text{sig}} \cdot M)$ |
| FFT-based (per block) | $O(N \log N)$ |
| FFT-based (total) | $O\!\left(\frac{N_{\text{sig}}}{L} \cdot N \log N\right)$ |

The crossover point where FFT convolution becomes faster depends on $M$.
As a rule of thumb, FFT convolution wins when $M > 64$.

## Block size selection

The FFT length $N$ should be chosen as a power of two satisfying
$N \geq L + M - 1$ (overlap-add) or $N \geq 2M$ (overlap-save, for
reasonable efficiency). Larger $N$ amortizes the FFT overhead over more
output samples but increases latency and memory.

| FFT length $N$ | Filter taps $M$ | Output per block $L$ | Efficiency |
|---|---|---|---|
| 256 | 64 | 193 | 75% |
| 512 | 64 | 449 | 88% |
| 1024 | 64 | 961 | 94% |
| 1024 | 256 | 769 | 75% |
| 2048 | 256 | 1793 | 88% |

Efficiency here is $L / N$, the fraction of FFT output that is valid.

## Library API

### Overlap-add convolver

```cpp
#include <sw/dsp/filter/fir/fast_convolution.hpp>
using namespace sw::dsp;

// Create a convolver for a 256-tap filter with 1024-point FFTs
std::vector<double> h(256);  // filter coefficients
OverlapAddConvolver<double> conv(h, 1024);

// Process a block of input
std::vector<double> input(768);
std::vector<double> output(768);
conv.process(input, output);
```

### Overlap-save convolver

```cpp
OverlapSaveConvolver<double> conv(h, 1024);
conv.process(input, output);
```

### Free functions

For one-shot (non-streaming) convolution of complete signals:

```cpp
#include <sw/dsp/filter/fir/fast_convolution.hpp>
using namespace sw::dsp;

auto y = overlap_add(signal, filter_coeffs);
auto z = overlap_save(signal, filter_coeffs);
// Both return the full linear convolution result
```

These functions automatically choose an appropriate FFT size.

## Mixed-precision notes

FFT-based convolution involves two additional sources of numerical error
beyond the filter coefficients themselves:

1. **FFT round-off** -- each butterfly operation introduces rounding. For an
   $N$-point FFT, the accumulated error grows as $O(\log N)$ in
   floating-point and can be larger in narrow formats.
2. **Spectral multiplication** -- multiplying two complex spectra can produce
   values outside the dynamic range of narrow types.

For these reasons, the `OverlapAddConvolver` and `OverlapSaveConvolver`
perform all internal FFT and spectral arithmetic in the widest available
scalar type, converting back to the output type only at the final stage.
This matches the library's three-scalar philosophy: use wide `StateScalar`
for internal computation, narrow `SampleScalar` for I/O.
