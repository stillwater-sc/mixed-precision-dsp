---
title: Polyphase Decimator
description: Efficient FIR-based decimation through polyphase decomposition for high-rate acquisition pipelines
---

*Part of the [Multirate Signal Processing](../multirate/overview/) section — see the overview for the broader theory and the [pattern catalog](../multirate/patterns/) for problem→API mapping.*

## What and Why

A polyphase decimator implements **decimation by M** with an N-tap FIR
prototype, but achieves the result with roughly $N$ multiplies per
output sample instead of the naive $N \cdot M$. It is the
channel-shaping stage in the canonical high-rate acquisition chain:

```text
ADC -> CIC ↓R1 -> half-band ↓2 -> half-band ↓2 -> polyphase FIR ↓Rn -> baseband
```

By the time the signal reaches the polyphase FIR, the rate has been
reduced enough that a sharp, fully-shaped FIR is affordable. The
polyphase form ensures we don't pay for the $M-1$ output samples we
throw away.

## How It Works

The polyphase identity rewrites a single $N$-tap FIR followed by
downsample-by-$M$ as $M$ sub-filters of length $\lceil N / M \rceil$,
each operating on every $M$-th input sample. Sub-tap arrays are
formed by

$$
h_q[p] \;=\; h[p \cdot M + q], \qquad q \in [0, M),\; p \in [0, \lceil N/M \rceil).
$$

A commutator routes incoming samples to sub-filter $q = (M - r) \bmod M$
where $r$ is the input index modulo $M$. When $r = 0$ all $M$ sub-filter
outputs are summed to produce one output sample.

### Computational Savings

| | Naive filter-then-downsample | Polyphase |
|---|---|---|
| Multiplies per **input** sample | $N$ | $N/M$ on average |
| Multiplies per **output** sample | $N \cdot M$ | $N$ |
| Memory | $N$ taps + delay line | $N$ taps + delay line (same) |

The savings scale linearly with the decimation factor.

## Library API

The library exposes `PolyphaseDecimator` from two paths:

- **Acquisition module** (recommended for DDC/decimation-chain workflows):
  ```cpp
  #include <sw/dsp/acquisition/polyphase_decimator.hpp>
  ```
- **FIR module** (for standalone multirate FIR work):
  ```cpp
  #include <sw/dsp/filter/fir/polyphase.hpp>
  ```

Both surface the same `sw::dsp::PolyphaseDecimator` class — the
acquisition header is a convenience shim that pulls it in via
transitive include and frames its role as a high-rate acquisition
component.

### Three-Scalar Parameterization

```cpp
template <DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspScalar SampleScalar = StateScalar>
class PolyphaseDecimator;
```

| Template parameter | Purpose |
|---|---|
| `CoeffScalar`  | Tap coefficients (filter design precision) |
| `StateScalar`  | Delay-line accumulation precision |
| `SampleScalar` | Input/output samples |

### Construction

```cpp
#include <sw/dsp/acquisition/polyphase_decimator.hpp>
#include <sw/dsp/filter/fir/fir_design.hpp>
#include <sw/dsp/windows/hamming.hpp>

using namespace sw::dsp;

// Design the prototype FIR at the input rate (factor must satisfy the
// anti-alias condition: passband well below 0.5 / factor)
const std::size_t num_taps = 65;
const std::size_t factor   = 4;

auto win  = hamming_window<double>(num_taps);
auto taps = design_fir_lowpass<double>(num_taps, 0.45 / factor, win);

PolyphaseDecimator<double> decim(taps, factor);
```

### Streaming

```cpp
for (std::size_t n = 0; n < input_size; ++n) {
    auto [ready, y] = decim.process(input[n]);
    if (ready) sink(y);  // emits once per `factor` inputs
}
```

`process` returns `{true, y}` exactly when the input index is a
multiple of the decimation factor. The first emit corresponds to input
index 0.

### Block Processing

```cpp
auto output = decim.process_block(std::span<const double>(input.data(), input.size()));
// output.size() == count of multiples of `factor` in [phase_at_entry, ...)
```

### Reset

```cpp
decim.reset();   // clears the delay lines and resets the commutator phase
```

### Querying the Decomposition

The same FIR module exposes a free function that returns the polyphase
sub-tap matrix without instantiating a runtime decimator. Useful for
pre-quantization analysis or pre-loading hardware coefficient memories:

```cpp
auto sub_filters = polyphase_decompose(taps, factor);
// sub_filters[q][p] == taps[p*factor + q] (zero-padded at the tail)
```

`polyphase_decompose` throws `std::invalid_argument` if `factor == 0`
or `taps` is empty.

## Mixed-Precision Use

`PolyphaseDecimator` is parameterized end-to-end on the caller's
scalar type. A typical embedded deployment might design taps in
`double` for accuracy, store them and accumulate in `posit<32, 2>` for
tapered precision near unity, and stream samples through as
`posit<16, 2>` for memory and bandwidth savings:

```cpp
using coeff_t  = double;
using state_t  = sw::universal::posit<32, 2>;
using sample_t = sw::universal::posit<16, 2>;

PolyphaseDecimator<coeff_t, state_t, sample_t> decim(taps, factor);
```

The library's regression tests verify polyphase output bit-exactly
matches direct FIR-then-downsample at `double`, and within ULP-scale
tolerance at `posit<32, 2>` and `cfloat<32, 8>`.

## Composition

This decimator drops in directly to the chain abstractions:

- **`DDC`** ([Digital Down-Converter](./ddc/)) — used as the I/Q
  decimation stage after the NCO mixer. Two parallel copies (one for
  I, one for Q) run in lockstep.
- **`DecimationChain`** ([Multi-Stage Decimation](./decimation-chain/))
  — composed with CIC and half-band stages via the
  `step_decimator` adapter. The `process` method matches the
  `{ready, y}` streaming protocol the chain expects.

See the [Multi-Stage Decimation Chain](./decimation-chain/) page for
the full CIC → half-band → polyphase shaping example.

## Historical References

- M. Bellanger, G. Bonnerot, M. Coudreuse, *"Digital Filtering by
  Polyphase Network: Application to Sample-Rate Alteration and
  Filter Banks,"* IEEE Transactions on ASSP, vol. 24, no. 2, 1976 —
  the polyphase identity for filter banks.
- R. E. Crochiere and L. R. Rabiner, *Multirate Digital Signal
  Processing*, Prentice Hall, 1983 — definitive treatment of
  polyphase decimation/interpolation.
- F. J. Harris, *Multirate Signal Processing for Communication
  Systems*, Prentice Hall, 2004 — modern reference; chapter on
  polyphase form motivates the M-fold computational savings.
