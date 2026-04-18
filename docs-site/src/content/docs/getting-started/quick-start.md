---
title: Quick Start
description: Design a filter and process signals in under 50 lines of C++
---

This guide walks through the core workflow: design a filter, process a signal,
and explore mixed-precision arithmetic types.

## Include headers

The umbrella header brings in the entire library:

```cpp
#include <sw/dsp/dsp.hpp>
```

For faster compile times, include only what you need:

```cpp
#include <sw/dsp/filter/iir/butterworth.hpp>
#include <sw/dsp/filter/filter.hpp>
#include <sw/dsp/signals/generators.hpp>
```

## Minimal example

Design a 4th-order Butterworth lowpass filter at 1 kHz and filter a signal:

```cpp
#include <sw/dsp/filter/iir/butterworth.hpp>
#include <sw/dsp/filter/filter.hpp>
#include <sw/dsp/signals/generators.hpp>
#include <iostream>
#include <cmath>

using namespace sw::dsp;

int main() {
    constexpr double sample_rate = 44100.0;
    constexpr double cutoff      = 1000.0;
    constexpr int    num_samples = 256;

    // Design a 4th-order Butterworth lowpass filter
    SimpleFilter<iir::ButterworthLowPass<4>> filter;
    filter.setup(4, sample_rate, cutoff);

    // Generate a test signal: 400 Hz (passes) + 8000 Hz (rejected)
    auto tone_low  = sine<double>(num_samples, 400.0,  sample_rate);
    auto tone_high = sine<double>(num_samples, 8000.0, sample_rate);

    // Filter sample by sample
    for (int n = 0; n < num_samples; ++n) {
        double input  = 0.5 * tone_low[n] + 0.5 * tone_high[n];
        double output = filter.process(input);
        std::cout << n << "  in=" << input << "  out=" << output << "\n";
    }

    return 0;
}
```

The `SimpleFilter` wrapper combines filter design (the cascade of biquad
coefficients) with processing state, so you can call `process()` directly.

## The three-scalar model

Every algorithm in the library is parameterized on three independent
arithmetic types:

| Role | Template parameter | Purpose |
|------|--------------------|---------|
| **Coefficients** | `CoeffScalar` | Precision used to design and store filter coefficients |
| **State** | `StateScalar` | Precision used for internal accumulation during processing |
| **Samples** | `SampleScalar` | Precision of input and output sample streams |

These map directly to the template parameters of each filter class:

```cpp
// Template: ButterworthLowPass<MaxOrder, CoeffScalar, StateScalar, SampleScalar>
//
// Design in double, accumulate in float, stream in float:
iir::ButterworthLowPass<4, double, float, float> filter;
filter.setup(4, 44100.0, 1000.0);
```

Wrapping in `SimpleFilter` adds per-sample processing:

```cpp
SimpleFilter<iir::ButterworthLowPass<4, double, float, float>> filter;
filter.setup(4, 44100.0, 1000.0);

float y = filter.process(0.5f);  // SampleScalar is float
```

When all three scalars default to `double`, the template parameters can be
omitted:

```cpp
SimpleFilter<iir::ButterworthLowPass<4>> filter;  // all double
```

## Using Universal number types

The library works with any type that satisfies the `DspField` or `DspScalar`
concepts, including types from the
[Universal](https://github.com/stillwater-sc/universal) number systems library.

### Custom floating-point with cfloat

`cfloat<nbits, es>` defines a custom-width IEEE-style float. Use it for
state and samples while keeping coefficient design in `double`:

```cpp
#include <sw/dsp/filter/iir/butterworth.hpp>
#include <sw/dsp/filter/filter.hpp>
#include <sw/universal/number/cfloat/cfloat.hpp>
#include <iostream>

using namespace sw::dsp;

int main() {
    // 24-bit custom float with 5-bit exponent and subnormal support
    using cf24 = sw::universal::cfloat<24, 5, uint32_t, true, false, false>;

    // Coefficients: double | State: cf24 | Samples: cf24
    SimpleFilter<iir::ButterworthLowPass<4, double, cf24, cf24>> filter;
    filter.setup(4, 44100.0, 1000.0);

    cf24 x{0.5};
    cf24 y = filter.process(x);

    std::cout << "Input:  " << double(x) << "\n";
    std::cout << "Output: " << double(y) << "\n";

    return 0;
}
```

### Fixed-point with fixpnt

`fixpnt<nbits, frac_bits>` provides fixed-point arithmetic:

```cpp
#include <sw/universal/number/fixpnt/fixpnt.hpp>

using fp16 = sw::universal::fixpnt<16, 14>;  // 16-bit, 14 fractional bits

SimpleFilter<iir::ButterworthLowPass<4, double, fp16, fp16>> filter;
filter.setup(4, 44100.0, 1000.0);

fp16 y = filter.process(fp16{0.25});
```

### IEEE half-precision

Universal provides a built-in `half` type (IEEE 754 binary16):

```cpp
#include <sw/universal/number/cfloat/cfloat.hpp>

using half = sw::universal::half;

SimpleFilter<iir::ButterworthLowPass<4, double, half, half>> filter;
filter.setup(4, 44100.0, 1000.0);
```

## Inspecting filter coefficients

Access the biquad cascade directly to examine coefficients or compute
frequency response:

```cpp
iir::ButterworthLowPass<4> filter;
filter.setup(4, 44100.0, 1000.0);

const auto& cascade = filter.cascade();
std::cout << "Number of biquad stages: " << cascade.num_stages() << "\n";

for (int i = 0; i < cascade.num_stages(); ++i) {
    const auto& c = cascade.stage(i);
    std::cout << "Stage " << i
              << ": b0=" << c.b0 << " b1=" << c.b1 << " b2=" << c.b2
              << " a1=" << c.a1 << " a2=" << c.a2 << "\n";
}
```

## Complete compilable example

This program designs a Butterworth lowpass filter, processes a mixed-frequency
signal, and prints the before/after peak amplitudes:

```cpp
#include <sw/dsp/filter/iir/butterworth.hpp>
#include <sw/dsp/filter/filter.hpp>
#include <sw/dsp/signals/generators.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace sw::dsp;

int main() {
    constexpr double fs      = 48000.0;   // sample rate
    constexpr double fc      = 2000.0;    // cutoff frequency
    constexpr int    order   = 4;
    constexpr int    N       = 1024;

    // Design the filter
    SimpleFilter<iir::ButterworthLowPass<8>> filter;
    filter.setup(order, fs, fc);

    // Print cascade info
    std::cout << "Butterworth LP order " << order
              << ", cutoff " << fc << " Hz"
              << ", " << filter.cascade().num_stages() << " biquad stages\n";

    // Generate input: 500 Hz (in passband) + 6000 Hz (in stopband)
    auto low  = sine<double>(N, 500.0,  fs);
    auto high = sine<double>(N, 6000.0, fs);

    std::vector<double> input(N), output(N);
    for (int n = 0; n < N; ++n) {
        input[n] = 0.5 * low[n] + 0.5 * high[n];
    }

    // Process
    for (int n = 0; n < N; ++n) {
        output[n] = filter.process(input[n]);
    }

    // Measure peak amplitudes (skip first 200 samples for transient)
    double peak_in = 0, peak_out = 0;
    for (int n = 200; n < N; ++n) {
        peak_in  = std::max(peak_in,  std::abs(input[n]));
        peak_out = std::max(peak_out, std::abs(output[n]));
    }

    std::cout << "Peak input amplitude:  " << peak_in  << "\n";
    std::cout << "Peak output amplitude: " << peak_out << "\n";
    std::cout << "The 6 kHz component is attenuated; "
              << "the 500 Hz component passes through.\n";

    return 0;
}
```

## Next steps

- Browse the filter types: `ButterworthHighPass`, `ButterworthBandPass`,
  `ButterworthBandStop`, `ButterworthLowShelf`, `ButterworthHighShelf`
- Try Chebyshev or Elliptic designs with the same three-scalar model
- Use `TransposedDirectFormII` for better numerical behavior:
  ```cpp
  SimpleFilter<iir::ButterworthLowPass<4>,
               TransposedDirectFormII<double>> filter;
  ```
- Explore the `analysis` module for stability and sensitivity metrics
