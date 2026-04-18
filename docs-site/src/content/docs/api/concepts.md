---
title: Concepts
description: C++20 concepts that constrain types across the mixed-precision DSP library
---

The library uses C++20 concepts to enforce type requirements at compile time.
This lets you write generic DSP algorithms that work with `double`, `float`,
and Universal number types (posit, cfloat, fixpnt, lns) while catching type
mismatches before any code is generated.

## Scalar concepts

### `DspField<T>`

The primary constraint for DSP computation types. A type satisfies `DspField`
when it supports the four arithmetic operators (`+`, `-`, `*`, `/`),
comparison, and round-trip conversion through `double` via `static_cast`.

```cpp
namespace sw::dsp {
template <typename T>
concept DspField = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
    { a - b } -> std::convertible_to<T>;
    { a * b } -> std::convertible_to<T>;
    { a / b } -> std::convertible_to<T>;
    { a < b } -> std::convertible_to<bool>;
    { static_cast<double>(a) } -> std::same_as<double>;
    { static_cast<T>(1.0) } -> std::same_as<T>;
};
}
```

Satisfied by `float`, `double`, and Universal types such as
`sw::universal::posit<32,2>`, `sw::universal::cfloat<16,5>`,
`sw::universal::fixpnt<16,8>`, and `sw::universal::lns<16,8>`.

### `DspScalar<T>`

A superset of `DspField` that also admits integer-like types which may not
support division. Use this when an algorithm only needs addition,
subtraction, and multiplication (e.g., FIR dot products with integer
coefficients).

### `ConvertibleToDouble<T>`

A narrower constraint requiring only that `static_cast<double>(t)` is valid.
The analysis module (stability, sensitivity, condition number) uses this
concept because it converts all values to `double` for numerical computation
regardless of the original arithmetic type.

## Complex type dispatch

### `complex_for_t<T>`

A type trait -- not a concept -- that selects the correct complex wrapper for
a given scalar:

| Scalar `T`                       | `complex_for_t<T>`                  |
|----------------------------------|--------------------------------------|
| `float`, `double`                | `std::complex<T>`                    |
| Universal types (posit, cfloat)  | `sw::universal::complex<T>`          |

All library code uses `complex_for_t<T>` instead of `std::complex<T>`
directly so that pole/zero computations, FFTs, and frequency response
evaluations work transparently with any arithmetic back-end.

```cpp
template <DspField T>
auto compute_poles(T a1, T a2) {
    using Complex = complex_for_t<T>;
    // discriminant, quadratic formula, etc.
}
```

## Filter concepts

### `FilterDesign<D>`

The base concept for any filter design object. It requires a single method
returning a const reference to the biquad cascade:

```cpp
template <typename D>
concept FilterDesign = requires(const D& d) {
    { d.cascade() } -> std::same_as<const typename D::CascadeType&>;
};
```

### `DesignableLowPass<D>`

Extends `FilterDesign`. A type satisfies this concept when it can be
configured as a low-pass filter with order, sample rate, and cutoff
frequency:

```cpp
template <typename D>
concept DesignableLowPass = FilterDesign<D> && requires(D d) {
    d.setup(unsigned{}, double{}, double{});  // order, sample_rate, cutoff
};
```

### `DesignableBandPass<D>`

Extends `FilterDesign`. Requires a four-argument `setup` for band-pass
configuration:

```cpp
template <typename D>
concept DesignableBandPass = FilterDesign<D> && requires(D d) {
    d.setup(unsigned{}, double{}, double{}, double{});  // order, fs, center, width
};
```

### `Processable<D>`

A filter that can accept samples and produce output. Requires `process()` and
`reset()`:

```cpp
template <typename D>
concept Processable = requires(D d, double x) {
    { d.process(x) } -> std::convertible_to<double>;
    d.reset();
};
```

## Writing generic algorithms with concepts

Concepts let you write functions that accept any qualifying type without
manual overloading or SFINAE:

```cpp
#include <sw/dsp/dsp.hpp>

template <DspField T>
auto compute_magnitude(const mtl::vec::dense_vector<T>& signal) {
    using std::abs;
    mtl::vec::dense_vector<T> mag(mtl::vec::size(signal));
    for (std::size_t i = 0; i < mtl::vec::size(signal); ++i) {
        mag[i] = abs(signal[i]);
    }
    return mag;
}

template <DesignableLowPass Design>
void apply_lowpass(Design& filt, unsigned order, double fs, double fc) {
    filt.setup(order, fs, fc);
    // filt.cascade() is now guaranteed to exist
}
```

Because `DspField` is satisfied by both native floating-point and Universal
types, the same `compute_magnitude` instantiation works for
`dense_vector<double>`, `dense_vector<posit<32,2>>`, or any other conforming
type -- the compiler verifies the contract at the call site and produces a
clear diagnostic if the type does not qualify.

## Mixed-precision parameterization

Library algorithms are typically parameterized on three independent scalar
types, each constrained by `DspField`:

```cpp
template <DspField CoeffScalar, DspField StateScalar, DspField SampleScalar>
class BiquadFilter { ... };
```

- **CoeffScalar** -- precision used for filter coefficients (design time).
- **StateScalar** -- precision used for accumulator state (processing).
- **SampleScalar** -- precision used for input/output samples (streaming).

This three-scalar design is the foundation of mixed-precision DSP: you can
store coefficients in a narrow posit, accumulate in double, and stream
samples as 16-bit cfloat, all within a single filter instance.
