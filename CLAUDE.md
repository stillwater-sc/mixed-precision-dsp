# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

Preferred entry points are the CMake presets in `CMakePresets.json`:

| Preset          | Build type | Tests | Apps | Regression | Binary dir              |
|-----------------|------------|-------|------|------------|-------------------------|
| `dev`           | Debug      | ON    | ON   | OFF        | `build/`                |
| `release`       | Release    | OFF   | OFF  | OFF        | `build-release/`        |
| `ci`            | Release    | ON    | ON   | OFF        | `build-ci/`             |
| `ci-regression` | Release    | ON    | OFF  | ON         | `build-ci-regression/`  |

`dev` also sets `CMAKE_EXPORT_COMPILE_COMMANDS=ON` for clangd.

```bash
# Configure via preset (FetchContent pulls Universal and MTL5 automatically)
cmake --preset dev
cmake --build build -j4
ctest --test-dir build --output-on-failure

# CI-style build
cmake --preset ci && cmake --build build-ci -j4
ctest --test-dir build-ci --output-on-failure

# Regression suite (slow — gated by DSP_BUILD_REGRESSION_TESTS)
cmake --preset ci-regression && cmake --build build-ci-regression -j4
ctest --test-dir build-ci-regression -L regression --output-on-failure

# Manual configure (no preset) — e.g. clang or cross-compile
cmake -B build_clang -DCMAKE_CXX_COMPILER=clang++ -Wno-dev
cmake --build build_clang -j4
ctest --test-dir build_clang --output-on-failure

# RISC-V cross-compile
cmake -B build_rv64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/riscv64-gcc.cmake -Wno-dev
cmake --build build_rv64 -j4
```

### CTest labels

Every test carries labels for selective runs:

- **`unit`** — attached to all unit tests (via `dsp_add_test()` in `tests/CMakeLists.txt`).
- **`regression`** — attached to tests under `tests/regression/` (via `dsp_add_regression_test()`).
- **Per-module** — first path segment of the folder argument, lowercased. Current modules: `acquisition`, `analysis`, `conditioning`, `estimation`, `filter`, `foundation`, `image`, `instrument`, `quantization`, `signals`, `spectral`, `spectrum`, `windows`.

Common invocations:

```bash
ctest --print-labels                # list every label in the current build
ctest -L unit                       # only unit tests
ctest -L regression                 # only regression tests (requires ci-regression preset)
ctest -LE regression                # everything except regression
ctest -L filter                     # only filter-module tests (IIR + FIR + generic)
```

The `-L` and `-LE` flags take extended regexes, so `ctest -L 'filter|spectrum'` selects tests matching either module.

## Architecture

Header-only C++20 library under `include/sw/dsp/`. Namespace: `sw::dsp`.

Three-scalar parameterization on every processing algorithm:
- `CoeffScalar` — filter coefficients (design precision)
- `StateScalar` — accumulator state (processing precision)
- `SampleScalar` — input/output samples (streaming precision)

Key type: `complex_for_t<T>` in `concepts/scalar.hpp` — dispatches to
`std::complex<T>` for native types, `sw::universal::complex<T>` for Universal
types. All library code uses this instead of `std::complex<T>` directly.

IIR filter pipeline: analog prototype → bilinear/Constantinides transform → cascade of biquads.

## Testing Rules

**Never use `assert()` in tests.** CI runs in Release mode where `NDEBUG` is
defined and `assert()` is stripped. All test checks must use explicit `if`
statements that throw `std::runtime_error` on failure:

```cpp
// WRONG — silent pass in Release
assert(value > 0);

// CORRECT — always executes
if (!(value > 0)) throw std::runtime_error("test failed: value > 0");
```

Every test `main()` should be wrapped in `try/catch` to report exceptions
cleanly.

## Umbrella Header

`#include <sw/dsp/dsp.hpp>` brings in the entire library. For faster
compile times, include individual module headers instead (e.g.,
`<sw/dsp/filter/iir/butterworth.hpp>`).

## Analysis Module

`analysis/stability.hpp` extracts poles from biquad coefficients by solving
`z^2 + a1*z + a2 = 0` directly. `analysis/sensitivity.hpp` measures how
pole positions shift under coefficient perturbation (finite differences).
`analysis/condition.hpp` estimates frequency response sensitivity to
coefficient errors. All analysis functions require `ConvertibleToDouble<T>`
since they convert to `double` for numerical computation.

## Filter Concepts

`concepts/filter.hpp` defines `FilterDesign`, `DesignableLowPass`,
`DesignableBandPass`, and `Processable`. These formalize the interface
that `SimpleFilter` and generic algorithms rely on.

## Coding Conventions

- No raw pointers for aggregates — use references, `std::array`, `std::span`
- `complex_for_t<T>` instead of `std::complex<T>` in template contexts
- ADL-friendly calls for complex operations: `using std::conj; conj(z);`
- All polynomial/prototype math parameterized on `T` — no hardcoded `double`
- Signal containers: `mtl::vec::dense_vector<T>`, not `std::vector<T>`
- Fixed-size storage: `std::array<T, N>`, not `std::vector`
- Denormal prevention: traits-aware via `DenormalPrevention<T>` (no-op for posits)
