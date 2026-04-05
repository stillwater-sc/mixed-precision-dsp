# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Configure (FetchContent pulls Universal and MTL5 automatically)
cmake -B build -Wno-dev

# Build all targets
cmake --build build -j4

# Run tests
ctest --test-dir build --output-on-failure

# Build with clang
cmake -B build_clang -DCMAKE_CXX_COMPILER=clang++ -Wno-dev
cmake --build build_clang -j4
ctest --test-dir build_clang --output-on-failure

# RISC-V cross-compile
cmake -B build_rv64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/riscv64-gcc.cmake -Wno-dev
cmake --build build_rv64 -j4
```

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

## Coding Conventions

- No raw pointers for aggregates — use references, `std::array`, `std::span`
- `complex_for_t<T>` instead of `std::complex<T>` in template contexts
- ADL-friendly calls for complex operations: `using std::conj; conj(z);`
- All polynomial/prototype math parameterized on `T` — no hardcoded `double`
- Signal containers: `mtl::vec::dense_vector<T>`, not `std::vector<T>`
- Fixed-size storage: `std::array<T, N>`, not `std::vector`
- Denormal prevention: traits-aware via `DenormalPrevention<T>` (no-op for posits)
