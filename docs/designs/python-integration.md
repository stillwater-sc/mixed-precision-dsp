# Python Integration Architecture for mixed-precision-dsp

**Date:** 2026-04-14
**Context:** Design for `mp-dsp-python` — nanobind Python bindings for the `sw::dsp` library

---

## 1. The Problem: Template Combinatorics

MTL5's Python binding is straightforward — instantiate `dense_vector<double>`, `dense2D<double>`, expose them with NumPy buffer protocol. The types are simple containers.

`sw::dsp` has three-scalar parameterization:

```cpp
ButterworthLowPass<4, CoeffScalar, StateScalar, SampleScalar>
```

If you expose even 6 types (double, float, half, posit<32,2>, posit<16,1>, cfloat<8,4>) across 3 positions, that's 6^3 = 216 instantiations **per filter family**. With 7 IIR families x 4 response types (LP/HP/BP/BS) x orders 1-8, you're looking at tens of thousands of template instantiations. Build times would be hours and the binary would be enormous.

---

## 2. The Right Architecture: Python Orchestrates, C++ Crunches

Don't expose the template parameterization to Python. Instead:

```
Python (research notebook)          C++ (mp-dsp-python via nanobind)
─────────────────────────          ──────────────────────────────────
design = dsp.butterworth(           # Designs in double internally
    order=4, fs=44100, fc=1000)

# Ask C++ to process with specific type combinations
result_f32 = design.process(        # Instantiates float internally
    signal, dtype="float32")        # Returns NumPy float64

result_p16 = design.process(        # Instantiates posit<16,1>
    signal, dtype="posit16_1")      # Returns NumPy float64

sqnr = dsp.sqnr_db(                # Compares in double
    result_f32, result_p16)

plt.plot(...)                       # Python does the plotting
```

**Key insight:** Python never sees the template types. It passes a string (`"posit16_1"`) and gets back `float64` NumPy arrays. The C++ side has a dispatch table of pre-instantiated type combinations — not the full combinatorial explosion, but the 10-15 configurations that are actually interesting for research.

---

## 3. Repository Structure

```
mp-dsp-python/
├── CMakeLists.txt            # nanobind, links sw::dsp + Universal
├── src/
│   ├── bindings.cpp          # nanobind module definition
│   ├── filter_dispatch.hpp   # runtime type dispatch to pre-instantiated templates
│   ├── signal_bindings.cpp   # signal generators → NumPy
│   ├── filter_bindings.cpp   # IIR/FIR design + process
│   ├── image_bindings.cpp    # image ops → NumPy 2D arrays
│   ├── analysis_bindings.cpp # stability, sensitivity, SQNR
│   └── types.hpp             # enum of supported arithmetic types
├── python/
│   └── mpdsp/
│       ├── __init__.py
│       ├── filters.py        # Pythonic wrapper classes
│       └── plotting.py       # matplotlib convenience
├── notebooks/
│   ├── mixed_precision_iir.ipynb
│   ├── sensor_noise_image.ipynb
│   └── sqnr_comparison.ipynb
├── tests/
│   └── test_basic.py
└── README.md
```

---

## 4. Pre-Instantiated Type Table

Instead of combinatorial explosion, pre-instantiate the configurations that matter:

| Config | Coeff | State | Sample | Research Question |
|--------|-------|-------|--------|-------------------|
| `reference` | double | double | double | Ground truth |
| `gpu_baseline` | float | float | float | GPU/embedded baseline |
| `ml_hw` | double | float | bfloat16 | ML accelerator |
| `sensor_8bit` | double | double | integer<8> | Standard ADC |
| `sensor_6bit` | double | double | integer<6> | Noise-limited sensor |
| `posit_full` | double | posit<32,2> | posit<16,1> | Posit research |
| `fpga_fixed` | double | fixpnt<32,24> | fixpnt<16,12> | FPGA datapath |
| `tiny_posit` | double | posit<8,2> | posit<8,2> | Ultra-low power |

That's 8 configs, not 216. Each is a concrete C++ instantiation behind a Python string key.

---

## 5. Dispatch Mechanism

The C++ dispatch layer maps runtime string keys to compile-time template instantiations:

```cpp
// types.hpp
enum class ArithConfig {
    reference, gpu_baseline, ml_hw,
    sensor_8bit, sensor_6bit,
    posit_full, fpga_fixed, tiny_posit
};

ArithConfig parse_config(const std::string& name);

// filter_dispatch.hpp
template <typename FilterDesign>
std::vector<double> process_with_config(
    const FilterDesign& design,
    std::span<const double> input,
    ArithConfig config)
{
    switch (config) {
    case ArithConfig::reference:
        return process_typed<double, double, double>(design, input);
    case ArithConfig::gpu_baseline:
        return process_typed<float, float, float>(design, input);
    case ArithConfig::posit_full:
        return process_typed<double, posit<32,2>, posit<16,1>>(design, input);
    // ...
    }
}
```

Each `process_typed<C,S,Sa>` converts the `double` input to `Sa`, processes through a `SimpleFilter` with the specified types, converts back to `double` for Python.

---

## 6. Python API Design

```python
import mpdsp
import numpy as np
import matplotlib.pyplot as plt

# Design a filter (always in double precision)
filt = mpdsp.butterworth_lowpass(order=4, sample_rate=44100, cutoff=1000)

# Generate test signal
signal = mpdsp.sine(length=1000, frequency=440, sample_rate=44100)

# Process with different arithmetic configurations
results = {}
for config in ["reference", "gpu_baseline", "posit_full", "sensor_6bit"]:
    results[config] = filt.process(signal, dtype=config)

# SQNR comparison
ref = results["reference"]
for name, result in results.items():
    if name != "reference":
        sqnr = mpdsp.sqnr_db(ref, result)
        print(f"{name:20s}  SQNR = {sqnr:.1f} dB")

# Stability analysis
poles = filt.poles()
margin = filt.stability_margin()
sensitivity = filt.worst_case_sensitivity()

# Image processing
img = mpdsp.checkerboard(256, 256, block_size=8)
noisy = mpdsp.add_noise(img, stddev=0.1)
edges = mpdsp.canny(noisy, low=0.1, high=0.3)
mpdsp.write_pgm("edges.pgm", edges)
```

---

## 7. Build System

```cmake
cmake_minimum_required(VERSION 3.22)
project(mp-dsp-python LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
find_package(dsp CONFIG REQUIRED)       # brings in Universal + MTL5

nanobind_add_module(mpdsp_core
    src/bindings.cpp
    src/signal_bindings.cpp
    src/filter_bindings.cpp
    src/image_bindings.cpp
    src/analysis_bindings.cpp
)

target_link_libraries(mpdsp_core PRIVATE sw::dsp)
target_compile_features(mpdsp_core PRIVATE cxx_std_20)
```

---

## 8. Phased Implementation

### Phase 1: Signal + SQNR
- Signal generators (sine, chirp, noise) → NumPy arrays
- ADC/DAC quantization with type dispatch
- SQNR measurement
- **Deliverable:** SQNR comparison notebook

### Phase 2: IIR Filters
- Butterworth, Chebyshev I/II, Elliptic design
- process() with type dispatch
- Frequency response, pole-zero access
- **Deliverable:** Mixed-precision IIR comparison notebook

### Phase 3: Analysis
- Stability margin, coefficient sensitivity, condition number
- Pole displacement under quantization
- **Deliverable:** Numerical quality analysis notebook

### Phase 4: Image Processing
- Generators, convolution, Sobel, Canny
- PGM/PPM/BMP I/O
- **Deliverable:** Sensor noise arithmetic precision notebook (Issue #41)

---

## 9. Why This Architecture

1. **Python is where DSP researchers work** — Jupyter + matplotlib + SciPy is the standard research workflow
2. **SQNR tables and plots are the value proposition** — the mixed-precision comparison needs Python visualization to be compelling
3. **Avoids combinatorial explosion** — 8 pre-instantiated configs vs. thousands of template permutations
4. **Clean separation of concerns** — C++ does the mixed-precision math, Python does orchestration and visualization
5. **Follows the mtl5/mtl5-python pattern** — separate repo, nanobind, NumPy interop
6. **The sensor noise argument (Issue #41) needs plots** — matplotlib is essential for the assessment documents
