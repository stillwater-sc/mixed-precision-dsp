# Application organization

Demonstration programs are grouped into a hierarchy that mirrors the
test-suite layout (see [`tests/README.md`](../tests/README.md) for the
broader rationale). Each application's `FOLDER` target property places it
under an `"Applications/..."` path that Visual Studio (and other IDE
generators) display in the Solution Explorer. Makefile / Ninja generators
ignore the folder property — there's no behavioral effect on command-line
builds.

The hierarchy is set up via `dsp_add_application(target_name folder)`,
defined in [CMakeLists.txt](CMakeLists.txt) at this directory's scope and
visible to all sub-CMakeLists.txt files.

## Hierarchy

```text
Applications/
├── Filter/                     IIR / FIR filter demonstrators
│   ├── butterworth_lowpass     IIR design + biquad cascade application
│   └── polyphase_filter        FIR multirate (polyphase) demo
├── Spectral/
│   └── fft_tradeoff            FFT precision/length trade-off analysis
├── SDR/                        Software-defined radio receiver demos
│   ├── acquisition_demo        End-to-end SDR pipeline (capstone for #84)
│   └── sdr_demo                TX→AWGN→RX link, precision sweep (capstone for #85)
├── Instrument/                 Digital-oscilloscope / spectrum-analyzer demos
│   ├── scope_demo              End-to-end scope pipeline (capstone for #133)
│   └── spectrum_analyzer_demo  End-to-end spectrum-analyzer pipeline (capstone for #134)
├── Estimation/
│   ├── ekf_bearing_range       Extended Kalman filter
│   └── ukf_tracking            Unscented Kalman filter
├── Image/
│   ├── edge_detection
│   ├── image_pipeline
│   └── mixed_precision_image
└── PrecisionTools/             Cross-cutting precision-comparison tooling
    ├── precision_sweep         Flagship precision-sweep tool (Issue #69)
    └── iir_precision_sweep     IIR-specific precision sweep (mp_comparison)
```

## Notes on placement decisions

### `Filter/` is flat (not split into IIR/FIR)

The test-suite mirror (`Tests/Filter/`) is split into IIR / FIR / Generic
because there are 9 filter tests. Applications has only one IIR demo and
one FIR demo, so flat `Applications/Filter/` is enough; the source-
directory names (`iir_demo`, `fir_demo`) still document each app's
character. Revisit if/when there are 5+ filter applications.

### `PrecisionTools/` is its own bucket

`precision_sweep` and `iir_precision_sweep` are not module-specific
demos — they're cross-cutting tools that exercise the library's
mixed-precision machinery to compare number-system performance across
configurations. Putting them under `Filter/` (because IIR is one of
the things they sweep) would understate their scope; a dedicated
`PrecisionTools/` folder keeps them visible as their own category.

### `SDR/` (not `Acquisition/`)

This mirrors the docs-site rename that landed with PR #131: the
`include/sw/dsp/acquisition/` directory was reframed as the SDR
receiver front-end module (its sidebar label is "Software-Defined
Radio"). The application demo is named `acquisition_demo` for legacy
reasons but the Solution Explorer folder uses the more accurate `SDR`.

## Adding a new application

In your subdirectory's `CMakeLists.txt`:

```cmake
dsp_add_application(my_demo "Applications/MyModule")
```

The first argument is the source file's stem (the file is expected to
be `my_demo.cpp` in the same directory); the second is the IDE folder
path. Pick the folder by mirroring the source-module structure. If
your app is brand-new module, add a top-level folder
(`Applications/NewModule`) and add an entry to the hierarchy tree above.

For applications that need additional configuration (compile
definitions, data files, multiple sources), the helper's signature
stays the same — just add the extra `target_*` calls right after the
`dsp_add_application()` line.

## Writing output

Demos must not write into the source tree — running one from the
repository root should leave nothing behind. `dsp_add_application()`
bakes a per-build output directory into every application via the
`DSP_DEMO_OUTPUT_DIR` compile definition; use the helpers in
[`common/demo_output.hpp`](common/demo_output.hpp) to reach it:

```cpp
#include <common/demo_output.hpp>

// A single file, overridable with --csv=<path>:
std::string csv_path = sw::dsp::demo::output_path("my_demo.csv");

// A demo that writes several files into a directory:
std::string outdir = sw::dsp::demo::output_dir();
```

That resolves to `<binary-dir>/demo-output/`, which the build creates
at configure time and `build*/` in `.gitignore` already covers. Keep
the `--csv=<path>` (or positional output directory) override, use an
explicit path verbatim when given, and print the path actually written
so a user can find the file.
