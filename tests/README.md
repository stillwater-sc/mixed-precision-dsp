# Test organization

The test suite is grouped into a hierarchy that mirrors `include/sw/dsp/`'s
module layout. Each test executable's `FOLDER` target property places it
under a `"Tests/..."` path that Visual Studio (and other IDE generators)
display in the Solution Explorer. Makefile / Ninja generators ignore the
folder property — there's no behavioral effect on command-line builds.

The hierarchy is set up via `dsp_add_test(test_name folder)` in
[CMakeLists.txt](CMakeLists.txt). The pattern is borrowed from Universal's
`compile_all()` macro, simplified for our 1-test-per-source-file convention.

## Hierarchy

```text
Tests/
├── Foundation/                  Cross-cutting type-system tests
│   ├── test_concepts            DspScalar / DspField / DspOrderedField satisfaction
│   └── test_projection          Type projection / embedding helpers
├── Signals/                     Signal data sources + I/O
│   ├── test_generators          Sine, square, chirp, multitone, etc.
│   └── test_io                  WAV / CSV file readers and writers
├── Windows/
│   └── test_windows             Hamming, Hann, Blackman, Kaiser, etc.
├── Quantization/
│   └── test_quantization        Uniform / non-uniform quantizers
├── Filter/
│   ├── IIR/                     Analog-prototype-derived recursive filters
│   │   ├── test_biquad
│   │   ├── test_butterworth
│   │   ├── test_chebyshev
│   │   ├── test_bessel
│   │   └── test_elliptic
│   ├── FIR/                     Finite-impulse-response designs
│   │   ├── test_fir
│   │   ├── test_fir_multirate
│   │   └── test_remez           Parks-McClellan equiripple
│   └── Generic/                 Design-agnostic filter operations (see below)
│       └── test_filtfilt        Zero-phase forward-backward filtering
├── Spectral/
│   ├── test_fft
│   ├── test_spectral            Z-transform, Laplace, PSD, spectrogram
│   └── test_bluestein           Chirp-z arbitrary-length DFT
├── Conditioning/
│   ├── test_conditioning        AGC, compressor, envelope detector
│   ├── test_src                 Rational sample-rate conversion
│   └── test_noise_shaping       Higher-order delta-sigma shaping
├── Acquisition/                 SDR / IF-receiver front-end primitives
│   ├── test_cic                 Cascaded integrator-comb
│   ├── test_halfband
│   ├── test_nco                 Numerically controlled oscillator
│   ├── test_ddc                 Digital down-converter
│   └── test_decimation_chain    Multistage decimation composition
├── Instrument/                  Oscilloscope / spectrum-analyzer primitives
│   ├── test_instrument_trigger           Edge / level / slope / qualifier
│   ├── test_instrument_ring_buffer       Pre/post-trigger capture
│   ├── test_instrument_calibration       Profile + equalizer FIR
│   ├── test_instrument_peak_detect       Min/max decimation
│   ├── test_instrument_display_envelope  Pixel-rate envelope
│   ├── test_instrument_fractional_delay  Sub-sample windowed-sinc FIR delay
│   ├── test_instrument_channel_aligner   Multi-channel time alignment
│   └── test_instrument_measurements      Cursor / scope-style measurements
├── Analysis/                    Stability, sensitivity, precision analysis
│   ├── test_analysis            Pole-zero / sensitivity / condition number
│   └── test_acquisition_precision  SNR/ENOB/SFDR/CIC bit-growth (acq-pipeline-specific)
├── Estimation/
│   └── test_kalman              Plus LMS / RLS adaptive estimators
└── Image/
    ├── test_image
    ├── test_image_generators
    └── test_image_io
```

## Notes on a few placement decisions

### Filter/Generic — does this category really earn its own folder?

Right now `Filter/Generic` contains only one test (`test_filtfilt` — the
zero-phase forward-backward filter operation). The concern: a folder with
one entry feels gratuitous, especially when the obvious alternative is to
fold `filtfilt` under `Filter/IIR` (since IIR filters are by far the most
common application of zero-phase processing).

The argument for keeping `Filter/Generic` as its own folder: `filtfilt`
is **design-agnostic**. It accepts any `Processable<T>` filter — IIR
biquad cascade, FIR, or anything else with a `process()` method. Putting
it under `IIR` would imply it's IIR-specific, which is wrong; future
filter post-processing primitives (e.g., a generic transient suppressor,
a generic group-delay equalizer) would belong in the same conceptual
category.

The trade-off is between "honest taxonomy" (Generic deserves its own
slot) and "minimum viable hierarchy" (one test isn't worth a folder).
We chose the former; if `Filter/Generic` is still 1 test deep at v0.7,
revisit and possibly fold it in.

### `test_acquisition_precision` lives under `Tests/Analysis`, not `Tests/Acquisition`

It's an analysis primitive (SNR, ENOB, NCO SFDR, CIC bit-growth
verification) that happens to be specialized for the acquisition pipeline.
Its source file is `include/sw/dsp/analysis/acquisition_precision.hpp`
(under analysis/, not acquisition/), and the test exercises analysis
behavior — comparing a reference signal to a test signal in dB.
Placing it under Analysis matches both the source layout and the test's
character.

### Foundation includes `test_projection`

`test_projection` exercises the type-system helpers that let any pipeline
be re-instantiated at a different scalar precision. It's not tied to any
specific signal-processing module — it's part of the cross-cutting
mixed-precision machinery. Same logical home as `test_concepts`.

## Adding a new test

```cmake
dsp_add_test(test_my_feature "Tests/MyModule")
```

The first argument is the source file's stem (the file is expected to be
`tests/test_my_feature.cpp`); the second is the IDE folder path. Pick
the folder by mirroring `include/sw/dsp/`'s module structure. If the new
test extends an existing module's coverage, drop it in the existing
folder. If it's a brand new module, create a new top-level folder
(`Tests/NewModule`) and add an entry to the hierarchy tree above.

If the test needs additional configuration (compile definitions, data
files, etc.), the helper's signature stays the same — just add the
extra `target_*` calls right after the `dsp_add_test()` line, as
`test_instrument_calibration` does for `TESTS_DATA_DIR`.
