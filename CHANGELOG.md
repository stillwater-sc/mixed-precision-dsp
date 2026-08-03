# Changelog

All notable changes to `mixed-precision-dsp` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases before v0.8.0 were tagged without a changelog file; the entries below
those are summarized from their release commits and are intentionally terse.

## [Unreleased]

Nothing yet.

## [0.8.0] — 2026-08-03

Two multi-issue tracks land on top of v0.7: the multirate demonstrator backlog
and the Pipeline Probe & Transfer-Function Monitor epic. All 10 issues in the
v0.8 milestone are closed.

### Added

**Pipeline Probe module — `sw::dsp::probe`**

- `SignalProbe` + `NoOpProbe` + `ProbedStage` capture infrastructure, letting any
  pipeline stage be instrumented without changing its type when probing is off
  (#155). `include/sw/dsp/probe/signal_probe.hpp`
- Four domain views over captured buffers — time, magnitude, phase, and I-Q
  constellation (#156). `include/sw/dsp/probe/views.hpp`

**Transfer-Function Monitor module — `sw::dsp::transfer_function`**

- `sweep_bode` numerical Bode analyzer for arbitrary LTI blocks — magnitude,
  phase, and group-delay sweeps (#157). `include/sw/dsp/transfer_function/bode.hpp`
- Closed-form analytical pole/zero extraction for all five filter families —
  Butterworth, Chebyshev I, Chebyshev II, Bessel, and Elliptic (#158, #202).
  `include/sw/dsp/transfer_function/pole_zero.hpp`

**Multirate demonstrators — `applications/multirate_examples/`**

- `audio_resampler` — 44.1/48 kHz rational sample-rate conversion (#136)
- `fractional_delay` — polyphase 1/L-sample fractional delay (#138)
- `channelizer` — Bellanger polyphase M-channel analysis bank (#137)
- `software_radio` — 100 MHz → 100 kHz SDR receiver chain (#139)

Every demo carries an acceptance criterion that is checked at exit, so a
regression in the underlying DSP fails the demo rather than printing bad numbers.

**Library**

- `sw::dsp::multirate` gains the `FractionalDelay` and `Channelizer` classes.
- `elliptic_sn_series` promoted from an internal helper in `filter/iir/elliptic.hpp`
  to public API in `include/sw/dsp/math/elliptic_integrals.hpp`, so the elliptic
  pole/zero prototype and the elliptic filter design share one implementation.

**Documentation**

- New docs-site sidebar categories for Pipeline Probes and the Transfer Function
  Monitor, with overview pages for `probe`, `probe/views`, and
  `transfer-function`, plus cross-references from the `analysis` and
  `acquisition` overviews (#159).

### Changed

- `filter/iir/elliptic.hpp` now consumes the shared `elliptic_sn_series` instead
  of its own local copy (net −23 lines).

## [0.7.0] — 2026-08-02

### Added

- `scope_demo_2ch` — two-channel oscilloscope demo (#173) exercising
  `CrossChannelTrigger` (AandB coincidence mode) and `ChannelAligner`
  (0.3-sample fractional skew compensation). Five precision plans including an
  asymmetric per-channel plan (posit32 chA + posit16 chB); acceptance criterion
  is cross-channel Pearson correlation > 0.99.

## [0.6.1] — 2026-08-02

Tagged as PATCH per project preference; scope covers everything on main since
v0.6.0 (2026-04-27), which under strict semver would have warranted a MINOR bump.

### Added

- Full spectrum-analyzer module: `RealtimeSpectrum` streaming FFT engine, RBW/VBW
  filters, swept-LO chirp generator, front-end corrector, trace averaging
  (5 modes), waterfall buffer, markers + peak-find, 5 detector reducers, and the
  `spectrum_analyzer_demo` application.
- `scope_demo` capstone application with per-stage precision plans and
  calibration-profile pre-distortion.
- `CMakePresets.json` with `dev` / `release` / `ci` / `ci-regression` presets,
  regression-test scaffolding, and the CTest label taxonomy.

### Fixed

- `windows`: `dolph_chebyshev_window` collapsed to a near-constant window at every
  (N, atten_db). The antisymmetric-real spectrum canceled in the cos-only IDFT,
  leaving only the DC bin. Fixed by dropping the (−1)^k twist and taking `abs()`
  of the Chebyshev polynomial to make W symmetric, then fftshifting the cos-IDFT
  result (#200).
- `filter/fir`: `design_fir_lowpass` produced NaN at the center tap under narrow
  cfloat types (#201).

## [0.6.0] — 2026-04-27

Version bump release; see the GitHub Release notes generated from conventional
commits for the change set.

## [0.5.0] — 2026-04-19

Version bump release; see the GitHub Release notes generated from conventional
commits for the change set.

## [0.4.1] — 2026-04-17

### Added

- Tag-based release workflow (`.github/workflows/release.yml`) that creates a
  GitHub Release with an auto-generated changelog from conventional commits when
  a `v*` tag is pushed, verifying the tag matches the CMake version first.

[Unreleased]: https://github.com/stillwater-sc/mixed-precision-dsp/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/stillwater-sc/mixed-precision-dsp/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/stillwater-sc/mixed-precision-dsp/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/stillwater-sc/mixed-precision-dsp/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/stillwater-sc/mixed-precision-dsp/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/stillwater-sc/mixed-precision-dsp/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/stillwater-sc/mixed-precision-dsp/releases/tag/v0.4.1
