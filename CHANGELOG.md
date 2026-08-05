# Changelog

All notable changes to `mixed-precision-dsp` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases before v0.8.0 were tagged without a changelog file; the entries below
those are summarized from their release commits and are intentionally terse.

## [Unreleased]

### Fixed

- `filter/fir/remez`: the Parks-McClellan designers returned filters that were
  not equiripple (#203). `remez()`, `design_fir_equiripple_lowpass/bandpass`,
  and `design_halfband()` all share the affected path. Four defects, each
  independently sufficient to break the result:
  - **`compute_delta()` returned the ripple with the wrong sign.** The delta
    formula must carry a leading minus to match the `D + (-1)^i * delta / W`
    convention `eval_approx()` interpolates. With the wrong sign the
    interpolant misses the last reference point, the error curve alternates
    only n-1 times instead of n, the extremal search can never assemble a full
    alternating set, and the exchange freezes on its first reference set — the
    "converges in two iterations and then sits still" symptom in the report.
  - **The tap extraction was not an orthogonal transform.** The inverse DCT
    sampled the half-open interval `[0, 0.5)` with uniform weights; DCT-I
    orthogonality requires the closed interval with half-weighted endpoints.
    Every recovered coefficient carried an O(1/M) error. This was the root
    cause identified in the report.
  - **The extremal search compared across transition bands.** The grid is a
    concatenation of per-band segments, so testing a band's last point against
    the next band's first point compares across a gap. Band edges — always
    extremal in a Parks-McClellan solution — could never enter the reference
    set, leaving the largest errors in the design uncontrolled. The search is
    now band-aware and treats every band edge as a candidate.
  - **The final extraction paired the reference set with a stale delta.** The
    exchange replaces its reference set after computing delta, so the tap
    extraction interpolated values the final set does not satisfy. On specs
    where the last exchange still moved points this produced filters whose
    measured stopband bore no relation to the reported ripple (45 dB claimed,
    9 dB delivered). Delta is now re-derived from the final set.

  Measured against `scipy.signal.remez` over 30 lowpass specifications
  (varying length, band edges, and 1:1 / 1:10 / 10:1 / 1:100 band weights),
  27 agree within 3% and the remaining 3 within 3.8%. For the report's
  reference case — 95 taps, bands {0, 0.20, 0.25, 0.5} — passband ripple goes
  from 2.40 dB to 0.0017 dB with unity DC gain, and stopband attenuation from
  49 dB to 80.2 dB. `design_halfband()` attenuation is now monotonic in both
  tap count and transition width; every cell where scipy converges now matches
  it.

- `filter/fir/remez`: all four linear-phase FIR types are now designed
  correctly (#205). The exchange solves for a cosine polynomial, which is only
  valid for Type I; Types II, III and IV realize their zero-phase amplitude as
  `A(f) = q(f) * P(cos 2*pi*f)` for `q = cos(pi*f)`, `sin(2*pi*f)` and
  `sin(pi*f)` respectively. That factor is now folded into the problem
  statement (`W' = W*q`, `D' = D/q`) so the exchange sees a pure
  cosine-polynomial problem, and is reapplied analytically when converting
  the recovered coefficients to taps. The polynomial degree was also wrong for
  Types II and III — both were one too high — and the grid now excludes the
  frequencies where `q` vanishes, since `D/q` is singular there.

  Type II (even tap counts) previously returned roughly half the intended DC
  gain with ~90 dB of passband ripple; at 64 taps on `{0, 0.20, 0.25, 0.5}` it
  now measures DC 0.9987, ripple 0.0234 dB, stopband 57.4 dB — the
  Parks-McClellan values to three digits. A 31-tap Hilbert transformer over
  `[0.05, 0.45]` went from 1.487 dB of ripple to 0.047 dB (reference: 0.047).
  Across 42 specifications spanning all four types and both parities, 40 agree
  with `scipy.signal.remez` within 3%; the two that do not are wide-band
  Hilbert designs where both implementations are past -160 dB.

  The convergence check added for #203 now applies to all four types rather
  than only the symmetric ones, since `delta` finally describes what every
  type returns.

- `filter/fir/remez`: the exchange keeps the best iterate seen rather than the
  last one, and the barycentric scale factor is derived from the actual node
  range instead of being fixed at 2. Together these fix designs whose optimum
  lies near the double-precision floor — a wide-band Hilbert transformer with
  a generous tap budget — where `delta` collapses toward rounding noise and
  the reference set could stampede into a narrow cluster and blow up the
  interpolation. A 64-tap Hilbert over `[0.10, 0.5]` now lands at 9.0e-10
  against scipy's 3.1e-10, where before it diverged outright.

- `filter/fir/remez`: `remez()` now throws `std::runtime_error` when the
  exchange fails to converge, instead of returning the non-converged iterate.
  Some specifications — notably half-band band edges with many taps and a
  narrow transition — demand more attenuation than double precision can
  represent and are degenerate for the exchange; reference implementations
  report this rather than returning a filter whose stopband sits above its
  passband. The check is the alternation theorem applied to the design that is
  about to be returned. A second acceptance route admits designs whose worst
  weighted error is below 1e-4 of the problem scale (about -80 dB) without
  proving them equiripple, since specifications solved almost exactly drive
  `delta` into rounding noise and the alternation test then compares against a
  meaningless number. Specifications that neither route admits are the ones
  `scipy.signal.remez` also refuses.
- `filter/fir/remez`: barycentric weights now fold a factor of 2 into each node
  difference, the standard Parks-McClellan scaling. The factor is common to all
  weights and cancels in every ratio it appears in, but it keeps the products
  near O(N) instead of growing like 2^N.

### Known limitations

- `design_halfband()` renormalizes the odd-offset taps so the DC gain is
  exactly 1. Since the Remez solution places a ripple extremum at DC, this
  scaling roughly doubles the stopband error, costing a consistent ~6 dB of
  attenuation (e.g. 87.1 dB before normalization, 81.1 dB after, at 51 taps
  with a 0.10 transition). It is what keeps the existing DC-gain test within
  tolerance at wide transition widths; trading it away is a deliberate design
  decision that has not been made. Tracked as #206.

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
