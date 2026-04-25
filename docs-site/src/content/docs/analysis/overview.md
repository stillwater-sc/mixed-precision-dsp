---
title: Numerical Analysis
description: Tools for analyzing filter stability, sensitivity, and acquisition pipeline precision
---

The `sw::dsp::analysis` namespace contains static analysis and
characterization tools that take a filter or pipeline and report
numerical health metrics. None of these are intended for the hot
processing path — they're design-time and verification helpers.

## Modules

| Header | Purpose |
|---|---|
| `sw/dsp/analysis/stability.hpp` | Pole/zero extraction, distance-to-unit-circle stability margins |
| `sw/dsp/analysis/sensitivity.hpp` | Pole displacement under coefficient quantization (finite-difference Jacobians) |
| `sw/dsp/analysis/condition.hpp` | Frequency-response sensitivity / condition number |
| `sw/dsp/analysis/acquisition_precision.hpp` | SNR, ENOB, NCO SFDR, CIC bit-growth, per-stage noise budgets, CSV export for visualization |

The umbrella header `sw/dsp/analysis/analysis.hpp` includes all four.

## When to Use What

- **Stability** — verify that a designed cascade has all poles inside
  the unit circle, with sufficient margin to survive coefficient
  quantization.
- **Sensitivity** — answer "if I drop CoeffScalar from `double` to
  `posit<16,1>`, where do my poles move?"
- **Condition number** — bound how much the frequency response will
  drift under coefficient perturbation, without computing the
  perturbed response itself.
- **Acquisition precision** — run real signals through a pipeline,
  measure end-to-end SNR / ENOB / SFDR, and export CSV-formatted
  Pareto data compatible with the existing `precision_sweep.csv`
  visualization. See the [acquisition pipeline precision
  analysis](./acquisition-precision/) page.
