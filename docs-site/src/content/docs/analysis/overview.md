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

## See also

- [Pipeline Probes](../probe/overview/) — sample-capture primitives
  for inspecting intermediate signals in an assembled pipeline. The
  probe stream feeds domain views (time / magnitude / phase / I-Q)
  that complement the analytical measurements here.
- [Transfer Function Monitor](../transfer-function/overview/) —
  numerical Bode sweeps for any LTI block, plus closed-form
  analytical pole/zero extraction from filter prototypes. The Bode
  side is empirical (measures the compiled filter's real response);
  the pole/zero side is the exact ideal. Together they let you
  quantify precision-induced deviation from ideal at the pole/zero
  level, complementing the stability + sensitivity primitives here.
