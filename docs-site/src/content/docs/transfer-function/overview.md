---
title: Transfer Function Monitor Overview
description: Numerical Bode sweeps for any LTI block plus analytical closed-form pole/zero extraction from filter prototypes, with dump-to-JSON/CSV for external visualization
---

The `transfer_function/` module (Epic [#160](https://github.com/stillwater-sc/mixed-precision-dsp/issues/160),
sub-issues [#157](https://github.com/stillwater-sc/mixed-precision-dsp/issues/157)
+ [#158](https://github.com/stillwater-sc/mixed-precision-dsp/issues/158))
adds two complementary tools for inspecting LTI block behavior:

| Tool | Header | Works on | Output |
|---|---|---|---|
| `sweep_bode` | `transfer_function/bode.hpp` | any LTI block with `process(T) + reset()` | (freq, mag_dB, phase_rad) triples via CSV |
| `PoleZeroPlot` builders | `transfer_function/pole_zero.hpp` | filter families whose prototype we own (Butterworth / Chebyshev I & II / Bessel) | s-plane + z-plane pole/zero locations via JSON |

Neither tool draws anything — they emit structured data that
[mp-dsp-python](https://github.com/stillwater-sc/mp-dsp-python) turns
into Bode plots and unit-circle pole/zero diagrams.

## When to reach for which

**Numerical Bode sweep** (`sweep_bode`) works on **any** LTI block that
exposes `process(T)` and `reset()`. It probes the block by feeding a
cosine test signal at each frequency in a log-spaced grid and
correlating the output against cos/sin bases to extract magnitude and
phase. Use it when:

- The block is user-composed (a cascade of your own filters + gains)
  and doesn't have a closed-form transfer function.
- You want to verify a filter's response empirically, including any
  precision-induced deviations from the ideal analytical response.

**Analytical pole/zero** (`butterworth_prototype`, etc.) works only on
filter families whose design we own analytically. It computes the exact
s-plane pole/zero locations from closed-form formulas, threads them
through Constantinides transformations (LP → HP / BP / BS), and maps
them to the z-plane via the bilinear transform. Use it when:

- You need exact pole/zero locations (for stability analysis, sensitivity
  studies, or ideal-response reference).
- You want to characterize a filter without instantiating it and
  running samples through it.

The two tools are complements: sweep_bode measures the compiled
filter's real response (precision loss and all); analytical pole/zero
tells you what it *should* be, so you can quantify the gap.

## Numerical Bode sweep

```cpp
#include <sw/dsp/filter/iir/butterworth.hpp>
#include <sw/dsp/transfer_function/bode.hpp>
using namespace sw::dsp;

SimpleFilter<iir::ButterworthLowPass<4>> lp;
lp.setup(4, /*fs=*/48000.0, /*cutoff=*/1000.0);

// Log-spaced sweep from 10 Hz to 20 kHz, 200 points.
auto bode = transfer_function::sweep_bode(
    lp, /*fs=*/48000.0,
    /*fmin=*/10.0, /*fmax=*/20000.0,
    /*num_points=*/200);

bode.dump_csv("butter_lp.bode.csv");
```

Per test frequency, `sweep_bode`:

1. Resets the block, feeds `settle_samples` cosine values to prime state.
2. Correlates the output against cos and sin bases (Hann-windowed to
   suppress non-integer-cycle bias) over an adaptive window whose
   length auto-scales with $f$ to hit `target_cycles` (default 32).
3. Extracts magnitude $A = \sqrt{a_c^2 + a_s^2}$ and phase
   $\phi = \arctan2(a_s, a_c)$.

CSV schema:

```
freq_hz,magnitude_dB,phase_rad
10,-0.0004,-0.0021
11.7,-0.0006,-0.0025
...
1000,-3.010,-1.5708
...
```

## Analytical pole/zero extraction

```cpp
#include <sw/dsp/transfer_function/pole_zero.hpp>
using namespace sw::dsp::transfer_function;

// s-plane LP prototype at cutoff = 1 kHz
auto p = butterworth_prototype(/*order=*/4, /*cutoff=*/1000.0);

// Turn it into a high-pass at 500 Hz (Constantinides substitution).
lp_to_hp(p, 500.0);

// Map to the z-plane at fs = 48 kHz.
apply_bilinear(p, /*fs=*/48000.0);

// Dump for the mp-dsp-python pole/zero viewer.
p.dump_json("butter_hp_z.json");
```

`PoleZeroPlot` carries both `s_poles` / `s_zeros` (the continuous-time
prototype after any LP→\* transform) and `z_poles` / `z_zeros`
(populated by `apply_bilinear`). The JSON dump keeps them side by side
so the viewer can render either.

### Supported prototype families

- `butterworth_prototype(order, cutoff)` — poles on an LHP unit
  circle; no finite zeros.
- `chebyshev1_prototype(order, cutoff, ripple_dB)` — poles on an
  LHP ellipse.
- `chebyshev2_prototype(order, cutoff, stopband_dB)` — inverse
  Chebyshev; both finite poles and $2\lfloor N/2\rfloor$ finite
  zeros on the imaginary axis.
- `bessel_prototype(order, cutoff)` — poles are the LHP roots of
  the reverse Bessel polynomial (via Laguerre root-finder).
- `elliptic_prototype(...)` — **stub, throws**. The library's
  `filter/iir/elliptic.hpp` computes elliptic pole/zero via internal
  Jacobi-sn helpers; exposing a reusable analytical API here awaits
  a follow-up that factors those helpers into `sw::dsp::math`.

### Constantinides transformations

Each of `lp_to_hp`, `lp_to_bp(low, high)`, `lp_to_bs(low, high)`
operates in place on a `PoleZeroPlot`, transforming its `s_poles` and
`s_zeros` arrays and updating the metadata (kind, cutoff/low/high).

### Bilinear transform

`apply_bilinear(plot, fs)` maps every $s$-plane pole and zero via
$z = (2f_s + s) / (2f_s - s)$, populating `z_poles` and `z_zeros`.
$s$-plane infinity maps to $z = -1$; any implicit infinity zeros
(where the s-plane has more poles than zeros) show up as $z = -1$
zeros to keep pole and zero counts equal.

### JSON dump

```json
{
  "design": "butterworth",
  "order": 4,
  "kind": "highpass",
  "cutoff_hz": 500.0,
  "low_hz": 0.0,
  "high_hz": 0.0,
  "sample_rate_hz": 48000.0,
  "ripple_dB": 0.0,
  "stopband_dB": 0.0,
  "s_poles": [[-1962.24, 2367.51], [-2367.51, 1962.24], ...],
  "s_zeros": [[0, 0], [0, 0], [0, 0], [0, 0]],
  "z_poles": [...],
  "z_zeros": [...]
}
```

The mp-dsp-python renderer overlays the z-plane pole/zero on the unit
circle for visual stability inspection and can chart the s → z
mapping side by side.

## Related pages

- [Pipeline Probes](../probe/overview/) — the sample-capture side of
  the introspection toolkit
- [Analysis - Stability](../analysis/stability/) — numerical pole
  extraction from biquad coefficients (complementary; use
  `pole_zero.hpp` when you have the analog prototype, `stability.hpp`
  when you only have a discretized cascade)
- [Analysis - Condition](../analysis/condition/) — frequency-response
  sensitivity to coefficient perturbation (pairs well with `sweep_bode`
  for empirical validation)
- [mp-dsp-python](https://github.com/stillwater-sc/mp-dsp-python) —
  Python peer repo that renders Bode plots and pole/zero diagrams
