---
title: End-to-End Demo
description: Capstone walkthrough composing NCO, mixer, polyphase decimation, and the precision-analysis primitives into a configurable IF receiver
---

The library ships a runnable demo at
[`applications/acquisition_demo/`](https://github.com/stillwater-sc/mixed-precision-dsp/tree/main/applications/acquisition_demo)
that ties the entire data-acquisition stack together: a simulated ADC,
a DDC composing NCO + complex mixer + polyphase decimation, and the
[`acquisition_precision`](../../analysis/acquisition-precision/)
analysis primitives — all swept across a representative range of
number-system configurations.

This page walks through the topology, the design decisions that fall
out of mixed-precision constraints, and how to read the output.

## Pipeline

```text
+------------------------+    +-----------------------------+    +-----------+
| simulate_adc(N_bits)   | -> | DDC<Coeff,State,Sample> {   | -> | I/Q       |
|   real cosine + noise  |    |   NCO at f_IF / f_S         |    | baseband  |
|   N-bit uniform quant  |    |   complex mixer (×conj)     |    |           |
+------------------------+    |   PolyphaseDecimator ×8     |    +-----------+
                              | }                           |          |
                              +-----------------------------+          |
                                                                       v
                                                       +------------------------+
                                                       | snr_db / enob / SFDR   |
                                                       | vs uniform-double ref  |
                                                       +------------------------+
                                                                       |
                                                                       v
                                                          acquisition_demo.csv
```

Each row in the CSV captures one (CoeffScalar, StateScalar,
SampleScalar) configuration's quality against the reference. The
schema matches the
[`AcquisitionPrecisionRow`](../../analysis/acquisition-precision/#csv-export)
format from the analysis module, so the same Python tooling that
reads `precision_sweep.csv` reads this file's identifier columns.

## ADC simulation

```cpp
mtl::vec::dense_vector<double> simulate_adc(int adc_bits, unsigned seed = 0xACDC);
```

A real-valued cosine at the IF frequency, plus low-level AWGN, then
uniform quantization to `adc_bits` levels across full scale. The
quantization step is `1 / 2^(adc_bits-1)` so the SNR floor follows the
classic $6.02 N + 1.76$ dB ceiling for any later stage downstream.

## DDC composition

```cpp
template <class CoeffScalar, class StateScalar, class SampleScalar>
std::vector<std::complex<double>>
run_ddc_pipeline(const mtl::vec::dense_vector<double>& adc_in_double);
```

The pipeline projects ADC samples into `SampleScalar`, runs them
through a `DDC<CoeffScalar, StateScalar, SampleScalar>` with a
polyphase FIR decimator (decimation factor 8), and casts the complex
output back to `double` for cross-precision comparison.

The DDC class itself does the work — see the [DDC reference](./ddc/) for
the math. The demo's contribution is the configuration sweep
infrastructure around it.

## Two mixed-precision design decisions

These come up the moment you try to instantiate the pipeline at
non-IEEE precision. They're worth understanding because they recur in
any real embedded DSP design.

### 1. Normalized rates

The `DDC` constructor takes a `StateScalar sample_rate` argument, but
the NCO inside only ever uses `frequency / sample_rate`. If you pass
absolute Hz (say `1.0e6`) and the StateScalar is a narrow type (e.g.,
`fixpnt<32, 28>` with range $\pm 8$), construction throws because the
sample rate saturates.

The fix is mathematically trivial:

```cpp
const double f_norm = params::kIfFrequencyHz / params::kSampleRateHz;
DDC_t ddc(static_cast<StateScalar>(f_norm),
          static_cast<StateScalar>(1.0),
          decim);
```

Pass `(IF/fs, 1.0)` instead of `(IF, fs)`. The phase increment ends up
identical (`f_norm / 1.0 == IF / fs`), and the StateScalar values stay
in $[0, 1]$ where every sane scalar type can represent them.

This is a general principle for embedded DSP: parameterize on
dimensionless ratios, not absolute units, whenever the underlying math
only needs the ratio.

### 2. Q4.28 (not Q16.16) for fixpnt

A first attempt at fixpnt parameterization picked `fixpnt<32, 16>`
(Q16.16 — 16 integer bits, 16 fractional). That gives ample integer
range ($\pm 32{,}768$) but only $2^{-16} \approx 1.5 \times 10^{-5}$
precision. The post-mixer baseband signal lives in $[-0.5, +0.5]$
where 16 fractional bits is poor. The configuration measured **0
ENOB**.

The fix is to choose the Q-format for the *signal range*, not for
some abstract "32-bit fixed-point". The demo uses `fixpnt<32, 28>`
(Q4.28):

- 4 integer bits → range $\pm 8$, comfortable for the post-mixer
  $[-0.5, 0.5]$ signal even after a few decibels of FIR-stage gain
- 28 fractional bits → precision $2^{-28} \approx 4 \times 10^{-9}$,
  competitive with IEEE float

Result: Q4.28 fixpnt measures ~24 ENOB through this pipeline, the
same neighborhood as IEEE float. The lesson is that fixpnt is a
*format choice*, not a number system; pick the integer/fractional
split based on the signal you'll actually run through it.

## Reading the output

A clean run with a 16-bit ADC at 1 MHz, IF at 100 kHz, decimating to
125 kHz:

```text
=== Number-system sweep (16-bit ADC) ===
Configuration                   Bits    SNR(dB)    ENOB
-------------------------------------------------------
uniform_double                   192        inf     ref
uniform_float                     96     144.62   23.73
uniform_posit32                   96     167.60   27.55
uniform_posit16                   48      71.30   11.55
uniform_cfloat32                  96     144.68   23.74
uniform_fixpnt32                  96     147.56   24.22
mixed_double_p32_p16             112      76.68   12.45
mixed_double_double_float        160     150.63   24.73
mixed_double_p32_float           128     150.59   24.72
mixed_double_float_float         128     144.62   23.73
mixed_double_fx32_fx32           128     147.56   24.22

=== ADC bit-depth scan (uniform-double pipeline) ===
ADC bits        SNR(dB)    ENOB
-------------------------------
8-bit             54.00    8.68
12-bit            77.98   12.66
14-bit            89.49   14.57
16-bit           101.75   16.61
```

A few patterns to note:

- **`uniform_posit32` wins on raw quality at 96 bits/config**, beating
  `uniform_float` by about 23 dB. Posit's tapered precision concentrates
  precision near unity, exactly where a normalized baseband signal lives
  after mixing — float spreads its precision across a much wider exponent
  range that the signal never visits.
- **`uniform_fixpnt32` (Q4.28) is competitive with float**. With the
  format chosen for the signal's actual dynamic range, fixpnt is a
  perfectly reasonable choice for embedded DSP.
- **The mixed-precision row `mixed_double_p32_p16`** keeps high-precision
  coefficients but runs state and samples narrower. SNR drops to ~77 dB
  (~12 ENOB) — that's the floor any 16-bit-streaming pipeline can
  reach, regardless of how much you spend on the coefficients.
- **The ADC scan** confirms the textbook $\text{SNR} \approx 6.02 N + 1.76$ dB
  ceiling propagates end-to-end: the FIR decimation gains a few dB of
  averaging beyond the raw ADC ceiling, but the slope and intercept
  match.

## Running it

```bash
cmake -B build
cmake --build build --target acquisition_demo
./build/applications/acquisition_demo/acquisition_demo [output.csv]
```

The CSV path is optional; default is `acquisition_demo.csv` in the
working directory. The console table is printed regardless.

## Extending the sweep

The demo's structure is intentionally flat — a single
`sweep_configurations()` function holds all the rows. To add a
configuration:

```cpp
rows.push_back(measure_config<MyCoeff, MyState, MySample>(
    "config_label",
    "MyCoeff",      // for the CSV
    "MyState",
    "MySample",
    coeff_bits + state_bits + sample_bits,
    adc_in,
    reference_out));
```

The reference is always uniform-double, computed once at the top of
the sweep. Every row is compared against the same reference, which is
what makes the SNR column meaningful as a cross-row metric.

## See also

- [DDC](./ddc/) — the pipeline's core component
- [Polyphase Decimator](./polyphase-decimator/) — channel-shaping stage
- [Decimation Chain](./decimation-chain/) — multi-stage rate reduction (alternative to a single-stage polyphase)
- [Acquisition Precision Analysis](../../analysis/acquisition-precision/) — the SNR/ENOB primitives the demo measures with
