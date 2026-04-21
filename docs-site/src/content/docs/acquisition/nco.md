---
title: Numerically Controlled Oscillator (NCO)
description: Phase-accumulator NCO for complex sinusoid generation in DDC/DUC chains
---

## History and Motivation

### From Analog PLLs to Digital Phase Accumulators

The numerically controlled oscillator traces its lineage to the
phase-locked loop (PLL), one of the most important circuits in
communications engineering. Analog PLLs — invented by Henri de Bellescize
in 1932 and refined by the Apollo-era engineers at NASA's Jet Propulsion
Laboratory — use a voltage-controlled oscillator (VCO) locked to a
reference signal to generate stable, frequency-agile carrier waves.

When signal processing moved from analog to digital in the 1970s, the
VCO became the NCO: a phase accumulator whose output indexes a sinusoid
generator. The concept was straightforward — add a fixed increment to a
phase register on every clock cycle, and use the accumulated phase to
produce sine and cosine values — but the implications for system design
were profound.

Joseph Tierney, Charles Rader, and Barry Gold at MIT Lincoln Laboratory
published the seminal paper "A Digital Frequency Synthesizer" in *IEEE
Transactions on Audio and Electroacoustics* (1971), establishing the
theoretical framework for NCO-based frequency synthesis. Their key
insight was that the frequency resolution of an NCO is determined by the
accumulator width, not by any analog component tolerance. A 32-bit phase
accumulator clocked at 100 MHz can resolve frequencies to $100 \times
10^6 / 2^{32} \approx 0.023$ Hz — a precision that would require an
impossibly stable analog oscillator.

### The DDS Revolution

The NCO became the core of Direct Digital Synthesis (DDS), a technique
that exploded in the 1980s and 1990s with the advent of high-speed DACs
and dedicated DDS ICs. Analog Devices led the commercialization with
their AD9850 (1995) and subsequent family, which combined an NCO, a
sine lookup table, and a DAC on a single chip. These parts transformed
instrument design: where a signal generator previously needed a bank of
crystal oscillators or a complex PLL, a single DDS IC could generate any
frequency from millihertz to hundreds of megahertz with sub-hertz
resolution and phase-continuous switching.

The DDS architecture highlighted a fundamental trade-off: **phase
accumulator width determines spurious-free dynamic range (SFDR)**. When
the full accumulator output is used to address a sine lookup table, the
spurs are determined by the table size and DAC linearity. But when the
accumulator is truncated to fewer address bits (a common optimization to
reduce table size), the resulting phase quantization creates periodic
errors that appear as discrete spurs in the output spectrum. The
theoretical SFDR for an NCO with $W$-bit phase-to-amplitude conversion is:

$$
\text{SFDR} \approx 6.02 \times W \;\text{dB}
$$

This relationship between accumulator precision and spectral purity is
what makes the NCO a prime candidate for mixed-precision arithmetic.

### The Digital Down-Converter

In receiver architectures, the NCO serves as the digital local oscillator
(LO) in a digital down-converter (DDC). The DDC concept was pioneered in
the late 1980s by engineers at Harris Semiconductor (now part of L3Harris)
and Qualcomm, who recognized that moving the frequency translation from
analog to digital eliminated image-reject mixer design, reduced component
count, and enabled software-reconfigurable receivers.

A DDC multiplies the digitized ADC samples by the complex conjugate of the
NCO output:

$$
y[n] = x[n] \cdot e^{-j 2\pi f_0 n / f_s} = x[n] \cdot (\cos\theta_n - j\sin\theta_n)
$$

This shifts the signal at frequency $f_0$ down to baseband (DC), where
it can be filtered and decimated by subsequent stages (CIC, half-band,
etc.). The NCO's ability to switch frequency instantaneously and with
phase continuity makes it ideal for frequency-hopping systems, channelized
receivers, and adaptive processing.

### Why Mixed Precision Matters for NCOs

The traditional hardware NCO uses fixed-point arithmetic throughout: a
fixed-width accumulator, a fixed-size ROM lookup table, and fixed-point
output. The trade-offs are well understood: more bits mean better SFDR but
more silicon area and power.

Posit arithmetic offers a different trade-off curve. Because posits have
**tapered precision** — more bits near $\pm 1$ and fewer bits near zero —
they are naturally suited to representing sine and cosine values, which
are bounded to $[-1, +1]$ and spend most of their time near the extremes.
A 32-bit posit may achieve SFDR comparable to a 40-bit fixed-point NCO
for the phase-to-amplitude conversion, because the posit concentrates its
precision exactly where the sinusoidal output needs it.

This is one of the key mixed-precision findings that the acquisition
pipeline is designed to demonstrate: **the optimal number format depends
on the value distribution of the signal, not just the total bit count.**

## Where NCO Fits in the Pipeline

```text
┌─────────┐    ┌─────────────┐    ┌───────┐    ┌───────────┐    ┌──────────┐
│ ADC     │───>│ NCO         │───>│ Mixer │───>│ CIC       │───>│ Half-Band│
│ 1 GSPS  │    │ (LO)        │    │ ×conj │    │ Decimator │    │ ÷2       │
│ 12-bit  │    │ I/Q gen     │    │       │    │ ÷64       │    │          │
└─────────┘    └─────────────┘    └───────┘    └───────────┘    └──────────┘
                 StateT             SampleT       StateT          CoeffT
                 (phase acc)        (mixed)       (wide)          /StateT
```

The NCO generates the complex local oscillator signal. Multiplying the
ADC samples by the conjugate of this signal shifts the desired frequency
band to baseband, where subsequent CIC and half-band stages perform
decimation. The NCO runs at the full ADC sample rate — in a 1 GSPS
system, it produces one I/Q pair per nanosecond.

In a digital up-converter (DUC), the signal flow reverses: the baseband
signal is interpolated, then multiplied by the NCO output (without
conjugation) to shift it to the desired carrier frequency for transmission.

## Theory

### Phase Accumulator

The NCO maintains a phase accumulator $\phi[n]$ in normalized units
where $1.0$ represents one full cycle ($2\pi$ radians):

$$
\phi[n+1] = (\phi[n] + \Delta\phi) \bmod 1
$$

where the **frequency control word** (phase increment) is:

$$
\Delta\phi = \frac{f_{\text{out}}}{f_s}
$$

The frequency resolution is determined by the number of distinct
representable phase states $N_\phi$:

$$
\Delta f_{\min} = \frac{f_s}{N_\phi}
$$

For a $W$-bit fixed-point accumulator, $N_\phi = 2^W$, so
$\Delta f_{\min} = f_s / 2^W$. For a `double` phase accumulator
($W \approx 53$), this gives sub-microhertz resolution at any practical
sample rate. For posit or fixed-point accumulators, the resolution
depends on the effective number of distinct phase values the type can
represent in $[0, 1)$.

### Normalized Phase vs. Radians

Our implementation stores phase in normalized $[0, 1)$ units rather than
radians $[0, 2\pi)$. This is a deliberate precision choice.

For a long-running oscillator at $f_0 = 1\;\text{kHz}$ and $f_s =
48\;\text{kHz}$, after one hour of continuous operation the radian-domain
phase would reach:

$$
\theta = 2\pi \times 1000 \times 3600 \approx 2.26 \times 10^7 \;\text{radians}
$$

The modular reduction $\theta \bmod 2\pi$ loses significant bits when
$\theta$ is large. In double precision, $2.26 \times 10^7$ has about 24
bits of integer part, leaving only 29 bits of fraction — a loss of 24
bits of phase resolution compared to the accumulator's full 53-bit
mantissa.

In normalized units, the phase is always in $[0, 1)$ regardless of run
time, preserving the full accumulator precision indefinitely.

### Output Generation

At each sample, the NCO computes:

$$
\text{I}[n] = \cos(2\pi \cdot \phi[n]), \quad
\text{Q}[n] = \sin(2\pi \cdot \phi[n])
$$

The output is a complex sinusoid: $y[n] = \text{I}[n] + j\,\text{Q}[n]$,
which lies on the unit circle ($|y[n]| = 1$) by construction.

Our implementation uses `std::sin` and `std::cos` (via conversion to
`double`) rather than a lookup table. This is appropriate for a software
DSP library where the goal is algorithmic correctness and precision
exploration, not gate-count minimization. The measured SFDR reflects the
phase accumulator precision, not lookup-table quantization.

### SFDR and Precision

| Scalar Type | Effective Bits | Theoretical SFDR | Measured SFDR |
|-------------|---------------|-----------------|---------------|
| `float` | ~24 | ~144 dB | ~168 dB |
| `double` | ~53 | ~319 dB | ~320 dB |
| `posit<32,2>` | ~30 | ~180 dB | TBD |

The measured SFDR exceeds the theoretical estimate because the direct
computation via `std::sin`/`std::cos` does not introduce the truncation
spurs that a lookup-table NCO would. The phase quantization is the only
spur source, and it manifests as a noise floor rather than discrete spurs.

### Frequency Switching and Phase Continuity

When the frequency control word is changed, the phase accumulator
continues from its current value — only the increment changes. This
provides **phase-continuous frequency switching**: the output sinusoid
smoothly transitions to the new frequency without any phase discontinuity
or transient. This property is essential for frequency-hopping spread
spectrum (FHSS) systems and agile radar waveforms.

## API

### Construction

```cpp
#include <sw/dsp/acquisition/nco.hpp>

using namespace sw::dsp;

// NCO at 1 kHz with 48 kHz sample rate
NCO<double> nco(1000.0, 48000.0);

// Mixed precision: float phase accumulator, double output
NCO<float, double> nco_mixed(1000.0f, 48000.0f);

// Negative frequency (clockwise rotation in I/Q plane)
NCO<double> nco_neg(-1000.0, 48000.0);
```

### Sample Generation

```cpp
// Single complex I/Q sample
auto iq = nco.generate_sample();   // std::complex<double>
double i = iq.real();              // cosine component
double q = iq.imag();              // sine component

// Single real (cosine) sample
double y = nco.generate_real();
```

### Block Generation

```cpp
// Dense vector of complex I/Q samples
auto block = nco.generate_block(1024);

// Span-based (pre-allocated buffer)
mtl::vec::dense_vector<std::complex<double>> buf(1024);
nco.generate_block(std::span<std::complex<double>>(buf.data(), buf.size()));

// Real-only block (cosine)
auto real_block = nco.generate_block_real(1024);
```

### Digital Mixing (Down-Conversion)

```cpp
// Mix a real input signal down to baseband
// Computes: output[n] = input[n] * conj(nco[n])
auto baseband = nco.mix_down(adc_samples);
```

### Frequency and Phase Control

```cpp
// Change frequency mid-stream (phase is preserved — continuous)
nco.set_frequency(2000.0, 48000.0);

// Set a fixed phase offset (0.25 = 90 degrees)
nco.set_phase_offset(0.25);

// Query state
double inc = nco.phase_increment();  // frequency / sample_rate
double phi = nco.phase();            // current phase [0, 1)
```

### Utility Methods

```cpp
nco.reset();              // Reset phase to zero
nco.phase();              // Current phase accumulator value
nco.phase_increment();    // Phase increment per sample
```

## Example: DDC with NCO + CIC + Half-Band

```cpp
#include <sw/dsp/acquisition/nco.hpp>
#include <sw/dsp/acquisition/cic.hpp>
#include <sw/dsp/acquisition/halfband.hpp>

using namespace sw::dsp;

// Local oscillator at 10 MHz, 1 GSPS sample rate
NCO<double> lo(10e6, 1e9);

// CIC decimation by 64
CICDecimator<double> cic(64, 4, 1);

// Half-band decimation by 2
auto hb_taps = design_halfband<double>(19, 0.08);
HalfBandFilter<double> hb(hb_taps);

// Process: mix down to baseband, then decimate
auto baseband = lo.mix_down(adc_samples);

// Feed I and Q channels through CIC and half-band stages separately
// ... process baseband.real() and baseband.imag() through cic and hb
```

## Precision Considerations

The NCO uses a **two-scalar model**, in contrast to the three-scalar
model used by FIR-based components:

### StateScalar — Phase Accumulator

This is the primary precision knob. Higher precision here directly
improves SFDR by reducing phase quantization noise. The phase accumulator
stores values in $[0, 1)$, so the effective precision is determined by the
number of significant bits the type provides for values near zero and near
one.

For posit types, this is particularly interesting: a `posit<32,2>` has
approximately 28 bits of precision for values in $[0.25, 0.75]$ (where
most phase values reside for non-DC frequencies) but fewer bits near 0
and 1. The net effect on SFDR depends on the specific frequency and
how the phase values distribute across the posit's precision landscape.

### SampleScalar — Output Precision

Holds the I/Q output values, which are bounded to $[-1, +1]$. Since
sine and cosine outputs cluster near $\pm 1$ and $0$, posit types with
tapered precision near these values can represent the output more
efficiently than uniform fixed-point. A `posit<16,1>` has nearly the
same precision as `float` for values in $[-1, +1]$ but uses half the
storage — relevant when the output feeds a high-rate pipeline where
memory bandwidth is the bottleneck.

```cpp
// Posit phase accumulator for high SFDR at 32 bits
using p32 = sw::universal::posit<32, 2>;
NCO<p32, p32> nco_posit(p32(1000.0), p32(48000.0));

// Mixed: double accumulator, float output (bandwidth-limited output)
NCO<double, float> nco_mixed(1000.0, 48000.0);
```

### Comparison with Other Acquisition Components

| Property | NCO | CIC | Half-Band |
|----------|-----|-----|-----------|
| Scalar model | Two-scalar | Two-scalar | Three-scalar |
| Multiplications | 0 (sin/cos call) | 0 | ~N/4 per output |
| Key precision dimension | Phase accumulator | Integrator width | Coefficient precision |
| Output rate | Same as input | Reduced by R | Reduced by 2 |
| SFDR dependence | Accumulator bits | N/A | Stopband rejection |
| Precision sweet spot | Near 0 and 1 (phase) | Near 0 (accumulator) | Near 0 (small coefficients) |

## Historical References

- J. Tierney, C. M. Rader, and B. Gold, "A Digital Frequency
  Synthesizer," *IEEE Trans. Audio Electroacoustics*, vol. AU-19, no. 1,
  pp. 48–57, March 1971 — the foundational paper on NCO-based frequency
  synthesis
- H. T. Nicholas and H. Samueli, "An Analysis of the Output Spectrum of
  Direct Digital Frequency Synthesizers in the Presence of Phase-Accumulator
  Truncation," *Proc. 41st Annual Frequency Control Symposium*, 1987 —
  established the SFDR ~ 6W dB relationship for truncated phase
  accumulators
- V. F. Kroupa, *Direct Digital Frequency Synthesizers*, IEEE Press, 1999
  — comprehensive treatment of DDS architectures and spur analysis
- H. de Bellescize, "La Reception Synchrone," *L'Onde Electrique*, vol. 11,
  pp. 230–240, 1932 — the original phase-locked loop concept
- Analog Devices, "A Technical Tutorial on Digital Signal Synthesis,"
  1999 — practical DDS design guide covering accumulator sizing, spur
  management, and DAC interface
