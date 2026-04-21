---
title: Numerically Controlled Oscillator (NCO)
description: Phase-accumulator NCO for complex sinusoid generation in DDC/DUC chains
---

## History and Motivation

A Numerically Controlled Oscillator (NCO) generates complex sinusoids
(I/Q pairs) by accumulating a phase value and computing sine/cosine at
each sample instant. NCOs are the core local oscillator in digital
down-converters (DDC) and up-converters (DUC), where they translate
signals between RF/IF and baseband.

The key property of an NCO is that its **spurious-free dynamic range
(SFDR)** is determined by the phase accumulator width:

$$
\text{SFDR} \approx 6.02 \times W \;\text{dB}
$$

where $W$ is the effective number of bits in the phase accumulator. This
makes the NCO a prime candidate for mixed-precision optimization: posit
arithmetic's tapered precision near $\pm 1$ (where sin/cos outputs
concentrate) can yield better SFDR than fixed-point at the same bit width.

## Where NCO Fits in the Pipeline

```text
┌─────────┐    ┌─────────────┐    ┌───────┐    ┌───────────┐    ┌──────────┐
│ ADC     │───>│ NCO         │───>│ Mixer │───>│ CIC       │───>│ Half-Band│
│ 1 GSPS  │    │ (LO)        │    │ ×conj │    │ Decimator │    │ ÷2       │
│ 12-bit  │    │ I/Q gen     │    │       │    │ ÷64       │    │          │
└─────────┘    └────────���────┘    └───────┘    └───────────┘    └───────���──┘
                 StateT             SampleT       StateT          CoeffT
                 (phase acc)        (mixed)       (wide)          /StateT
```

The NCO generates the complex local oscillator signal. Multiplying the
ADC samples by the conjugate of this signal shifts the desired frequency
band to baseband, where subsequent CIC and half-band stages perform
decimation.

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

### Output Generation

At each sample, the NCO computes:

$$
\text{I}[n] = \cos(2\pi \cdot \phi[n]), \quad
\text{Q}[n] = \sin(2\pi \cdot \phi[n])
$$

The output is a complex sinusoid: $y[n] = \text{I}[n] + j\,\text{Q}[n]$.

### SFDR and Precision

| Scalar Type | Effective Bits | Theoretical SFDR | Measured SFDR |
|-------------|---------------|-----------------|---------------|
| `float` | ~24 | ~144 dB | ~168 dB |
| `double` | ~53 | ~319 dB | ~320 dB |
| `posit<32,2>` | ~30 | ~180 dB | TBD |

The measured SFDR often exceeds the theoretical estimate because the
phase-to-sinusoid computation via `std::sin`/`std::cos` does not
introduce the truncation spurs that a lookup-table NCO would.

### Normalized Phase vs. Radians

Storing phase in normalized $[0, 1)$ units rather than radians $[0, 2\pi)$
avoids precision loss: for long-running oscillators, radian-domain phase
values grow large, and the modular reduction $\bmod\, 2\pi$ loses
significant bits. In normalized units, the phase is always bounded,
preserving full accumulator precision.

## API

### Construction

```cpp
#include <sw/dsp/acquisition/nco.hpp>

using namespace sw::dsp;

// NCO at 1 kHz with 48 kHz sample rate
NCO<double> nco(1000.0, 48000.0);

// Mixed precision: float phase accumulator, double output
NCO<float, double> nco_mixed(1000.0f, 48000.0f);
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
std::vector<std::complex<double>> buf(1024);
nco.generate_block(std::span<std::complex<double>>(buf));

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
// Change frequency mid-stream (phase is preserved)
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

// Extract real part for CIC (or process I and Q channels separately)
std::vector<double> cic_out;
// ... feed real/imag parts through CIC and half-band stages
```

## Precision Considerations

The NCO uses a two-scalar model:

- **StateScalar** holds the phase accumulator. Higher precision here
  directly improves SFDR by reducing phase quantization noise. This is
  the primary knob for trading compute cost against spectral purity.

- **SampleScalar** holds the output I/Q values. Since sin/cos outputs
  are bounded to $[-1, +1]$, posit types with tapered precision near
  these values can represent the output more efficiently than uniform
  fixed-point.

```cpp
// Posit phase accumulator for high SFDR at 32 bits
using p32 = sw::universal::posit<32, 2>;
NCO<p32, p32> nco_posit(p32(1000.0), p32(48000.0));
```

### Comparison with Other Acquisition Components

| Property | NCO | CIC | Half-Band |
|----------|-----|-----|-----------|
| Scalar model | Two-scalar | Two-scalar | Three-scalar |
| Multiplications | 0 (sin/cos call) | 0 | ~N/4 per output |
| Key precision dimension | Phase accumulator | Integrator width | Coefficient precision |
| Output rate | Same as input | Reduced by R | Reduced by 2 |
| SFDR dependence | Accumulator bits | N/A | Stopband rejection |
