---
title: Digital Down-Converter (DDC)
description: Composing NCO, complex mixer, and decimation filter for band selection at high sample rates
---

## History and Motivation

### From Superheterodyne to All-Digital Receivers

The digital down-converter is the direct descendant of the analog
superheterodyne receiver, patented by Edwin Armstrong in 1918. A
superheterodyne mixes the incoming radio signal with a local oscillator
to translate a band of interest to an intermediate frequency (IF), where
it can be filtered by fixed-tuned, high-Q components. For most of the
twentieth century this was the dominant architecture for radio — every
AM and FM broadcast receiver, every military radio, every television
tuner used some form of heterodyne mixing.

The transition from analog to digital receivers began in the 1980s, when
ADCs became fast enough to sample directly at IF rather than at baseband.
This "IF sampling" architecture — pioneered in military and aerospace
applications and later commercialized for cellular base stations —
replaced the final analog mixer and baseband filter with a digital
front-end: an ADC running at tens of MHz followed by a digital
down-converter. The DDC performs the same function as the analog
block it replaced (translate to baseband, filter to the channel
bandwidth), but does it with numeric precision, channel agility, and
zero analog drift.

### The DDC ASIC Era

The early 1990s saw a wave of dedicated DDC integrated circuits from
Harris (later Intersil), Graychip (later acquired by Texas Instruments),
and Analog Devices. The Harris HSP50016 "Digital Down Converter" (1992)
combined a 32-bit NCO, a complex mixer, and a fifth-order CIC decimator
in a single part — exactly the block diagram this module implements.
The Graychip GC4016 followed in the late 1990s with four independent
channels and programmable FIR decimation, targeting the then-booming
cellular infrastructure market. Analog Devices' AD6620 combined a CIC
pre-decimator with two cascaded half-band filters and a general-purpose
polyphase FIR, establishing what is still the canonical DDC chain
architecture.

These parts all shared the same signal flow:

$$
x[n] \longrightarrow \times \longrightarrow h_{\text{lp}}[n] \longrightarrow \downarrow R \longrightarrow y[m]
$$

where $x[n]$ is a real sample from the ADC, the multiplier mixes with
the complex NCO output $\exp(-j\omega_0 n)$, $h_{\text{lp}}[n]$ is a
lowpass decimation filter, and $\downarrow R$ decimates by the rate
reduction factor $R$. What varied between parts was the choice of
decimation filter — CIC for efficiency at high rates, half-band for
stopband attenuation near 2x, full polyphase FIR for arbitrary shaping.

### The Software-Defined Radio Era

When FPGAs grew large enough to implement entire DDC chains around 2000,
the dedicated ASIC DDC faded. Software-defined radios such as the Ettus
USRP and the various GNU Radio flows adopted the same architecture in
reconfigurable logic, adding the flexibility to retune instantaneously
and to instantiate many parallel channels. Today, nearly every wireless
system — cellular handsets, Wi-Fi access points, satellite receivers,
digital oscilloscopes, spectrum analyzers, radar receivers — contains
at least one DDC chain, whether as dedicated silicon, as an FPGA block,
or as code running on a DSP core.

### Why Mixed-Precision Matters

The DDC is a prime candidate for mixed-precision because its stages have
asymmetric precision requirements. The NCO's phase accumulator needs
many bits to achieve good SFDR (see the [NCO documentation](./nco/)),
but its sine/cosine output only needs enough bits to meet the system
dynamic range. The mixer multiplies two full-range signals and can use
narrower state than either input. The decimation filter has its own
bit-growth rules: a CIC's accumulator width depends on $M \lceil
\log_2(RD) \rceil$ of bit growth, while a polyphase FIR's accumulator
need only be wide enough to avoid coefficient quantization.

Posit arithmetic's tapered precision near $\pm 1$ — where the NCO
sinusoids and most normalized signals live — can deliver the same
spectral purity as IEEE-754 float at fewer bits. This is most
pronounced in long DDC chains where rounding errors accumulate.

## What the DDC Computes

Given a real input sequence $x[n]$ sampled at rate $f_s$ and a
target center frequency $f_0$, the DDC produces a complex baseband
sequence $y[m]$ at the decimated rate $f_s / R$:

$$
y[m] = \sum_{n} h_{\text{lp}}[mR - n] \cdot x[n] \cdot e^{-j 2\pi f_0 n / f_s}
$$

The complex multiplication is performed on each input sample (at the
high rate), and the lowpass filter output is sampled at the lower rate.
In efficient implementations the filter runs at the decimated rate via
polyphase decomposition or via the CIC structure, so the arithmetic
cost per input sample is much less than one full filter evaluation.

### Spectral Interpretation

A real tone at $f_0$ has spectral energy at both $+f_0$ and $-f_0$.
After multiplication by $e^{-j2\pi f_0 n / f_s}$:

- The $+f_0$ component shifts to DC.
- The $-f_0$ component shifts to $-2f_0$.

The lowpass filter is responsible for rejecting the $-2f_0$ image (and
any other out-of-band energy) before decimation. A properly designed
DDC has passband wide enough to pass the signal of interest and stopband
attenuation sufficient to suppress out-of-band energy below the system
noise floor.

### Tuning Range and Resolution

The NCO determines the DDC's tuning range ($\pm f_s / 2$, any frequency
in the first Nyquist zone) and frequency resolution:

$$
\Delta f = \frac{f_s}{2^W}
$$

for a $W$-bit phase accumulator. A 32-bit NCO at $f_s = 100\,\text{MHz}$
resolves to $\sim 0.023\,\text{Hz}$ — finer than any practical channel
spacing.

## How the DDC Works in this Library

The `sw::dsp::DDC<CoeffScalar, StateScalar, SampleScalar, Decimator>`
class composes three components:

- An `NCO<StateScalar, SampleScalar>` that generates the complex local
  oscillator.
- A pair of identical `Decimator` instances — one for the in-phase (I)
  stream, one for the quadrature (Q) stream. The prototype decimator
  passed to the constructor is copied twice; both copies are initialized
  to the same state and fed identical sample counts, so they always emit
  on the same cycle.
- A streaming mixer that multiplies the real input by the NCO's
  conjugate: `y[n] = x[n] * conj(lo[n])`.

### Choice of Decimator

The `Decimator` template parameter selects the decimation filter type.
The library supports three options out of the box, dispatched
automatically by an internal adapter:

| Decimator | When to use | API |
|-----------|-------------|-----|
| `PolyphaseDecimator` | General-purpose, arbitrary $R$ and sharp cutoff | `process(x)` returns `{ready, y}` |
| `HalfBandFilter` | Fixed 2:1 decimation, ~75% zero taps | `process_decimate(x)` returns `{ready, y}` |
| `CICDecimator` | High decimation ratios at very high rates; no multiplies | `push(x)`, `output()` |

Use a CIC as the first stage when the input is ADC samples at GHz rates,
followed by a half-band or polyphase for channel shaping. See the
[CIC](./cic/) and [Half-Band](./halfband/) documentation for the
individual stage designs.

### Construction

```cpp
#include <sw/dsp/acquisition/ddc.hpp>
#include <sw/dsp/filter/fir/polyphase.hpp>
#include <sw/dsp/filter/fir/fir_design.hpp>
#include <sw/dsp/windows/hamming.hpp>

using namespace sw::dsp;

double fs     = 48000.0;   // input sample rate
double f_if   = 6000.0;    // band of interest
std::size_t R = 4;         // decimation ratio

// Design anti-alias taps at the input rate
auto window = hamming_window<double>(65);
auto taps   = design_fir_lowpass<double>(65, 0.45 / R, window);

PolyphaseDecimator<double> decim(taps, R);
DDC<double> ddc(f_if, fs, decim);
```

### Streaming

```cpp
for (std::size_t n = 0; n < input_size; ++n) {
    auto [ready, z] = ddc.process(input[n]);
    if (ready) {
        // z is a complex baseband sample at fs / R
        process_baseband(z);
    }
}
```

### Block Processing

```cpp
auto output = ddc.process_block(input);  // dense_vector<complex_for_t<SampleScalar>>
```

### Retuning

```cpp
ddc.set_center_frequency(new_f);
```

Retuning changes the NCO phase increment but does not clear the
decimator state, so a brief transient is visible at the output.

## Worked Example: Tone-to-Baseband

A real tone at the NCO center frequency translates to a DC complex
output with magnitude $\tfrac{1}{2}$:

```cpp
// Input: x[n] = cos(2 * pi * f_if * n / fs)
for (std::size_t n = 0; n < N; ++n) {
    input[n] = std::cos(2.0 * pi * f_if * n / fs);
}

auto out = ddc.process_block(input);

// After the filter transient, |out[m]| ≈ 0.5, arg(out[m]) ≈ 0
```

The factor of $\tfrac{1}{2}$ comes from the decomposition of the real
cosine into two complex exponentials; only one of them falls inside the
filter passband.

## Architecture Diagrams

### Single-Stage DDC

```
  ADC input (real, fs)
         │
         ▼
      ┌─────┐        ┌──────────────┐
      │  ×  │◄──────►│  NCO (conj)  │
      └─────┘        └──────────────┘
         │
         ▼  (complex, fs)
      ┌──────────────────┐
      │ lowpass + ↓R     │
      └──────────────────┘
         │
         ▼  (complex, fs/R)
     baseband I/Q
```

### Multi-Stage DDC (Typical ASIC Architecture)

```
  ADC (fs)  →  mix  →  CIC ↓R1  →  half-band ↓2  →  half-band ↓2  →  FIR ↓1  →  I/Q
              ▲
              NCO
```

Large rate reductions are split across stages so each filter can be
designed to its own passband/stopband requirements. The CIC handles the
bulk of the rate reduction efficiently; half-bands provide deep stopband
attenuation near the 2x point; a final polyphase FIR shapes the channel
response.

## Template Parameters

```cpp
template <DspField CoeffScalar = double,
          DspField StateScalar = CoeffScalar,
          DspField SampleScalar = StateScalar,
          class Decimator = PolyphaseDecimator<CoeffScalar, StateScalar, SampleScalar>>
class DDC;
```

- `CoeffScalar` — filter tap precision (design precision)
- `StateScalar` — NCO phase accumulator and decimator state
- `SampleScalar` — input samples and decimator output samples
- `Decimator` — the decimator type; any class exposing `process`,
  `process_decimate`, or `push`/`output` (see table above)

## Historical References

- E. H. Armstrong, *"Some Recent Developments in the Audion Receiver,"*
  Proceedings of the IRE, 1915 — the original superheterodyne.
- J. Tierney, C. Rader, B. Gold, *"A Digital Frequency Synthesizer,"*
  IEEE Transactions on Audio and Electroacoustics, 1971 — the NCO
  foundation the DDC builds on.
- E. B. Hogenauer, *"An Economical Class of Digital Filters for
  Decimation and Interpolation,"* IEEE Transactions on Acoustics,
  Speech, and Signal Processing, 1981 — the CIC structure used as the
  first DDC stage.
- F. Harris, *"Multirate Signal Processing for Communication Systems,"*
  Prentice Hall, 2004 — comprehensive treatment of the DDC and its
  multirate building blocks.
- Harris Semiconductor, *HSP50016 Digital Down Converter Datasheet*,
  1992 — the canonical DDC ASIC that defined the reference architecture.
