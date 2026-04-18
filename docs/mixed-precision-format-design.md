# Mixed-Precision Arithmetic Format Design for DSP

This document describes the arithmetic number types targeted by `sw::dsp`
and the rationale for each format family. The goal is to provide
researchers and system designers with a comprehensive set of types
spanning five number systems — **integer**, **fixed-point**, **posit**,
**logarithmic (LNS)**, and **custom float** — so that every stage of
a DSP pipeline can be evaluated under the type that best matches its
numerical requirements.

## Historical context: mixed precision was always the practice

The three-scalar model in `sw::dsp` — separate types for coefficients,
state, and samples — is not a new idea. It formalizes what early DSP
engineers did implicitly from the very beginning of digital signal
processing in the 1950s and 1960s.

### The original mixed-precision pipeline

The earliest real-time DSP systems enforced mixed precision by hardware
necessity:

| Pipeline stage | Typical width | Reason |
|---------------|--------------|--------|
| ADC input samples | 12 bits | Converter technology limit |
| Filter coefficients | 12–16 bits | Memory cost per tap |
| MAC accumulator | 24–36 bits | Double-width to prevent overflow |
| DAC output samples | 12 bits | Converter technology limit |

This pattern appeared across Bell Labs digital telephony (Q15 coefficient
arithmetic with 32-bit accumulators), Lincoln Labs radar processing
(12-bit I/Q samples, 18-bit arithmetic on PDP-series machines), and
military vocoder implementations (12-bit input, 16-bit LPC coefficients).

### Key systems and their word lengths

**1950s — Vacuum tube era:**

- **Whirlwind I** (MIT, 1951): 16-bit fixed-point, ones'-complement.
  Earliest real-time computing; established 16 bits as a practical
  minimum.
- **AN/FSQ-7 SAGE** (IBM/Lincoln Labs, 1958): 32-bit fixed-point,
  ones'-complement. Real-time radar track-while-scan.
- **TX-2** (Lincoln Labs, 1958): 36-bit fixed-point, ones'-complement.
  Research platform for early digital filter experiments.

**1960s — Transistor era:**

- **PDP-1/4/7/9** (DEC, 1960–66): 18-bit twos'-complement. The
  workhorse machines for Gold, Rader, and other Lincoln Labs DSP
  researchers. The 18-bit word length appears throughout the
  foundational DSP literature.
- **PDP-8** (DEC, 1965): 12-bit twos'-complement. Demonstrated that
  useful DSP was possible at 12 bits for narrow-band applications.
- **CDC 1604/3600** (1960–63): 48-bit ones'-complement. Used for
  offline filter design and spectral analysis where quantization
  was not a constraint.

**Late 1960s — Early IC era:**

- **PDP-11** (DEC, 1970): 16-bit twos'-complement. Became the
  standard DSP research platform and cemented the Q15 convention.
- **Lincoln Labs FDP** (1971): 18-bit pipelined MAC. First
  purpose-built DSP computer; real-time 1024-point complex FFT.
- **Bell Labs DSP-1** (1979): 20-bit twos'-complement, Q19 format.
- **TMS32010** (TI, 1982): 16-bit Q15 with 32-bit accumulator.
  Industry standard that defined fixed-point DSP for decades.

### The Q-format convention

The **fractional fixed-point** convention became universal in DSP
because multiplying two numbers in [-1, +1) always produces a result
in [-1, +1) — overflow is impossible from multiplication alone.
Stable IIR filter coefficients naturally satisfy |coefficient| ≤ 1
(or can be scaled to do so), and ADC outputs are normalized to [-1, +1).

| Notation | Total bits | Integer | Fractional | Range |
|----------|-----------|---------|-----------|-------|
| Q7 | 8 | 0 | 7 | [-1.0, +0.992] |
| Q11 | 12 | 0 | 11 | [-1.0, +0.9995] |
| Q15 | 16 | 0 | 15 | [-1.0, +0.99997] |
| Q17 | 18 | 0 | 17 | [-1.0, +0.999992] |
| Q23 | 24 | 0 | 23 | [-1.0, +0.9999999] |
| Q31 | 32 | 0 | 31 | [-1.0, +0.9999999995] |

All include 1 sign bit. The `fixpnt<N, N-1>` format in Universal
corresponds to the standard Q(N-1) convention.

### What the foundational literature tells us

The quantitative results from Jackson (1970), Kaiser (1966), Knowles
& Edwards (1965), and Gold & Rader (1969) established:

- **Coefficient quantization** sensitivity grows as poles approach the
  unit circle. Narrow-band filters need more coefficient bits.
- **Roundoff noise** power in cascade-form IIR filters is proportional
  to 2^(-2b) where b is the number of fractional bits.
- **Biquad sections** are preferred over high-order direct forms
  precisely because they are less sensitive to coefficient
  quantization (Jackson 1970).
- **Minimum practical word length** for telephony-quality IIR filtering:
  12 bits for coefficients, 16 bits comfortable, 24-bit accumulators
  essential.

These constraints are not artifacts of 1960s hardware. They are
fundamental properties of the algorithms. Modern DSP on custom
arithmetic types encounters exactly the same trade-offs — which
is why this library exists.

---

## Number type families for DSP

### 1. Fixed-point (`fixpnt<nbits, rbits>`)

The natural format for DSP: deterministic latency, exact representation
of the quantization grid, and direct correspondence to hardware
multiplier widths.

#### Standard DSP fixed-point formats

**Sample I/O formats (Q(N-1) — fractional, normalized to [-1, +1)):**

| Universal type | Q-notation | Historical role |
|---------------|-----------|-----------------|
| `fixpnt<8,7>` | Q7 | 8-bit PCM, µ-law equivalent |
| `fixpnt<12,11>` | Q11 | 12-bit ADC (radar, sonar, early telephony) |
| `fixpnt<16,15>` | Q15 | 16-bit audio, TMS320 standard |
| `fixpnt<24,23>` | Q23 | Professional audio (24-bit ADC) |

**Coefficient formats (extra integer bit for headroom):**

| Universal type | Q-notation | Use case |
|---------------|-----------|----------|
| `fixpnt<16,14>` | Q2.14 | Coefficients with [-2, +2) range |
| `fixpnt<24,20>` | Q4.20 | Wide-range coefficients |
| `fixpnt<32,24>` | Q8.24 | High-precision design coefficients |

**Accumulator formats (double-width + guard bits):**

| Universal type | Q-notation | Use case |
|---------------|-----------|----------|
| `fixpnt<32,31>` | Q31 | Double-width for 16-bit paths |
| `fixpnt<32,16>` | Q16.16 | Wide dynamic range accumulator |
| `fixpnt<40,31>` | Q9.31 | 16×16 MAC + 8 guard bits |
| `fixpnt<48,32>` | Q16.32 | Double-width for 24-bit paths |

### 2. Posit (`posit<nbits, es>`)

Posits provide tapered precision — highest accuracy near 1.0 (where
DSP signals and coefficients concentrate) with gradual degradation
toward the extremes. No denormals, no NaN (except the single
Not-a-Real value at the antipodal point), and exact representation
of zero.

#### Standard posits (es=2)

The standard posit configuration uses es=2 throughout, enabling
trivial up/down conversion across the pipeline. These are the
posit equivalents of IEEE 754's float/double hierarchy:

| Name | Type | Bits | Role |
|------|------|------|------|
| p6 | `posit<6,2>` | 6 | Minimum viable DSP exploration |
| p8 | `posit<8,2>` | 8 | Ultra-low-precision edge/sensor |
| p12 | `posit<12,2>` | 12 | Compact real-time samples |
| p16 | `posit<16,2>` | 16 | Standard sample streaming |
| p24 | `posit<24,2>` | 24 | Balanced state accumulator |
| p32 | `posit<32,2>` | 32 | Coefficient design, high-precision state |

The consistent es=2 across all sizes means that narrowing from p32
to p16 to p8 is a well-defined projection — the exponent field
scales identically, and only regime/fraction bits are lost.

#### High-precision posits (es=0)

With es=0, the regime field directly encodes powers of 2 (useed = 2).
This gives **maximum fraction bits** at any given word length — the
posit equivalent of a fixed-point format, but with the regime providing
a modest dynamic range that fixed-point lacks.

| Type | Dynamic range | Precision near 1.0 |
|------|--------------|-------------------|
| `posit<4,0>` | ~4 decades | 1 fraction bit |
| `posit<6,0>` | ~4 decades | 3 fraction bits |
| `posit<8,0>` | ~5 decades | 5 fraction bits |
| `posit<12,0>` | ~6 decades | 9 fraction bits |
| `posit<16,0>` | ~7 decades | 13 fraction bits |

These are attractive as **coefficient representations** where values
cluster near ±1 and the extra fraction bits translate directly to
reduced coefficient quantization noise.

#### Moderate-range posits (es=1)

With es=1, useed = 4. A middle ground between es=0's precision focus
and es=2's dynamic range:

| Type | Dynamic range | Precision near 1.0 |
|------|--------------|-------------------|
| `posit<4,1>` | ~8 decades | 0 fraction bits |
| `posit<6,1>` | ~10 decades | 2 fraction bits |
| `posit<8,1>` | ~13 decades | 4 fraction bits |
| `posit<12,1>` | ~20 decades | 8 fraction bits |
| `posit<16,1>` | ~26 decades | 12 fraction bits |

These compete directly with fixed-point for **sample streaming**: they
offer more dynamic range than Q-format at the cost of non-uniform
quantization steps — a trade-off that may or may not matter depending
on the signal statistics and filter topology.

### 3. Custom float (`cfloat<nbits, es>`)

Miniature IEEE-like floating-point for ML/DSP crossover and for
studying the impact of floating-point format parameters on DSP quality.

| Universal type | Analogue | Use case |
|---------------|----------|----------|
| `cfloat<8,2>` | E2M5 | Narrow ML inference |
| `cfloat<8,4>` | FP8 E4M3 | ML training/inference |
| `cfloat<8,5>` | FP8 E5M2 | ML gradient format |
| `cfloat<16,5>` | IEEE binary16 | Standard half precision |
| `cfloat<16,8>` | bfloat16 | ML accelerator format |
| `cfloat<24,5>` | — | Extended half (more mantissa) |
| `cfloat<32,8>` | IEEE binary32 | Standard float |

### 4. Logarithmic Number System (`lns<nbits, rbits>`)

LNS represents values as fixed-point logarithms: x is stored as
log₂(|x|) in fixed-point plus a sign bit. Multiplication becomes
addition, division becomes subtraction — attractive for filter
coefficient application where multiply-accumulate dominates.

Addition is expensive in LNS (requires a table lookup or CORDIC),
so LNS is best suited for multiply-heavy pipelines where additions
can be deferred or are infrequent.

| Universal type | Fractional log bits | Use case |
|---------------|-------------------|----------|
| `lns<8,5>` | 5 | Ultra-compact multiplier-free DSP |
| `lns<12,8>` | 8 | Compact real-time, ~0.4% accuracy |
| `lns<16,10>` | 10 | Standard LNS word, ~0.1% accuracy |
| `lns<32,22>` | 22 | High-precision LNS |

### 5. Integer (`integer<nbits>`)

For quantization studies, ADC/DAC modeling, and direct correspondence
to converter word lengths:

| Universal type | Use case |
|---------------|----------|
| `integer<6>` | Minimum quantization studies |
| `integer<8>` | 8-bit ADC/PCM |
| `integer<10>` | 10-bit video ADC |
| `integer<12>` | 12-bit instrumentation ADC |
| `integer<14>` | 14-bit communications ADC |
| `integer<16>` | 16-bit audio ADC |
| `integer<24>` | 24-bit professional audio ADC |

---

## The precision sweep matrix

The precision sweep tool (planned for v0.5.0) should evaluate filters
across a representative subset of these types. The default sweep
configuration:

### Coefficient types (high precision)
`double`, `float`, `posit<32,2>`, `posit<16,0>`, `fixpnt<32,24>`,
`fixpnt<16,14>`, `lns<32,22>`

### State/accumulator types (medium-high precision)
`double`, `float`, `posit<32,2>`, `posit<24,2>`, `fixpnt<32,31>`,
`fixpnt<40,31>`, `lns<16,10>`

### Sample types (variable precision)
`float`, `posit<16,2>`, `posit<12,2>`, `posit<8,2>`, `posit<16,1>`,
`posit<8,1>`, `fixpnt<16,15>`, `fixpnt<12,11>`, `fixpnt<8,7>`,
`cfloat<16,5>`, `cfloat<8,4>`, `integer<16>`, `integer<12>`,
`integer<8>`, `lns<16,10>`, `lns<8,5>`

---

## References

- Jackson, L.B. "Roundoff-noise analysis for fixed-point digital
  filters realized in cascade or parallel form," *IEEE Trans. Audio
  Electroacoustics*, AU-18(2), pp. 107–122, June 1970.
- Kaiser, J.F. "Digital Filters," in *System Analysis by Digital
  Computer*, Kuo & Kaiser, eds., Wiley, 1966.
- Knowles, J.B. and Edwards, E.M. "Effect of a finite-word-length
  computer in a sampled-data feedback system," *Proc. IEE*, 112(6),
  pp. 1197–1207, June 1965.
- Gold, B. and Rader, C.M. *Digital Processing of Signals*, McGraw-Hill,
  1969.
- Oppenheim, A.V. and Weinstein, C.J. "Effects of finite register length
  in digital filtering and the fast Fourier transform," *Proc. IEEE*,
  60(8), pp. 957–976, August 1972.
- Gustafson, J.L. "Posit Arithmetic," Technical Report, 2017.
- Oppenheim, A.V. and Schafer, R.W. *Discrete-Time Signal Processing*,
  Prentice Hall, 1975.
- Weste, N.H.E. and Harris, D.M. *CMOS VLSI Design*, 4th ed.,
  Addison-Wesley, 2011. (Energy and area estimates for arithmetic units.)
