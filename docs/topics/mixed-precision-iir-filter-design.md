# Mixed-Precision IIR Filter Design and Deployment

A practitioner's guide to numerical sensitivity in IIR filter design
and how to select contracted arithmetic types for energy-efficient deployment.

---

## 1. Why Precision Matters in IIR Filters

An IIR (Infinite Impulse Response) filter feeds back its own output,
creating a recursive structure where numerical errors accumulate and
interact across time steps. Unlike FIR filters, where a coefficient error
simply shifts one tap's weight, an IIR coefficient error shifts a *pole*
— and poles control the entire dynamic behavior of the system.

The classical IIR design pipeline has five stages, each with distinct
precision requirements:

```
Analog Prototype  →  Frequency Transform  →  Bilinear Transform  →  Cascade  →  Processing
  (pole placement)    (Constantinides)        (s-plane to z-plane)   (biquads)    (samples)
       ↑                    ↑                       ↑                   ↑            ↑
   needs high           needs high              needs high          can be        can be
   precision            precision               precision           narrower      narrow
```

The first three stages are *design-time* computations that produce the
filter coefficients. The last two stages are *runtime* computations that
process data. This distinction is the foundation of mixed-precision filter
design: **design in high precision, deploy in contracted precision.**

---

## 2. Three Examples of Numerical Sensitivity

### Example 1: The Condition Number Explosion

The *condition number* of a filter measures how sensitive its frequency
response is to small perturbations in its coefficients. A condition number
of 10⁶ means a 1-part-per-million coefficient error can shift the response
by up to 100%. Measured from our analysis tools:

| Filter Design | Order | Cutoff | Condition Number |
|---------------|-------|--------|-----------------|
| Butterworth LP | 2 | 1 kHz / 44.1 kHz | 1.4 × 10⁶ |
| Butterworth LP | 4 | 1 kHz / 44.1 kHz | 3.1 × 10⁸ |
| Butterworth LP | 8 | 1 kHz / 44.1 kHz | 1.4 × 10¹³ |
| Chebyshev I LP | 8 | 1 kHz / 44.1 kHz | 6.9 × 10¹⁴ |
| Butterworth LP | 4 | 20 Hz / 44.1 kHz | 1.6 × 10¹⁵ |

The pattern is clear: **higher order and lower cutoff frequency produce
exponentially worse conditioning.** An 8th-order Butterworth at 1 kHz
has a condition number of 10¹³ — this means that `float32` (which has
~7 decimal digits of precision) can only guarantee 7 - 13 = -6 correct
digits in the frequency response. In other words, the response is
essentially meaningless if you design the filter in single precision.

This is why MATLAB uses `double` for filter design. It's not conservatism
— it's necessity for high-order designs.

**Takeaway:** Always design IIR filters in `double` (or higher). The
condition number tells you whether `float` design is acceptable: if
the condition number exceeds ~10⁷, single-precision design will produce
incorrect filter characteristics.

### Example 2: Poles Near the Unit Circle

A discrete-time IIR filter is stable if and only if all poles lie strictly
inside the unit circle (|p| < 1). The *stability margin* — the distance
from the nearest pole to the unit circle — determines how "close to
instability" the filter operates.

| Filter Design | Max Pole Radius | Stability Margin |
|---------------|----------------|-----------------|
| Butterworth LP4, fc=10 kHz | 0.671 | 0.329 |
| Butterworth LP4, fc=1 kHz | 0.947 | 0.053 |
| Butterworth LP4, fc=100 Hz | 0.995 | 0.005 |
| Butterworth LP4, fc=20 Hz | 0.999 | 0.001 |
| Butterworth BP4, fc=1 kHz, bw=50 Hz | 0.999 | 0.001 |

Low cutoff frequencies push poles toward the unit circle because the
bilinear transform compresses low frequencies into a tiny arc near z = 1.
A 20 Hz lowpass at 44.1 kHz has poles at radius 0.999 — only 0.1% from
instability.

When coefficients are quantized to a narrower type, poles shift. If the
shift exceeds the stability margin, the filter becomes unstable (oscillates
or diverges). The pole displacement from `double` to `float32`:

| Filter Design | Stability Margin | Pole Displacement (float32) | Safe? |
|---------------|-----------------|----------------------------|-------|
| Butterworth LP4, fc=10 kHz | 0.329 | 1.1 × 10⁻⁸ | Yes (huge margin) |
| Butterworth LP4, fc=1 kHz | 0.053 | 1.4 × 10⁻⁷ | Yes |
| Butterworth LP4, fc=100 Hz | 0.005 | 3.5 × 10⁻⁶ | Yes (but tighter) |
| Butterworth LP4, fc=20 Hz | 0.001 | 3.6 × 10⁻⁶ | Yes (barely) |

For `float32`, the displacement is always well within the margin. But for
narrower types (16-bit fixed-point, 8-bit posit), the displacement grows
proportionally to the quantization step size, and the 20 Hz filter with
its 0.001 margin becomes a real concern.

**Takeaway:** Before deploying with contracted coefficient types, compute
`pole_displacement()` and compare it to `stability_margin()`. If
displacement approaches 10% of the margin, the design is at risk.

### Example 3: Coefficient Sensitivity and the Cascade-of-Biquads Defense

A single high-order polynomial has terrible numerical properties. An
8th-order transfer function represented as one numerator and one
denominator polynomial is far more sensitive to coefficient errors than
the same filter factored into four 2nd-order biquad sections.

This is why every practical IIR implementation uses the cascade-of-biquads
(second-order sections) form. Each biquad has at most two poles, and a
2nd-order polynomial `z² + a₁z + a₂ = 0` has a condition number
proportional to `1/|p₁ - p₂|` — the distance between the poles. For
well-separated poles, this is small.

Our coefficient sensitivity metric measures the partial derivative
∂(pole_radius)/∂(coefficient). For a Butterworth 4th-order cascade:

```
Worst-case coefficient sensitivity: 0.57
```

This means a coefficient perturbation of ε shifts the pole radius by
at most 0.57ε. For `float32` (ε ≈ 10⁻⁷), the pole radius shifts by
~6 × 10⁻⁸ — negligible.

But if you represented the same filter as a single 8th-order polynomial
instead of four biquads, the sensitivity would be orders of magnitude
worse. The cascade-of-biquads representation is not just a convenience
— it is the primary numerical defense against coefficient quantization
errors.

**Takeaway:** Always use cascade-of-biquads form. Never represent IIR
filters as high-order polynomials, especially when deploying with
contracted types.

---

## 3. The Design → Project → Verify → Deploy Workflow

`sw::dsp` provides the `project_onto<T>` and `embed_into<T>` operators
for explicit type conversion at precision boundaries:

```cpp
// Step 1: Design in double (always)
iir::ButterworthLowPass<4> design;
design.setup(4, 44100.0, 1000.0);
const auto& cascade = design.cascade();         // Cascade<double, 2>

// Step 2: Project onto deployment type
auto cascade_f16 = project_onto<half>(cascade);  // Cascade<half, 2>

// Step 3: Verify quality
double disp   = pole_displacement(cascade, cascade_f16);
double margin = stability_margin(cascade_f16);
bool   stable = is_stable(cascade_f16);

// Step 4: Deploy (or reject and try a wider type)
if (stable && disp < 0.1 * margin) {
    // Safe to deploy with half-precision coefficients
} else {
    // Need wider coefficients for this filter
    auto cascade_f32 = project_onto<float>(cascade);
    // ... verify again
}
```

The verification step is not optional. Different filter designs have
wildly different sensitivity to coefficient quantization, and the only
way to know whether a contracted type is acceptable is to measure.

---

## 4. Three Deployment Examples with Quantified Benefits

### Deployment 1: Audio Equalizer on ARM Cortex-M (float32 coefficients)

**Application:** 4th-order parametric EQ, 10 bands, 48 kHz sample rate.

**Design:** Coefficients designed in `double`, projected onto `float32`.

| Metric | double | float32 |
|--------|--------|---------|
| Pole displacement | — | 1.4 × 10⁻⁷ |
| Stability margin preserved | 0.053 | 0.053 |
| Coefficient storage per biquad | 40 bytes | 20 bytes |
| MAC operation | 64-bit FPU (if available) | 32-bit FPU (single-cycle) |

**Benefit:** Most ARM Cortex-M4/M7 processors have single-cycle `float32`
multiply-accumulate but no `double` hardware. Using `double` coefficients
forces software emulation at ~20× the cost per MAC. For 10 bands × 2
biquads × 5 coefficients × 48,000 samples/sec = 4.8M MACs/sec, the
difference is:

| | float32 FPU | double (software) |
|-|-------------|-------------------|
| Cycles per MAC | 1 | ~20 |
| Total cycles/sec | 4.8M | 96M |
| Power (at 100 MHz, 50 mW/MHz) | 2.4 mW | 48 mW |

**Savings: 20× less energy, 20× less compute.** And the pole displacement
of 10⁻⁷ is 2000× smaller than the stability margin — the quality loss
is immeasurable.

### Deployment 2: SDR Channelizer on FPGA (fixpnt<16,14> coefficients)

**Application:** 8th-order Butterworth lowpass, fc=1 MHz, fs=50 MHz,
used as a channel selection filter in a software-defined radio.

**Design:** Coefficients designed in `double`, projected onto 16-bit
fixed-point `fixpnt<16,14>` (14 fractional bits, range [-2, 2)).

| Metric | double (design) | fixpnt<16,14> (FPGA) |
|--------|----------------|---------------------|
| Coefficient storage | 40 bytes/biquad | 10 bytes/biquad |
| Multiplier size | 64×64-bit (huge) | 16×16-bit (1 DSP slice) |
| DSP slices per biquad | N/A | 5 (a1, a2, b0, b1, b2) |
| Clock rate | N/A | 250 MHz |

With fc/fs = 1/50, the stability margin is comfortable (~0.05), and
16-bit coefficients with 14 fractional bits give a quantization step
of 2⁻¹⁴ ≈ 6 × 10⁻⁵. The pole displacement is on the order of 10⁻⁵,
well within the margin.

**FPGA resource comparison for 4 biquad stages:**

| Implementation | DSP48 slices | Block RAM | LUTs |
|----------------|-------------|-----------|------|
| 32-bit float | 20 | 0 | ~2000 (FP logic) |
| 16-bit fixed | 20 | 0 | ~200 |
| Cost ratio | 1× | — | 10× less logic |

**Savings: 10× less FPGA fabric.** The 16-bit multipliers map directly
to the DSP48 hard multiplier blocks without needing the floating-point
wrapper logic. At 250 MHz, a single DSP48 slice processes 250M
multiply-accumulates per second, far exceeding the 50 MHz sample rate.

### Deployment 3: Edge AI Sensor Preprocessing (posit<8,2> everything)

**Application:** 2nd-order lowpass anti-aliasing filter before an 8-bit
ADC in a battery-powered IoT sensor. The sensor noise floor is ~6 bits,
so the ADC's bottom 2 bits are noise.

**Design:** Single biquad designed in `double`, projected onto `posit<8,2>`.

A `posit<8,2>` has 8 bits total with 2 exponent bits, giving ~3 decimal
digits of precision near 1.0 and more dynamic range than `int8_t`.

| Metric | double | posit<8,2> |
|--------|--------|-----------|
| Coefficient storage | 40 bytes | 5 bytes |
| Multiplier width | 64 bits | 8 bits |
| Energy per MAC | ~10 pJ (65nm) | ~0.1 pJ (65nm) |
| Silicon area per MAC | ~5000 µm² | ~50 µm² |

For a 2nd-order lowpass with fc well above the sensor's noise floor, the
condition number is ~10⁶ and the stability margin is ~0.1. A `posit<8,2>`
quantization step of ~0.06 near the pole location gives a pole
displacement of ~0.03 — within 30% of the margin. This is marginal but
acceptable when the signal itself is only 6 bits of valid data.

**Savings: 100× less energy per MAC, 100× less silicon area.** For a
sensor sampling at 1 kHz, the filter uses ~10,000 MACs/sec. At 0.1 pJ
per MAC, that's 1 nW — enabling decade-long battery life from a coin cell.

The key insight: **when the sensor noise floor limits the effective
precision of the data to 6 bits, processing that data in 64-bit
arithmetic wastes 58 bits of energy on every operation.** The
`posit<8,2>` filter matches the precision to the information content.

---

## 5. Decision Framework

Use this table to select the coefficient type for deployment:

| Filter Order | Cutoff / Sample Rate | Stability Margin | Recommended Coeff Type |
|-------------|---------------------|------------------|----------------------|
| 2 | > 0.1 | > 0.05 | `posit<8,2>` or `fixpnt<8,6>` |
| 2 | > 0.01 | > 0.005 | `float16` or `fixpnt<16,14>` |
| 4 | > 0.01 | > 0.01 | `float32` or `fixpnt<16,14>` |
| 4 | < 0.01 | < 0.005 | `float32` (verify displacement) |
| 8 | > 0.01 | > 0.005 | `float32` (verify displacement) |
| 8 | < 0.01 | < 0.002 | `double` (margin too thin) |
| Any | Any | Any | Always verify with `pole_displacement()` |

The process:

1. **Design** in `double` (non-negotiable for order > 2).
2. **Project** onto candidate type: `auto target = project_onto<T>(cascade)`.
3. **Measure** `pole_displacement(original, target)`.
4. **Compare** to `stability_margin(target)`.
5. **Accept** if displacement < 10% of margin; **reject** and try wider type otherwise.

---

## 6. Tools in `sw::dsp`

| Function | What it measures |
|----------|-----------------|
| `biquad_poles(bq)` | Extract the two poles of a biquad section |
| `max_pole_radius(cascade)` | Radius of the nearest-to-instability pole |
| `stability_margin(cascade)` | Distance from nearest pole to unit circle (1 - max_radius) |
| `is_stable(cascade)` | Boolean: all poles inside unit circle? |
| `all_poles(cascade)` | Collect all poles for visualization |
| `coefficient_sensitivity(bq)` | ∂(pole_radius)/∂(a₁) and ∂(pole_radius)/∂(a₂) |
| `worst_case_sensitivity(cascade)` | Maximum sensitivity across all stages |
| `pole_displacement(original, quantized)` | How far poles moved after type projection |
| `biquad_condition_number(bq)` | Frequency response sensitivity to coefficient errors |
| `cascade_condition_number(cascade)` | Worst-case condition number across stages |
| `project_onto<T>(cascade)` | Convert coefficients to narrower type (lossy) |
| `embed_into<T>(cascade)` | Convert coefficients to wider type (lossless) |

---

## 7. Further Reading

- Oppenheim, A.V. & Schafer, R.W. *Discrete-Time Signal Processing* — Chapter 6 covers
  filter structure and finite-precision effects in detail.
- Jackson, L.B. *Digital Filters and Signal Processing* — rigorous treatment of
  quantization noise in recursive structures.
- Gustafson, J.L. *Posit Arithmetic* (2017) — the posit number system and its
  advantages for DSP applications with tapered precision.
- Widrow, B. & Kollár, I. *Quantization Noise* — comprehensive theory of
  quantization effects in signal processing systems.
