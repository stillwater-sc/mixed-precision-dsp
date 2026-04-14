# Generalizing project_onto / embed_into: From DSP to Linear Algebra

**Date:** 2026-04-14
**Context:** Analysis of whether `project_onto<T>` / `embed_into<T>` should be generalized beyond `sw::dsp` into MTL5 as fundamental linear algebra operations.

---

## 1. The Question

`project_onto<T>` and `embed_into<T>` were introduced in `sw::dsp` for
mixed-precision filter coefficient management. But element-wise type
conversion on vectors and matrices is not a DSP concern — it's a linear
algebra concern. Every mixed-precision application (ML training/inference,
sensor fusion, constraint solving, scientific computing) needs to convert
containers between arithmetic types.

Should these operations be in MTL5?

**Yes.** The element-wise overloads for `dense_vector<T>`, `dense2D<T>`,
and sparse matrices belong in MTL5. The domain-specific overloads
(`BiquadCoefficients`, `Cascade`) stay in `sw::dsp`.

---

## 2. Proposed Layering

```
MTL5 (linear algebra layer):
  project_onto<Target>(dense_vector<Source>)   → dense_vector<Target>
  project_onto<Target>(dense2D<Source>)         → dense2D<Target>
  project_onto<Target>(compressed2D<Source>)    → compressed2D<Target>
  embed_into<Target>(dense_vector<Source>)      → dense_vector<Target>
  embed_into<Target>(dense2D<Source>)           → dense2D<Target>
  embed_into<Target>(compressed2D<Source>)      → compressed2D<Target>

  Concepts:
    ProjectableOnto<Target, Source>  (digits(Target) <= digits(Source))
    EmbeddableInto<Target, Source>   (digits(Target) >= digits(Source))

sw::dsp (domain layer):
  project_onto<Target>(BiquadCoefficients<Source>)  → BiquadCoefficients<Target>
  project_onto<Target>(Cascade<Source, N>)           → Cascade<Target, N>
  embed_into<Target>(BiquadCoefficients<Source>)     → BiquadCoefficients<Target>
  embed_into<Target>(Cascade<Source, N>)             → Cascade<Target, N>
  // These call MTL5's versions internally for any vector/matrix members
```

When MTL5 provides the container overloads, `sw::dsp` drops its own
vector/matrix implementations and delegates to MTL5.

---

## 3. Block Formats

MTL5 supports block-structured matrices: `dense2D<dense2D<T>>` where
each element is itself a matrix. What happens when you project a block
matrix?

### Element-wise projection (flat)

```cpp
// Each scalar in each block gets cast
dense2D<dense2D<double>> block_matrix;
auto projected = project_onto<float>(block_matrix);
// Result: dense2D<dense2D<float>>
```

This requires recursive projection: the outer `project_onto` iterates
over blocks and calls `project_onto` on each inner `dense2D`. The
implementation:

```cpp
template <DspField Target, DspField Source, typename Params>
dense2D<Target, Params> project_onto(const dense2D<Source, Params>& src) {
    // If Source is itself a matrix type, recurse
    // Otherwise, element-wise static_cast
}
```

### Are projected block systems solvable?

Yes. LU, Cholesky, QR, and iterative solvers all work on projected
matrices. The solution quality degrades proportionally to:

```
solution_error ≈ condition_number(A) × quantization_step(T)
```

For a well-conditioned system (κ ≈ 10²) projected to `float32`
(ε ≈ 10⁻⁷), the solution error is ~10⁻⁵ — 5 correct digits.

For an ill-conditioned system (κ ≈ 10¹⁰) projected to `float32`,
the solution error is ~10³ — no correct digits. The projection is
technically valid but the answer is useless.

**The analysis tools (`condition_number`, `pole_displacement`) are the
gatekeepers.** Always measure before deploying with a contracted type.

### Are projected block systems serializable?

Yes, and this is a strong use case. Projecting a model matrix for
serialization is principled lossy compression:

```cpp
// Sender: project to half precision (50% bandwidth savings)
auto compact = project_onto<half>(model_matrix);
serialize(compact, wire);

// Receiver: embed back for computation
auto full = embed_into<double>(deserialize<half>(wire));

// Quantify the compression loss
double error = frobenius_norm(original - full) / frobenius_norm(original);
```

This is exactly what ML model quantization does (FP32 weights → INT8 for
inference), but generalized to any type pair with compile-time directional
enforcement.

---

## 4. Resampling: A Different Operation

Projection/embedding converts between *types* at the same sampling grid.
Resampling converts between *sampling grids* at the same type. They are
orthogonal operations that compose.

| Operation | Precision | Grid | Analogy |
|-----------|-----------|------|---------|
| `project_onto<T>` | Contracts | Same | Projection onto subspace |
| `embed_into<T>` | Expands | Same | Embedding into superspace |
| `downsample` / `restrict` | Same | Contracts | Restriction operator |
| `upsample` / `prolongate` | Same | Expands | Prolongation operator |

### The multigrid connection

In multigrid methods, the restriction operator `R` maps a fine-grid
vector to a coarse grid (fewer elements), and the prolongation operator
`P` maps back. Combined with projection/embedding, you get a complete
algebra for mixed-precision, multi-resolution computation:

```
Fine grid, high precision   ──project──►  Fine grid, low precision
        │                                         │
     restrict                                  restrict
        │                                         │
        ▼                                         ▼
Coarse grid, high precision ──project──►  Coarse grid, low precision
```

The four corners of this diagram are four different representations of
the same mathematical object, each optimized for a different purpose:

- **Fine/high:** Training, design, analysis
- **Fine/low:** Inference, streaming, deployment
- **Coarse/high:** Multigrid coarse correction, model distillation
- **Coarse/low:** Maximum compression, edge deployment

### DSP already has the resampling primitives

- `sw::dsp` has decimation and interpolation (polyphase, Issue #33)
- Image processing has Gaussian blur + downsample (image pyramid)
- The `project_onto` + `downsample` composition can be a convenience
  function but should be built from independent primitives

### ML/AI: both operations in one step

In ML model compression, the combined operation is common:

```
Training:    1024×1024 weight matrix in float32
Deployment:  512×512  weight matrix in int8
```

This is `project_onto<int8>(restrict(weights))` — type contraction
plus spatial contraction. The two operations compose cleanly:

```cpp
// Spatial contraction (average pooling or SVD truncation)
auto small = restrict(weights, 512, 512);

// Type contraction
auto quantized = project_onto<int8_t>(small);

// Quantify total loss
double spatial_loss = frobenius_norm(weights - prolongate(small)) / norm;
double type_loss = frobenius_norm(small - embed_into<float>(quantized)) / norm;
double total_loss = frobenius_norm(weights - embed_into<float>(prolongate(quantized))) / norm;
```

---

## 5. Summary of Operations

```
                    Type axis
                    ◄───────────────────►
                    wider          narrower

Grid axis    ▲     embed_into     project_onto
             │     (lossless)     (lossy)
           finer
                   prolongate     restrict + project
           coarser (interpolate)  (decimate)
             │
             ▼
```

These four operations form a complete basis for mixed-precision,
multi-resolution data management across domains:

- **DSP:** design in double → project coefficients to fixed-point →
  decimate/interpolate signals for sample rate conversion
- **ML/AI:** train in float32 → project weights to int8 →
  restrict layers for model pruning
- **Sensor fusion:** embed sensor readings into double for fusion →
  project fused state to narrower type for telemetry
- **Scientific computing:** multigrid with mixed precision at each
  grid level — coarse grids use narrower types because the solution
  is smooth and doesn't need fine precision

---

## 6. Recommendations

1. **MTL5 issue:** Add `project_onto`/`embed_into` for `dense_vector`,
   `dense2D`, and `compressed2D` with `ProjectableOnto`/`EmbeddableInto`
   concepts. Reference the `sw::dsp` implementation as the prototype.

2. **MTL5 issue (follow-up):** Block-recursive projection for nested
   matrix types (`dense2D<dense2D<T>>`).

3. **Keep resampling separate.** Projection and resampling are orthogonal
   operations that compose. Don't conflate them into a single function.

4. **The serialization use case** should be documented as a first-class
   workflow: project for wire format, embed on receiver. This is
   principled lossy compression with quantifiable error bounds.

5. **sw::dsp defers to MTL5** once MTL5 provides the container overloads.
   Domain-specific overloads (BiquadCoefficients, Cascade) remain in
   sw::dsp and delegate to MTL5 for any internal vector/matrix members.
