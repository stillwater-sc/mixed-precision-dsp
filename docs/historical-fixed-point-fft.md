# Fixed-Point FFT: Scaling Strategies and Historical Practice

The Cooley-Tukey FFT algorithm (1965) was originally implemented in
36-bit floating-point on an IBM 7094. Within three years, real-time
radar, sonar, and telecommunications demands forced fixed-point
implementations on 12–18 bit minicomputers. The techniques developed
to handle finite word-length effects in the FFT remain directly
relevant to mixed-precision DSP research.

## The bit-growth problem

Each radix-2 butterfly computes:

$$X = A + W \cdot B$$
$$Y = A - W \cdot B$$

If $A$ and $B$ are bounded by magnitude $M$, then $X$ and $Y$ can
reach $2M$. Over $\nu = \log_2 N$ stages, the output can grow by
a factor of $N$ relative to the input. For a 1024-point FFT, that
is 10 bits of growth — catastrophic for a 16-bit fixed-point machine.

Three scaling strategies emerged to handle this, each with different
precision trade-offs.

## Strategy 1: Divide-by-2 at each stage

After every butterfly stage, right-shift all values by 1 bit. Over
$\log_2 N$ stages, the total scaling is $1/N$, and overflow is
prevented at every stage since each butterfly's 2× growth is
immediately compensated.

**Advantages:**
- Simple: arithmetic right shift costs zero logic in hardware
- Quantization noise is distributed across stages rather than
  concentrated at the input
- Became the universal standard (TMS320 family, most DSP libraries)

**Disadvantages:**
- Loses $\log_2 N$ bits of precision unconditionally
- Quiet spectral components in the presence of loud ones may
  fall below the noise floor

**Noise analysis (Welch 1969):** For $B$-bit fixed-point with
per-stage scaling, the output noise variance is:

$$\sigma^2 \approx \frac{2}{3} \cdot 2^{-2B} \cdot \log_2 N$$

The critical insight: noise grows as $\log_2 N$, **not** $N$. This
single result — proved by P.D. Welch in 1969 — made large fixed-point
FFTs practical. A 1024-point FFT with 16-bit arithmetic achieves
approximately 80 dB SNR, degrading by only ~1 dB per doubling of $N$.

### TMS320 convention

The Texas Instruments TMS320C10 (1982) — the first commercially
successful fixed-point DSP — established the standard:

- Q15 data format (16-bit, signed fractional, range $[-1, +1)$)
- 16×16 → 32-bit multiply-accumulate
- Divide-by-2 at each butterfly stage
- Twiddle factors stored as Q15 values in on-chip ROM
- Bit-reversed addressing mode in hardware (TMS320C25, 1986)

This convention dominated fixed-point DSP for two decades and remains
the reference implementation.

## Strategy 2: Block floating-point (BFP)

Described by Oppenheim and Weinstein (1972), block floating-point
maintains a **common exponent** shared by all values in the FFT array.
Scaling is adaptive — only applied when overflow actually threatens.

**Algorithm:**
1. Before each butterfly stage, find the maximum magnitude across all
   values
2. If the maximum exceeds half-scale, right-shift all values by 1 and
   increment the block exponent
3. If the maximum is less than quarter-scale, left-shift all values
   by 1 and decrement the block exponent (recovering precision)
4. Compute the butterfly stage
5. The block exponent tracks cumulative scaling

**Advantages:**
- Adaptive dynamic range: quiet signals keep their precision
- Achieves near-floating-point SNR with fixed-point arithmetic cost
- Only one exponent word overhead per stage (not per sample)

**Disadvantages:**
- Requires O($N$) magnitude scan before each stage
- One large value forces the entire block to scale down (the
  "loud neighbor" problem)
- More complex control logic than unconditional shift

Block floating-point was the preferred technique for high-quality
military signal processing — radar and sonar systems where dynamic
range mattered more than simplicity.

## Strategy 3: Input pre-scaling by 1/N

Divide all input samples by $N$ before computing the FFT. No
per-stage logic needed.

This approach is rarely used in practice. For a 1024-point FFT with
16-bit inputs, pre-scaling loses 10 bits immediately, leaving only
5 bits of magnitude. Small signals vanish entirely. The only
advantage is implementation simplicity.

## Twiddle factor quantization

The twiddle factors $W_N^k = e^{-j2\pi k/N} = \cos(2\pi k/N) - j\sin(2\pi k/N)$
must also be stored in fixed-point.

### Storage optimization

Symmetry reduces table size dramatically:
- $\cos$ and $\sin$ are related by quarter-cycle shift → one table
- Quadrant symmetry → only $N/8$ unique values (first octant)
- An $N$-point FFT needs $N/2$ distinct twiddle factors
- With full symmetry exploitation: $N/8$ words of ROM

### Precision requirements

Twiddle factor quantization produces **deterministic error** (not
noise-like), manifesting as spectral leakage and spurious tones.
From Oppenheim and Weinstein (1972), the error is bounded by:

$$|X̂(k) - X(k)| \leq \frac{\pi}{2} \cdot \log_2 N \cdot 2^{-B_w} \cdot \sum |x(n)|$$

where $B_w$ is the number of bits in the twiddle factor table.

**Rule of thumb:** Twiddle factors need at least as many bits as the
data path, and preferably 2–4 extra bits. Using fewer bits for
twiddle factors than for data is a false economy — the deterministic
spectral artifacts are more objectionable than the distributed
roundoff noise.

### Typical table precision

| Application | Data bits | Twiddle bits | Notes |
|------------|----------|-------------|-------|
| Radar (1970s) | 12 | 12–16 | ROM in pipeline FFT stages |
| TMS320C10 | 16 | 16 (Q15) | On-chip ROM, 256 words for 256-pt |
| High-purity spectral | 16 | 20–24 | External PROM |
| Motorola DSP56000 | 24 | 24 | On-chip ROM |

## Double-length accumulators

Nearly all fixed-point FFT implementations use double-width
accumulators for the butterfly multiply-accumulate:

$$\text{acc}_{32} = A_{16} + W_{16} \cdot B_{16}$$

The 16×16 → 32-bit product preserves full intermediate precision;
rounding error is introduced only once when truncating back to 16
bits for storage. This is complementary to the scaling strategies
above — it reduces the noise introduced at each butterfly but does
not solve the overflow problem.

Double-length accumulators became a universal hardware feature in
DSP processors. The TMS320C10 had a 32-bit accumulator; the
DSP56000 had a 56-bit accumulator (24×24 → 48-bit product +
8 guard bits).

## Minimum practical word lengths

The SNR of a fixed-point FFT with $B$-bit arithmetic, $N$-point
transform, and per-stage scaling:

$$\text{SNR} \approx 6.02 \cdot (B-1) - 10 \cdot \log_{10}(\log_2 N) \quad \text{dB}$$

| FFT size | Min bits | SNR at 12-bit | SNR at 16-bit | SNR at 24-bit |
|----------|---------|--------------|--------------|--------------|
| 64 | ~12 | 58 dB | 82 dB | 130 dB |
| 256 | ~14 | 57 dB | 81 dB | 129 dB |
| 1024 | ~16 | 56 dB | 80 dB | 128 dB |
| 4096 | ~18 | 55 dB | 79 dB | 127 dB |

The gentle $\log_2 N$ degradation (Welch 1969) is why 16-bit
fixed-point FFTs were so successful — even for large $N$, the SNR
penalty is small. The minimum practical word length for a useful
FFT is approximately $\log_2 N + 6$ bits.

## DIT vs DIF noise properties

The decimation-in-time (DIT) and decimation-in-frequency (DIF)
formulations of the Cooley-Tukey algorithm have subtly different
fixed-point noise characteristics:

**DIT (Cooley-Tukey original):** Twiddle multiplication happens
**before** the butterfly addition. Rounding occurs on the product
before the sum, so errors are introduced at the input to each stage.

**DIF (Gentleman-Sande, 1966):** Butterfly addition happens **before**
twiddle multiplication. The addition is exact (no rounding for
fixed-point add), and rounding occurs on the output of each stage.

In practice, the noise difference is small — typically less than 1 dB
for 16-bit arithmetic. Both achieve the same asymptotic $\log_2 N$
noise scaling. The choice between DIT and DIF is usually driven by
addressing convenience (DIT requires bit-reversed input; DIF requires
bit-reversed output).

## Implications for mixed-precision DSP

The historical fixed-point FFT strategies map directly to the
mixed-precision research questions this library addresses:

### Per-stage scaling ↔ uniform low-precision pipeline

The divide-by-2-per-stage approach is equivalent to running the
entire FFT in a single narrow type (e.g., `fixpnt<16,15>` or
`posit<16,2>`). The library's FFT implementation can measure how
different type systems handle the accumulated quantization noise.

### Block floating-point ↔ adaptive mixed-precision

BFP's adaptive scaling is conceptually similar to using types with
tapered precision (posits) or to dynamically selecting between
precision levels. The block exponent is a crude form of what
posit regime bits provide automatically.

### Double-length accumulator ↔ three-scalar model

The universal practice of accumulating in double width and truncating
for storage is exactly the three-scalar pattern: wide `StateScalar`
for the MAC, narrow `SampleScalar` for output. The library formalizes
what hardware engineers did implicitly.

### Twiddle factor precision ↔ coefficient scalar

The twiddle factor table precision corresponds to `CoeffScalar` in the
three-scalar model. The historical rule — at least as many bits for
twiddle factors as for data — translates to: `CoeffScalar` should be
at least as wide as `SampleScalar`, and preferably wider.

### Configurations worth sweeping

The precision sweep tool should evaluate these historically-grounded
FFT configurations:

| Configuration | CoeffScalar (twiddle) | StateScalar (accumulator) | SampleScalar (data) |
|--------------|----------------------|--------------------------|-------------------|
| TMS320 classic | `fixpnt<16,15>` | `fixpnt<32,31>` | `fixpnt<16,15>` |
| DSP56000 style | `fixpnt<24,23>` | `fixpnt<48,32>` | `fixpnt<24,23>` |
| Radar 12-bit | `fixpnt<16,15>` | `fixpnt<32,31>` | `fixpnt<12,11>` |
| Posit pipeline | `posit<32,2>` | `posit<32,2>` | `posit<16,2>` |
| Posit narrow | `posit<16,2>` | `posit<24,2>` | `posit<8,2>` |
| Cross-system | `double` | `posit<32,2>` | `fixpnt<16,15>` |

## Key references

- Cooley, J.W. and Tukey, J.W. "An Algorithm for the Machine
  Calculation of Complex Fourier Series," *Mathematics of Computation*,
  19(90), pp. 297–301, April 1965.
- Gentleman, W.M. and Sande, G. "Fast Fourier Transforms — For Fun
  and Profit," *AFIPS Conference Proceedings*, 29, pp. 563–578, 1966.
- Stockham, T.G. "High-Speed Convolution and Correlation," *AFIPS
  Conference Proceedings*, 28, pp. 229–233, 1966.
- Welch, P.D. "A Fixed-Point Fast Fourier Transform Error Analysis,"
  *IEEE Trans. Audio and Electroacoustics*, AU-17(2), pp. 151–157,
  June 1969.
- Weinstein, C.J. "Roundoff Noise in Floating Point Fast Fourier
  Transform Computation," *IEEE Trans. Audio and Electroacoustics*,
  AU-17(3), pp. 209–215, September 1969.
- Gold, B. and Rader, C.M. *Digital Processing of Signals*,
  McGraw-Hill, 1969.
- Bergland, G.D. "A Guided Tour of the Fast Fourier Transform,"
  *IEEE Spectrum*, pp. 41–52, July 1969.
- Oppenheim, A.V. and Weinstein, C.J. "Effects of Finite Register
  Length in Digital Filtering and the Fast Fourier Transform,"
  *Proceedings of the IEEE*, 60(8), pp. 957–976, August 1972.
- Rabiner, L.R. and Gold, B. *Theory and Application of Digital
  Signal Processing*, Prentice-Hall, 1975.
