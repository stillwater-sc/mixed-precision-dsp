# Multi-Stage Decimation Chain

**Date:** 2026-04-22
**Issue:** #90 (part of Epic #84, High-Rate Data Acquisition Pipeline)
**Author:** design captured during resolve-issue session

---

## 1. Goal

Provide a composable multi-stage decimation pipeline at
`include/sw/dsp/acquisition/decimation_chain.hpp`.

High-rate acquisition systems (software radio, radar, instrument front
ends) achieve large decimation ratios by cascading heterogeneous
filters: a CIC at the input rate for efficient bulk rate reduction,
one or more half-band filters near the final 2x, and a shaping
polyphase FIR at the output rate. Each stage has different precision
requirements, so the container must allow per-stage type parameters.

## 2. Design

### 2.1 Class signature

```cpp
template <DspField Sample, class... Stages>
class DecimationChain {
    std::tuple<Stages...> stages_;
    Sample input_rate_;
public:
    DecimationChain(Sample input_rate, Stages... stages);

    std::pair<bool, Sample> process(Sample in);
    mtl::vec::dense_vector<Sample> process_block(...);

    Sample       input_rate()       const;
    Sample       output_rate()      const;
    std::size_t  total_decimation() const;   // product of per-stage ratios

    template <std::size_t I> auto&       stage();
    template <std::size_t I> const auto& stage() const;

    void reset();
};
```

### 2.2 Why a variadic tuple rather than runtime type erasure

Keeps the library header-only and zero-overhead. Each stage keeps its
own `CoeffScalar / StateScalar / SampleScalar` template parameters; the
chain only constrains the pass-through `Sample` type at the stage
boundaries.

A runtime `std::vector<std::unique_ptr<DecimStageBase<T>>>` alternative
would:
- Require heap allocation (forbidden on embedded / RT targets).
- Lose the per-stage precision via virtual dispatch erasing types.
- Force a common base class that none of the existing decimators
  implement, so every stage would need an adapter.

### 2.3 Streaming semantics

`process(in)` threads a sample through the stages. Stage 0 consumes
every input; stage `i+1` consumes the output of stage `i` only when
stage `i` emits. The chain returns `{true, y}` only when the final
stage emits, so outputs come out at rate `input_rate / prod(ratios)`.

Implementation uses `if constexpr` recursion over the tuple index:

```cpp
template <std::size_t I>
std::pair<bool, Sample> process_impl(Sample in) {
    if constexpr (I == sizeof...(Stages)) return {true, in};
    else {
        auto& stage = std::get<I>(stages_);
        auto [ready, out] = detail::step_decimator(stage, in);
        if (!ready) return {false, Sample{}};
        return process_impl<I+1>(out);
    }
}
```

`detail::step_decimator` is reused unchanged from `ddc.hpp`; it already
dispatches across `process`, `process_decimate`, and `push`/`output`.

### 2.4 Decimation ratio query

`total_decimation()` needs each stage's rate reduction. A helper:

```cpp
template <class T>
constexpr std::size_t decimation_ratio_of(const T& t) {
    if constexpr (requires { t.decimation_ratio(); })
        return static_cast<std::size_t>(t.decimation_ratio());
    else if constexpr (requires { t.factor(); })
        return t.factor();
    else if constexpr (requires (T x, typename T::sample_scalar s) { x.process_decimate(s); })
        return 2;
    else
        static_assert(always_false<T>, "decimation_ratio_of: unknown stage type");
}
```

Hard-coded `2` for half-band: the library's `HalfBandFilter` is
structurally fixed at 2:1 when used via `process_decimate`. Non-2:1
half-bands are a theoretical but unimplemented generalization and
would need to extend this dispatch.

### 2.5 CIC droop compensation

CIC's frequency response is `|H(f)| = |sinc(pi f D R)|^M /
|sinc(pi f D)|^M`, which droops across the passband. A compensation
FIR run after the CIC at the decimated rate flattens the response.

Design via frequency sampling:

```cpp
template <DspField T>
mtl::vec::dense_vector<T> design_cic_compensator(
    std::size_t num_taps,   // odd, linear phase
    int cic_stages,         // M
    int cic_ratio,          // R
    T   passband);          // normalized passband edge (0, 0.5)
```

Algorithm:
1. Sample the desired magnitude `1 / |H_cic(f)|` at `N = num_taps`
   points across `[0, passband]` and a smooth rolloff `[passband, 0.5]`.
2. Use the symmetric frequency-sampling formula to synthesize a
   linear-phase FIR of length `num_taps`.
3. Window with Hamming to suppress Gibbs ripple.

Not optimal (an iterative Remez design would do better), but easy,
analytic, and sufficient to demonstrate passband flattening.

### 2.6 Cross-precision at stage boundaries

**Scope decision:** the chain's `Sample` template parameter is
uniform across stage boundaries. Each stage may internally use
different coefficient/state precision, but the sample stream type
does not change. Cross-precision casting at boundaries is a future
extension (would need an explicit `cast_adapter<From, To>` wrapper
stage).

## 3. Tests

Located at `tests/test_decimation_chain.cpp`:

1. **Acceptance 3-stage chain:** CIC-64 → HB-2 → FIR-4 with a known
   sinusoidal input; check total decimation, output sample count, and
   that a passband tone survives while an out-of-band tone is
   suppressed.
2. **Per-stage precision mixing:** same chain topology with mixed
   `double` / `posit<32,2>` stage types.
3. **Droop compensation flatness:** measure CIC-only passband roll
   vs. CIC+compensator passband flatness across the passband.
4. **Streaming vs block equivalence:** per-sample `process()` and
   `process_block()` agree bit-exact.
5. **Reset reproducibility:** re-feeding the same input after
   `reset()` reproduces the output.
6. **Posit mixed-precision end-to-end:** same chain entirely in
   `posit<32,2>`.

## 4. Documentation

`docs-site/src/content/docs/acquisition/decimation-chain.md`:
- History: Hogenauer 1981 CIC, Harris multirate cascades, the
  DDC ASIC decimation chains (HSP50016, AD6620, GC4016).
- Why: large `R` is too expensive in one FIR stage; cascade of
  short filters is optimal.
- What: signal flow, per-stage rates, bit-growth budget.
- How: code snippets for the acceptance chain, choice of
  compensator, retuning.

## 5. Out of scope (deferred)

- Precision transitions at stage boundaries (adapter stage).
- Runtime-reconfigurable stage order.
- Interpolation chain (DUC direction) — separate issue.
- Multirate SRC (rational L/M) — covered by `SRC` in `src.hpp`.
- Parallel channelizer (one-NCO-per-channel fan-out) — separate issue.

## 6. Files touched

| File | Change |
|------|--------|
| `include/sw/dsp/acquisition/decimation_chain.hpp` | NEW (~250 lines) |
| `include/sw/dsp/acquisition/acquisition.hpp` | +1 include |
| `tests/test_decimation_chain.cpp` | NEW |
| `tests/CMakeLists.txt` | +1 `dsp_add_test` entry |
| `docs-site/src/content/docs/acquisition/decimation-chain.md` | NEW |
