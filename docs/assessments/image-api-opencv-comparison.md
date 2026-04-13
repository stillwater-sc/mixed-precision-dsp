# Image Processing API: OpenCV Comparison and Design Decisions

**Date:** 2026-04-12
**Context:** Design assessment for `sw::dsp` image processing module (Issue #8)

---

## 1. OpenCV's Image Processing API

### 1.1 Image Container: `cv::Mat`

OpenCV's core container is `cv::Mat` — a reference-counted N-dimensional dense array.

**Key characteristics:**
- **Interleaved memory layout** — channels are packed per pixel: `B0 G0 R0 B1 G1 R1 ...`
- **BGR channel order** — historical artifact from early camera driver conventions
- **Runtime type encoding** — `CV_MAKETYPE(depth, channels)` produces an integer type code (e.g., `CV_8UC3` = 3-channel 8-bit unsigned). No compile-time type safety on depth or channel count.
- **Row-major with optional padding** — `step[0]` may exceed `cols * elemSize` for alignment
- **Up to 512 channels** supported (`CV_CN_MAX`)

**Channel decomposition:**
```cpp
// Split interleaved BGR into 3 separate single-channel Mats (full copy)
std::vector<cv::Mat> planes;
cv::split(bgr_image, planes);   // planes[0]=B, planes[1]=G, planes[2]=R

// Merge back (full copy)
cv::Mat merged;
cv::merge(planes, merged);
```

`split()` and `merge()` perform full memory copies. This is the canonical path whenever a single-channel algorithm (e.g., Canny) must be applied to a color image.

### 1.2 2D Convolution

```cpp
void cv::filter2D(InputArray src, OutputArray dst, int ddepth,
                  InputArray kernel, Point anchor = Point(-1,-1),
                  double delta = 0, int borderType = BORDER_DEFAULT);
```

- The same kernel is applied to every channel independently (no cross-channel coupling).
- `ddepth` controls output precision at runtime (e.g., `CV_16S` for derivative filters on `CV_8U` input).
- Internally switches to DFT-based convolution for large kernels.

**Separable variant:**
```cpp
void cv::sepFilter2D(InputArray src, OutputArray dst, int ddepth,
                     InputArray kernelX, InputArray kernelY,
                     Point anchor = Point(-1,-1), double delta = 0,
                     int borderType = BORDER_DEFAULT);
```

Applies row filter then column filter — O(k) per pixel instead of O(k²).

### 1.3 Morphological Operations

```cpp
void cv::erode(InputArray src, OutputArray dst, InputArray kernel, ...);
void cv::dilate(InputArray src, OutputArray dst, InputArray kernel, ...);
void cv::morphologyEx(InputArray src, OutputArray dst, int op, InputArray kernel, ...);
```

- Structuring elements created via `cv::getStructuringElement(shape, ksize)` with `MORPH_RECT`, `MORPH_CROSS`, `MORPH_ELLIPSE`.
- `morphologyEx` provides compound operations: `MORPH_OPEN`, `MORPH_CLOSE`, `MORPH_GRADIENT`, `MORPH_TOPHAT`, `MORPH_BLACKHAT`, `MORPH_HITMISS`.
- Multi-channel: operates per channel independently. `MORPH_HITMISS` requires single-channel.

### 1.4 Edge Detection

```cpp
void cv::Sobel(InputArray src, OutputArray dst, int ddepth,
               int dx, int dy, int ksize = 3, ...);

void cv::Canny(InputArray image, OutputArray edges,
               double threshold1, double threshold2,
               int apertureSize = 3, bool L2gradient = false);
```

- Sobel works on multi-channel images (per channel independently).
- **Canny requires single-channel input** — users must convert to grayscale or `split()` first.
- A second Canny overload accepts pre-computed gradient images `(dx, dy)`.

### 1.5 Border Handling

| Mode | Behavior | Default for |
|------|----------|-------------|
| `BORDER_CONSTANT` | Pad with fixed value | Morphology (max for erode, 0 for dilate) |
| `BORDER_REPLICATE` | Clamp to edge pixel | — |
| `BORDER_REFLECT` | Mirror including edge | — |
| `BORDER_REFLECT_101` | Mirror excluding edge | Convolution (`BORDER_DEFAULT`) |
| `BORDER_WRAP` | Periodic tiling | — |
| `BORDER_TRANSPARENT` | Leave dst unchanged at borders | — |

### 1.6 Design Philosophy Summary

1. **Interleaved, not planar** — channels packed per pixel; decomposition is explicit via `split()`
2. **Per-channel independence** — convolution and morphology apply the same kernel to each channel
3. **Output depth is explicit** — `ddepth` parameter for precision control
4. **In-place allowed** — `src` and `dst` can alias; temp allocated internally
5. **Separability is first-class** — `sepFilter2D` and Sobel's internal use
6. **Runtime type system** — integer-encoded depth+channels, no compile-time safety
7. **Standardized border handling** — single enum across all spatial filters

---

## 2. Our Planned API (`sw::dsp`)

### 2.1 Image Container: Planar `Image<T, Channels>`

We use `mtl::mat::dense2D<T>` as the single-channel image primitive, wrapped in a thin multi-channel container:

```cpp
template <DspField T, std::size_t Channels = 1>
struct Image {
    std::array<mtl::mat::dense2D<T>, Channels> planes;

    std::size_t rows() const { return planes[0].num_rows(); }
    std::size_t cols() const { return planes[0].num_cols(); }

    dense2D<T>&       operator[](std::size_t c)       { return planes[c]; }
    const dense2D<T>& operator[](std::size_t c) const { return planes[c]; }
};

// Convenience aliases
template <typename T> using GrayImage = Image<T, 1>;
template <typename T> using RGBImage  = Image<T, 3>;
template <typename T> using RGBAImage = Image<T, 4>;
```

**Rationale for planar layout:**
- Consistent with MTL5 — `dense2D<T>` is our container; no new 3D tensor needed
- Consistent with our audio pattern — stereo WAV already uses separate `dense_vector<T>` per channel
- All single-channel algorithms work directly — `convolve2d(img[0], kernel)` with no adaptation
- No split/merge copy cost — channels are already separate planes
- Cache-friendly for DSP — filtering processes one channel at a time; each plane is contiguous
- Mixed precision per channel — `Image<posit<16,1>, 3>` for color, `Image<uint8_t, 1>` for alpha

### 2.2 Core Functions Operate on Single Planes

All core algorithms accept `dense2D<T>` (single channel):

```cpp
// 2D convolution
template <DspField T, DspField K>
dense2D<T> convolve2d(const dense2D<T>& image, const dense2D<K>& kernel,
                      BorderMode border = BorderMode::reflect_101);

// Separable filter
template <DspField T, DspField K>
dense2D<T> separable_filter(const dense2D<T>& image,
                            const dense_vector<K>& row_kernel,
                            const dense_vector<K>& col_kernel,
                            BorderMode border = BorderMode::reflect_101);

// Morphology
template <DspOrderedField T>
dense2D<T> erode(const dense2D<T>& image, const dense2D<bool>& element);

template <DspOrderedField T>
dense2D<T> dilate(const dense2D<T>& image, const dense2D<bool>& element);

// Edge detection
template <DspField T, DspField K = double>
dense2D<T> sobel_x(const dense2D<T>& image, int ksize = 3);

template <DspField T>
    requires ConvertibleToDouble<T>
dense2D<T> canny(const dense2D<T>& image,
                 double low_threshold, double high_threshold,
                 double sigma = 1.0);
```

A convenience helper applies any function across all channels:

```cpp
template <DspField T, std::size_t C, typename Func>
Image<T, C> apply_per_channel(const Image<T, C>& img, Func&& func) {
    Image<T, C> result;
    for (std::size_t i = 0; i < C; ++i)
        result[i] = func(img[i]);
    return result;
}
```

### 2.3 Border Handling

```cpp
enum class BorderMode {
    zero,          // pad with T{0}
    constant,      // pad with user-specified value
    replicate,     // clamp to edge
    reflect,       // mirror including edge pixel
    reflect_101,   // mirror excluding edge pixel (default)
    wrap           // periodic
};
```

### 2.4 RGBA Channel Handling

Alpha is fundamentally different from color channels — it is a mask/weight, not a signal. Planar layout makes this distinction natural:

- **Filter RGB, preserve alpha:** process only `img[0..2]`, leave `img[3]` untouched
- **Filter alpha separately:** apply different parameters (e.g., binary morphology on alpha mask)
- **Alpha blending:** per-pixel operation across planes, straightforward with planar access
- **Mixed types:** color channels as `posit<16,1>`, alpha as `uint8_t` — possible with separate `Image` instances

---

## 3. Comparison Matrix

| Aspect | OpenCV | `sw::dsp` | Advantage |
|--------|--------|-----------|-----------|
| **Type safety** | Runtime integers (`CV_8UC3`) | Compile-time templates (`Image<T, C>`) | `sw::dsp` — type errors caught at compile time |
| **Memory layout** | Interleaved (pixel-packed) | Planar (channel-separated) | `sw::dsp` — no split/merge cost for per-channel DSP |
| **Mixed precision** | Runtime `ddepth` parameter | Template parameters `T`, `K` | `sw::dsp` — kernel, image, and output types independent at compile time |
| **Channel handling** | Implicit (same kernel per channel) | Explicit (operate on planes) | Trade-off: OpenCV is more concise for simple cases; ours is more flexible |
| **Border modes** | 7 modes including `TRANSPARENT` | 6 modes (no transparent/isolated) | OpenCV — more modes, but ours covers all DSP-relevant cases |
| **Separability** | `sepFilter2D` | `separable_filter` | Equivalent |
| **Canny input** | Single-channel only | Single-channel only | Equivalent — but our planar layout means no `split()` needed |
| **In-place operation** | Supported (internal temp) | Not planned (functional style) | OpenCV — saves allocation for large images |
| **Arithmetic types** | `float`, `double`, `uint8_t`, etc. | Any `DspField` including Universal types | `sw::dsp` — posit, cfloat, fixpnt, half, etc. |
| **BGR convention** | Yes (historical) | No — RGB by convention | `sw::dsp` — no legacy baggage |
| **Dependencies** | Large runtime library | Header-only, MTL5 only | `sw::dsp` — embeddable, no link-time cost |

---

## 4. What We Should Add (Improvements Over Current Issue #8)

1. **`Image<T, Channels>` thin wrapper** in `image.hpp` — not a heavy class, just `std::array<dense2D<T>, C>` with accessors
2. **`BorderMode::replicate` and `BorderMode::reflect_101`** — OpenCV's default and most useful mode; our original issue only had zero-pad, reflect, wrap
3. **`apply_per_channel()` helper** — convenience for bulk per-plane operations
4. **Color space utilities** — `rgb_to_gray<T>()` (weighted sum: 0.299R + 0.587G + 0.114B)
5. **Structuring element factory** — `make_rect_element(rows, cols)`, `make_cross_element(size)`, `make_ellipse_element(size)`
6. **Compound morphology** — `open()`, `close()`, `gradient()`, `tophat()`, `blackhat()` built from `erode()`/`dilate()`

### What We Intentionally Omit

- **In-place operation** — functional style (return new matrix) is safer and easier to reason about; the allocation cost is acceptable for our use cases
- **DFT-accelerated large-kernel convolution** — can be added later; spatial-domain convolution covers typical kernel sizes (3×3 to 15×15)
- **`BORDER_TRANSPARENT`** — niche; not needed for standard DSP workflows
- **Runtime type dispatch** — we use templates; no need for OpenCV's `switch(depth)` pattern

---

## 5. Conclusion

Our planar `Image<T, Channels>` design is a better fit than OpenCV's interleaved `cv::Mat` for a DSP-focused library because:

1. **Per-channel processing is the dominant pattern** in image filtering — planar layout avoids the split/merge overhead that OpenCV pays
2. **Mixed-precision parameterization** is our core value proposition — template-based typing is superior to runtime `ddepth` for this purpose
3. **Consistency with MTL5 and our audio API** — the same container philosophy (`dense2D`, `dense_vector`) across all domains
4. **Alpha channel is naturally decoupled** — treated as a separate plane rather than forced into the same interleaved structure

The main trade-off is that per-pixel operations (color space conversion, alpha blending) require iterating across planes rather than reading contiguous pixel structs. This is acceptable because these operations are memory-bound and the stride cost is negligible compared to the flexibility gained.
