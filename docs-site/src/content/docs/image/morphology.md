---
title: Morphological Operations
description: Dilation, erosion, opening, and closing for binary and grayscale image processing
---

Mathematical morphology is a framework for analyzing and processing
images based on set-theoretic and lattice operations. It is widely used
for noise removal, shape analysis, and feature extraction in both binary
and grayscale images.

## Structuring elements

A **structuring element** (SE) is a small shape -- typically a square,
cross, or disk -- that defines the neighborhood used by each morphological
operation. The SE is represented as a binary matrix where 1-valued
entries indicate included positions:

```cpp
#include <sw/dsp/image/morphology.hpp>

using namespace sw::dsp;

// 3x3 square structuring element (all ones)
auto se = image::structuring_element::square(3);

// 5x5 cross-shaped structuring element
auto se_cross = image::structuring_element::cross(5);

// 7x7 disk structuring element
auto se_disk = image::structuring_element::disk(7);
```

The size of the structuring element controls the scale of the features
affected by the operation.

## Dilation

Dilation expands bright regions (or foreground in binary images). For a
grayscale image $f$ and structuring element $B$, dilation is defined as
the local maximum:

$$
(f \oplus B)[i, j] = \max_{(m, n) \in B} f[i + m, j + n]
$$

Each output pixel takes the maximum value within the SE-shaped
neighborhood. This fills small holes, connects nearby bright regions,
and thickens features.

```cpp
Image<double, 1> img(480, 640);
// ... load image ...

auto dilated = image::dilate(img, se);
```

## Erosion

Erosion shrinks bright regions. It is the dual of dilation, using the
local minimum:

$$
(f \ominus B)[i, j] = \min_{(m, n) \in B} f[i + m, j + n]
$$

Each output pixel takes the minimum value within the neighborhood.
Erosion removes small bright spots, separates touching objects, and
thins features.

```cpp
auto eroded = image::erode(img, se);
```

## Compound operations

### Opening

**Opening** is erosion followed by dilation with the same structuring
element:

$$
f \circ B = (f \ominus B) \oplus B
$$

Opening removes small bright features (noise, thin protrusions) while
preserving the overall shape and size of larger objects. It is
**idempotent**: applying it twice gives the same result as applying it
once.

```cpp
auto opened = image::morphological_open(img, se);
```

### Closing

**Closing** is dilation followed by erosion:

$$
f \bullet B = (f \oplus B) \ominus B
$$

Closing fills small dark gaps and holes while preserving the size of
larger dark regions. Like opening, it is idempotent.

```cpp
auto closed = image::morphological_close(img, se);
```

### Relationship between opening and closing

Opening and closing are duals: opening removes bright features smaller
than the SE, while closing removes dark features smaller than the SE.
Applied together, they form a powerful noise-removal pipeline:

```cpp
// Remove salt-and-pepper noise: open then close
auto cleaned = image::morphological_close(
    image::morphological_open(img, se), se);
```

## Gradient and other derived operations

The **morphological gradient** highlights edges by computing the
difference between dilation and erosion:

$$
\text{grad}(f) = (f \oplus B) - (f \ominus B)
$$

```cpp
auto grad = image::dilate(img, se);
// subtract erosion from dilation to get the morphological gradient
```

## Applications

### Noise removal

Opening removes isolated bright pixels (salt noise), and closing removes
isolated dark pixels (pepper noise). The combination effectively
suppresses impulse noise without the blurring introduced by linear
filters like the Gaussian.

### Feature extraction

By choosing a structuring element matched to the shape of interest,
morphological operations can extract specific features. For example, a
horizontal line SE detects and preserves horizontal structures while
removing vertical ones.

### Pre-processing for segmentation

Morphological operations clean up binary masks produced by thresholding
or edge detection. Opening separates touching objects, and closing fills
gaps in object boundaries, improving subsequent contour extraction or
connected-component labeling.

## Mixed-precision morphology

Dilation and erosion use comparison operations ($\max$ and $\min$) rather
than multiply-accumulate. This means they are less sensitive to
accumulator precision than convolution-based filters. However, the pixel
type still matters: a narrow type with few representable values produces
a coarser intensity lattice, which can merge distinct intensity levels
during the $\max$/$\min$ operations.

```cpp
using Pixel = sw::universal::posit<8, 0>;

Image<Pixel, 1> img(480, 640);
auto dilated = image::dilate(img, se);
auto eroded  = image::erode(img, se);
```

For most applications, morphological operations are well-suited to
narrow pixel types since the comparison-based logic does not amplify
quantization error the way differentiation does.

## API summary

| Function                          | Description                          |
|-----------------------------------|--------------------------------------|
| `image::dilate(img, se)`          | Local maximum over structuring element |
| `image::erode(img, se)`           | Local minimum over structuring element |
| `image::morphological_open(img, se)`  | Erosion then dilation            |
| `image::morphological_close(img, se)` | Dilation then erosion            |
