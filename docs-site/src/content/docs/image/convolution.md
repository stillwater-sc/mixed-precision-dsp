---
title: 2D Convolution
description: Spatial convolution, separable kernels, and mixed-precision filtering for images
---

Two-dimensional convolution is the foundation of spatial filtering in
image processing. It applies a small kernel (filter mask) to every pixel
in an image, producing a weighted sum of the local neighborhood.

## 2D convolution equation

The discrete 2D convolution of an image $f$ with a kernel $g$ of size
$(2a+1) \times (2b+1)$ is:

$$
(f * g)[i, j] = \sum_{m=-a}^{a} \sum_{n=-b}^{b} f[m, n] \cdot g[i - m, j - n]
$$

For each output pixel, the kernel slides over the image and computes
the weighted sum of all covered input pixels.

### Boundary handling

At image borders the kernel extends beyond the valid pixel region.
Common strategies include:

- **Zero padding** -- pixels outside the image are treated as zero
- **Replicate** -- the nearest edge pixel is repeated
- **Reflect** -- the image is mirrored at the boundary

## Separable kernels

A 2D kernel $g[m, n]$ is **separable** if it can be expressed as the outer
product of two 1D vectors:

$$
g[m, n] = g_{\text{row}}[m] \cdot g_{\text{col}}[n]
$$

Separability reduces the cost of an $M \times M$ convolution from
$O(M^2)$ to $O(2M)$ multiplications per pixel, since the 2D operation
decomposes into two successive 1D passes.

### Gaussian kernel

The 2D Gaussian kernel is separable. A Gaussian with standard deviation
$\sigma$ has the continuous form:

$$
G(x, y) = \frac{1}{2\pi\sigma^2} e^{-(x^2 + y^2)/(2\sigma^2)}
$$

The library constructs a discrete, truncated, and normalized version
of the specified size.

### Sobel kernel

The $3 \times 3$ Sobel operators for horizontal and vertical gradients
are also separable:

$$
S_x =
\begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}
= \begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix}
  \begin{bmatrix} -1 & 0 & 1 \end{bmatrix}
$$

## Library API

### General 2D convolution

```cpp
#include <sw/dsp/image/convolution.hpp>

using namespace sw::dsp;

Image<double, 1> img(480, 640);
// ... load or generate image ...

// Define a 3x3 sharpening kernel
mtl::mat::dense2D<double> kernel(3, 3);
kernel = {{ 0, -1,  0},
           {-1,  5, -1},
           { 0, -1,  0}};

auto sharpened = image::convolve2d(img, kernel);
```

### Gaussian blur

The `gaussian_blur()` function constructs the Gaussian kernel internally
and exploits separability for efficient computation:

```cpp
// Gaussian blur with kernel size 5 and sigma = 1.4
auto blurred = image::gaussian_blur(img, 5, 1.4);
```

### Mixed-precision convolution

The kernel type `K` can differ from the image pixel type `T`. This lets
you store images in a compact representation while keeping kernel
coefficients at full precision:

```cpp
using Pixel  = sw::universal::posit<8, 0>;
using Kernel = double;

Image<Pixel, 1> img(480, 640);
// ...

mtl::mat::dense2D<Kernel> kernel(3, 3);
// ... fill kernel ...

// Accumulation happens in Kernel precision
auto result = image::convolve2d<Pixel, Kernel>(img, kernel);
```

The accumulator uses the wider of the two types, ensuring that the sum
of products does not suffer from narrow-type overflow or rounding.

## Performance considerations

### Separable decomposition

For kernels that are separable, the library automatically detects this
and applies two 1D passes instead of the full 2D convolution. The speedup
grows with kernel size:

| Kernel size | 2D multiplies/pixel | Separable multiplies/pixel |
|-------------|--------------------|-----------------------------|
| $3 \times 3$   | 9                  | 6                           |
| $5 \times 5$   | 25                 | 10                          |
| $7 \times 7$   | 49                 | 14                          |
| $11 \times 11$ | 121                | 22                          |

### Memory layout

The planar image layout (one `dense2D<T>` per channel) ensures that
convolution accesses contiguous memory within each channel, maximizing
cache utilization. For multi-channel images, each channel is convolved
independently.

## Precision impact

In 2D convolution the accumulator sums $M^2$ products. For an $11 \times 11$
kernel, that is 121 multiply-accumulate operations per output pixel. With
an 8-bit integer pixel type, intermediate sums can easily exceed the
representable range. Using a wider accumulator type -- or a tapered number
system like posits -- prevents silent overflow and preserves image quality.
