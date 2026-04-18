---
title: Edge Detection
description: Sobel, Laplacian, and Canny edge detection with mixed-precision arithmetic
---

Edge detection identifies locations in an image where intensity changes
sharply. These boundaries carry the most informative structural content
and are the starting point for many computer vision pipelines.

## Gradient-based edge detection

An edge corresponds to a large spatial derivative. For a 2D image
$f(i, j)$ the gradient is a vector:

$$
\nabla f = \left(\frac{\partial f}{\partial i},\; \frac{\partial f}{\partial j}\right)
$$

The **gradient magnitude** indicates edge strength:

$$
|\nabla f| = \sqrt{\left(\frac{\partial f}{\partial i}\right)^2 + \left(\frac{\partial f}{\partial j}\right)^2}
$$

and the **gradient direction** indicates edge orientation:

$$
\theta = \arctan\!\left(\frac{\partial f / \partial i}{\partial f / \partial j}\right)
$$

## Sobel operator

The Sobel operator approximates the gradient using $3 \times 3$
convolution kernels:

$$
G_x =
\begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix},
\quad
G_y =
\begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}
$$

The gradient magnitude is then $|\nabla f| \approx \sqrt{G_x^2 + G_y^2}$.

```cpp
#include <sw/dsp/image/edge.hpp>

using namespace sw::dsp;

Image<double, 1> img(480, 640);
// ... load image ...

// Horizontal and vertical gradients
auto gx = image::sobel_x(img);
auto gy = image::sobel_y(img);

// Combined gradient magnitude
auto edges = image::gradient_magnitude(gx, gy);
```

## Laplacian

The Laplacian is a second-order derivative operator that detects edges as
zero crossings:

$$
\nabla^2 f = \frac{\partial^2 f}{\partial i^2} + \frac{\partial^2 f}{\partial j^2}
$$

A common discrete approximation uses the $3 \times 3$ kernel:

$$
L =
\begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}
$$

```cpp
auto lap = image::laplacian(img);
```

The Laplacian responds to edges of all orientations equally but is more
sensitive to noise than the Sobel operator.

## Canny edge detection

The Canny algorithm is a multi-stage edge detector that produces thin,
well-localized edges with minimal false detections:

1. **Gaussian smoothing** -- suppress noise with a Gaussian blur
2. **Gradient computation** -- compute gradient magnitude and direction
   (typically via Sobel)
3. **Non-maximum suppression** -- thin edges by keeping only local maxima
   along the gradient direction
4. **Hysteresis thresholding** -- use two thresholds $t_{\text{low}}$ and
   $t_{\text{high}}$: strong edges (above $t_{\text{high}}$) are kept,
   weak edges (between $t_{\text{low}}$ and $t_{\text{high}}$) are kept
   only if connected to a strong edge

```cpp
// Canny edge detection with sigma=1.4, low=0.05, high=0.15
auto canny_edges = image::canny(img, 1.4, 0.05, 0.15);
```

### Parameter selection

| Parameter         | Effect of increasing                      |
|-------------------|-------------------------------------------|
| $\sigma$          | More smoothing, fewer noise edges, less detail |
| $t_{\text{low}}$  | Fewer weak edges retained                 |
| $t_{\text{high}}$ | Fewer strong edges, sparser output        |

## The precision argument for edge detection

Edge detection is especially sensitive to arithmetic precision because
it computes **differences** of neighboring pixel values. When the input
signal is noisy and the pixel type has limited precision, the following
problems arise:

- **Gradient quantization**: small but real edges vanish because the
  narrow type cannot represent the difference.
- **False edges from rounding**: quantization noise creates artificial
  gradients that the detector reports as edges.

Using a wider accumulator type during the Sobel convolution preserves
small gradient differences that a narrow type would discard:

```cpp
using Pixel = sw::universal::posit<8, 0>;
using State = double;

Image<Pixel, 1> img(480, 640);

// Sobel with accumulation in double precision
auto gx = image::sobel_x<Pixel, State>(img);
auto gy = image::sobel_y<Pixel, State>(img);
```

This mixed-precision approach is particularly valuable in sensor
applications where the raw data arrives in a narrow format (e.g., 8-bit
or 10-bit ADC output) but the processing pipeline needs higher fidelity
to distinguish real edges from sensor noise.

## Summary of API

| Function                | Description                          |
|-------------------------|--------------------------------------|
| `image::sobel_x(img)`   | Horizontal gradient via Sobel kernel |
| `image::sobel_y(img)`   | Vertical gradient via Sobel kernel   |
| `image::laplacian(img)` | Second-order Laplacian edges         |
| `image::canny(img, ...)` | Full Canny edge detection pipeline  |
| `image::gradient_magnitude(gx, gy)` | $\sqrt{g_x^2 + g_y^2}$ |
