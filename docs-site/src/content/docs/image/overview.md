---
title: Image Processing Overview
description: Images as 2D signals, the Image container, and I/O support in mixed-precision DSP
---

Images are two-dimensional signals. Each pixel is a sample, and the
spatial axes (row, column) replace the single time axis of 1D signal
processing. The `sw::dsp` image processing module brings the same
mixed-precision philosophy to 2D operations.

## Images as 2D signals

A grayscale image is a real-valued function $f(i, j)$ sampled on a
discrete grid. A color image extends this to multiple channels (e.g., RGB
with three planes). All filtering, transform, and analysis concepts from
1D DSP generalize naturally to two dimensions.

## The Image container

The library provides an `Image<T, Channels>` class template that stores
pixel data in **planar layout** -- each channel is a separate
`mtl::mat::dense2D<T>` matrix:

```cpp
#include <sw/dsp/image/image.hpp>

using namespace sw::dsp;

// Single-channel (grayscale) image, 640x480
Image<double, 1> gray(480, 640);

// Three-channel (RGB) image
Image<double, 3> color(480, 640);

// Access the red channel (channel 0)
mtl::mat::dense2D<double>& red = color.channel(0);
```

### Pixel value conventions

All pixel values are normalized to the range $[0, 1]$:

| Value | Meaning       |
|-------|---------------|
| 0.0   | Black (minimum intensity) |
| 1.0   | White (maximum intensity) |

This convention avoids integer-scaling ambiguities and integrates cleanly
with floating-point and posit arithmetic types.

### Using alternative number types

The type parameter `T` can be any arithmetic type supported by the
library, including Universal number types:

```cpp
using Pixel = sw::universal::posit<16, 2>;
Image<Pixel, 1> narrow_image(480, 640);
```

## Synthetic image generators

The library includes generators for creating test images useful in
algorithm development and validation:

```cpp
#include <sw/dsp/image/generators.hpp>

// 8x8 checkerboard pattern on a 256x256 image
auto checker = image::checkerboard<double>(256, 256, 8);

// Horizontal gradient from 0 to 1
auto grad = image::gradient<double>(256, 256);

// Vertical step edge at column 128
auto edge = image::step_edge<double>(256, 256, 128);
```

### Checkerboard

Alternating black and white squares of configurable size. Useful for
testing spatial frequency response and aliasing behavior.

### Gradient

A smooth ramp from 0 to 1 along one axis. Useful for verifying intensity
mapping and quantization effects across the full dynamic range.

### Step edge

A sharp transition from 0 to 1 at a specified column (or row). The
canonical test input for edge detection algorithms.

## Image I/O

The library supports reading and writing images in several formats:

```cpp
#include <sw/dsp/image/io.hpp>

// Load a grayscale PGM image
auto img = image::load_pgm<double>("input.pgm");

// Save as PGM
image::save_pgm(img, "output.pgm");

// Load a color PPM image
auto color_img = image::load_ppm<double>("photo.ppm");

// Save as PPM
image::save_ppm(color_img, "photo_out.ppm");

// BMP format (8-bit grayscale or 24-bit RGB)
auto bmp = image::load_bmp<double>("input.bmp");
image::save_bmp(bmp, "output.bmp");
```

### Supported formats

| Format | Extension | Channels | Notes |
|--------|-----------|----------|-------|
| PGM    | `.pgm`    | 1        | Portable Graymap, ASCII or binary |
| PPM    | `.ppm`    | 3        | Portable Pixmap, ASCII or binary  |
| BMP    | `.bmp`    | 1 or 3   | Windows Bitmap, uncompressed      |

These formats were chosen for simplicity and portability. They require no
external codec libraries, keeping the build dependency-free.

## Mixed-precision image processing

The three-scalar parameterization from the filter module extends to image
operations. A typical configuration uses a narrow pixel type for storage
and a wider type for intermediate computations:

```cpp
using Pixel  = sw::universal::posit<8, 0>;   // 8-bit posit storage
using Kernel = double;                        // full-precision kernels
using State  = double;                        // accumulator

// Load image into narrow pixel type
Image<Pixel, 1> img(480, 640);

// Convolution accumulates in State precision
auto blurred = image::gaussian_blur<Kernel, State>(img, 5, 1.0);
```

This approach reduces memory footprint for large images while preserving
the accuracy of spatial filtering operations.
