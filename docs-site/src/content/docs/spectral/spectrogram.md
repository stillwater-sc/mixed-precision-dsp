---
title: Spectrograms
description: Short-time Fourier transform and time-frequency analysis in mixed-precision DSP
---

A spectrogram displays how the frequency content of a signal evolves over
time. It is built from the Short-Time Fourier Transform (STFT), which
applies the DFT to successive windowed segments of a signal.

## Short-Time Fourier Transform

The STFT of a signal $x[n]$ with window $w[n]$ of length $M$ is:

$$
X[m, k] = \sum_{n=0}^{M-1} x[n + mH]\, w[n]\, e^{-j2\pi kn/M}
$$

where $m$ is the time-frame index and $H$ is the hop size (the number of
samples between successive frames). The result is a 2D complex matrix with
time along one axis and frequency along the other.

## Time-frequency tradeoff

The window length $M$ controls a fundamental tradeoff:

| Window length | Frequency resolution | Time resolution |
|--------------|---------------------|-----------------|
| Long         | High ($\Delta f = f_s/M$ is small) | Low (each frame spans more time) |
| Short        | Low ($\Delta f$ is large) | High (frames are narrow in time) |

This tradeoff is a consequence of the uncertainty principle: you cannot
have arbitrarily fine resolution in both time and frequency simultaneously.
Choosing the right window length depends on the application.

## Spectrogram as magnitude-squared STFT

The spectrogram is the squared magnitude of the STFT:

$$
S[m, k] = |X[m, k]|^2
$$

It is typically displayed as a heatmap with time on the horizontal axis,
frequency on the vertical axis, and color representing power (often in
decibels).

## Library API

The `sw::dsp::spectral::spectrogram()` function computes the spectrogram
in one call:

```cpp
#include <sw/dsp/spectral/spectrogram.hpp>

using namespace sw::dsp;

mtl::vec::dense_vector<double> signal(44100);  // 1 second at 44.1 kHz
// ... fill signal ...

// Compute spectrogram: 512-sample window, hop size 128, Hann window
auto S = spectral::spectrogram(signal, 512, 128, window::hann);
```

The return value is an `mtl::mat::dense2D<double>` matrix where each row
corresponds to a time frame and each column to a frequency bin.

### Accessing individual frames

Each row of the spectrogram matrix is one time frame. You can extract a
single frame for further analysis:

```cpp
// Number of frames and frequency bins
size_t nframes = num_rows(S);
size_t nbins   = num_cols(S);

// Convert the entire spectrogram to decibels
auto S_db = spectral::psd_db(S);
```

### Mixed-precision STFT

The three-scalar parameterization lets you store samples in a compact
type while accumulating the FFT butterflies in a wider type:

```cpp
using Sample = sw::universal::posit<16, 2>;
using State  = double;

mtl::vec::dense_vector<Sample> signal(44100);
// ...

auto S = spectral::spectrogram<State, Sample>(
    signal, 512, 128, window::hann);
```

## Choosing parameters

### Window length

For **speech analysis** a window of 20--40 ms captures individual phonemes
while providing adequate frequency resolution. At 16 kHz sampling rate
this corresponds to 320--640 samples.

For **vibration monitoring** in rotating machinery, longer windows
(hundreds of milliseconds) are used to resolve closely spaced harmonics.

### Hop size

The hop size $H$ determines the time spacing between frames. Common
choices are $H = M/4$ (75% overlap) or $H = M/2$ (50% overlap). Smaller
hop sizes give smoother time evolution at the cost of more computation.

### Window function

The Hann window is the default choice for most spectrogram work. When
sidelobe suppression is critical (e.g., detecting weak tones near strong
ones), the Blackman-Harris window is preferable.

## Applications

### Speech analysis

Spectrograms are the standard visualization for speech signals. Formant
frequencies appear as horizontal bands, plosive consonants as vertical
bursts, and fricatives as broadband noise. Mixed-precision computation
allows efficient real-time spectrogram display on resource-constrained
devices.

### Vibration monitoring

In condition monitoring of rotating machinery, the spectrogram reveals
how vibration harmonics shift as speed changes. Bearing faults appear as
characteristic frequency patterns that evolve over time.

### Music information retrieval

Spectrograms form the input to many music analysis algorithms, including
pitch tracking, onset detection, and instrument classification. The
mel-scaled spectrogram is a common variant that warps the frequency axis
to match human pitch perception.
