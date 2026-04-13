include(CTest)

# Helper function to create test executables
function(dsp_add_test test_name)
	add_executable(${test_name} ${test_name}.cpp)
	target_link_libraries(${test_name} PRIVATE sw::dsp)
	add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Phase 1: Foundation
dsp_add_test(test_concepts)

# Phase 2: Biquad Engine + Cascade
dsp_add_test(test_biquad)

# Phase 3: Transforms + Butterworth
dsp_add_test(test_butterworth)

# Signal generators (Issue #10)
dsp_add_test(test_generators)

# Signal file I/O (Issue #11)
dsp_add_test(test_io)

# Chebyshev I/II + RBJ (Issue #2)
dsp_add_test(test_chebyshev)

# Bessel + Legendre (Issue #2)
dsp_add_test(test_bessel)

# Elliptic (Issue #2)
dsp_add_test(test_elliptic)

# Windows + Signal + Sampling (Issue #3)
dsp_add_test(test_windows)

# Quantization (Issue #3)
dsp_add_test(test_quantization)

# FIR filters (Issue #4)
dsp_add_test(test_fir)

# Spectral methods + TransferFunction (Issue #5)
dsp_add_test(test_fft)

# Spectral analysis: Z-transform, Laplace, PSD, spectrogram (Issue #5)
dsp_add_test(test_spectral)

# Signal conditioning (Issue #6)
dsp_add_test(test_conditioning)

# Estimation: Kalman, LMS, RLS (Issue #7)
dsp_add_test(test_kalman)

# Image processing (Issue #8)
dsp_add_test(test_image)
