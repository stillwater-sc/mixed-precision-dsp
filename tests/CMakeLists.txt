include(CTest)

# Helper: create a test executable, place it under a Visual Studio
# folder (mirroring the include/sw/dsp/ module layout), and register
# it with ctest. The `folder` argument is a "/"-separated path the
# IDE displays in its Solution Explorer; Makefile / Ninja generators
# ignore the FOLDER target property so this is a no-op for them.
#
# Modeled after Universal's compile_all() macro, simplified for our
# 1-test-per-source-file convention. See README.md in this directory
# for the test-folder hierarchy and rationale.
function(dsp_add_test test_name folder)
	add_executable(${test_name} ${test_name}.cpp)
	target_link_libraries(${test_name} PRIVATE sw::dsp)
	set_target_properties(${test_name} PROPERTIES FOLDER "${folder}")
	add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# =============================================================================
# Foundation — concept satisfaction + cross-cutting type system tests
# =============================================================================
dsp_add_test(test_concepts        "Tests/Foundation")
dsp_add_test(test_projection      "Tests/Foundation")

# =============================================================================
# Signals — generators + I/O
# =============================================================================
dsp_add_test(test_generators      "Tests/Signals")
dsp_add_test(test_io              "Tests/Signals")

# =============================================================================
# Windows
# =============================================================================
dsp_add_test(test_windows         "Tests/Windows")

# =============================================================================
# Quantization
# =============================================================================
dsp_add_test(test_quantization    "Tests/Quantization")

# =============================================================================
# Filter — IIR designs, FIR designs, and design-agnostic filter operations
# =============================================================================
# IIR
dsp_add_test(test_biquad          "Tests/Filter/IIR")
dsp_add_test(test_butterworth     "Tests/Filter/IIR")
dsp_add_test(test_chebyshev       "Tests/Filter/IIR")
dsp_add_test(test_bessel          "Tests/Filter/IIR")
dsp_add_test(test_elliptic        "Tests/Filter/IIR")
# FIR
dsp_add_test(test_fir             "Tests/Filter/FIR")
dsp_add_test(test_fir_multirate   "Tests/Filter/FIR")
dsp_add_test(test_remez           "Tests/Filter/FIR")
# Generic — design-agnostic post-processing (see Tests/README.md)
dsp_add_test(test_filtfilt        "Tests/Filter/Generic")

# =============================================================================
# Spectral — general spectral analysis (FFT, PSD, spectrogram)
# =============================================================================
dsp_add_test(test_fft             "Tests/Spectral")
dsp_add_test(test_spectral        "Tests/Spectral")
dsp_add_test(test_bluestein       "Tests/Spectral")

# =============================================================================
# Spectrum — spectrum-analyzer-specific primitives (detectors, RBW, VBW,
# swept LO, real-time spectrum, spectrogram buffer). See Epic #134.
# Note: distinct from Spectral above (Spectral = general DSP transforms;
# Spectrum = analyzer-specific stages between those transforms and the
# trace memory).
# =============================================================================
dsp_add_test(test_spectrum_detectors        "Tests/Spectrum")
dsp_add_test(test_spectrum_trace_averaging  "Tests/Spectrum")
dsp_add_test(test_spectrum_markers          "Tests/Spectrum")
dsp_add_test(test_spectrum_vbw_filter       "Tests/Spectrum")
dsp_add_test(test_spectrum_rbw_filter       "Tests/Spectrum")
dsp_add_test(test_spectrum_realtime         "Tests/Spectrum")

# =============================================================================
# Conditioning — AGC, compressor, envelope, sample-rate conversion, noise shaping
# =============================================================================
dsp_add_test(test_conditioning    "Tests/Conditioning")
dsp_add_test(test_src             "Tests/Conditioning")
dsp_add_test(test_noise_shaping   "Tests/Conditioning")

# =============================================================================
# Acquisition — SDR / IF receiver primitives
# =============================================================================
dsp_add_test(test_cic             "Tests/Acquisition")
dsp_add_test(test_halfband        "Tests/Acquisition")
dsp_add_test(test_nco             "Tests/Acquisition")
dsp_add_test(test_ddc             "Tests/Acquisition")
dsp_add_test(test_decimation_chain "Tests/Acquisition")

# =============================================================================
# Instrument — oscilloscope / spectrum analyzer primitives (see #132)
# =============================================================================
dsp_add_test(test_instrument_trigger           "Tests/Instrument")
dsp_add_test(test_instrument_ring_buffer       "Tests/Instrument")
dsp_add_test(test_instrument_calibration       "Tests/Instrument")
target_compile_definitions(test_instrument_calibration PRIVATE
	TESTS_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data")
dsp_add_test(test_instrument_peak_detect       "Tests/Instrument")
dsp_add_test(test_instrument_display_envelope  "Tests/Instrument")
dsp_add_test(test_instrument_fractional_delay  "Tests/Instrument")
dsp_add_test(test_instrument_channel_aligner   "Tests/Instrument")
dsp_add_test(test_instrument_measurements      "Tests/Instrument")

# =============================================================================
# Analysis — stability, sensitivity, condition number, and per-pipeline
# precision-analysis primitives (e.g., acquisition_precision)
# =============================================================================
dsp_add_test(test_analysis              "Tests/Analysis")
dsp_add_test(test_acquisition_precision "Tests/Analysis")

# =============================================================================
# Estimation — Kalman, LMS, RLS adaptive estimators
# =============================================================================
dsp_add_test(test_kalman          "Tests/Estimation")

# =============================================================================
# Image
# =============================================================================
dsp_add_test(test_image           "Tests/Image")
dsp_add_test(test_image_generators "Tests/Image")
dsp_add_test(test_image_io        "Tests/Image")
