# Applications (demonstration programs)

# Helper: create an application executable, link sw::dsp, and place it
# under a Visual Studio folder. Same pattern as dsp_add_test in
# tests/CMakeLists.txt — see tests/README.md for the broader rationale
# behind the IDE folder organization.
#
# Defined at the parent (applications/) scope so each sub-directory's
# CMakeLists.txt can call it directly. CMake function visibility flows
# from parent to child via add_subdirectory().
# Demos write their CSV output here rather than to the working directory,
# so running one from the repository root no longer leaves files in the
# source tree. Every demo still accepts --csv=<path> (or a positional
# output directory) to override. See applications/common/demo_output.hpp.
set(DSP_APPLICATIONS_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(DSP_DEMO_OUTPUT_DIR  "${CMAKE_BINARY_DIR}/demo-output")
file(MAKE_DIRECTORY "${DSP_DEMO_OUTPUT_DIR}")

function(dsp_add_application target_name folder)
	add_executable(${target_name} ${target_name}.cpp)
	target_link_libraries(${target_name} PRIVATE sw::dsp)
	# <common/demo_output.hpp> resolves against applications/.
	target_include_directories(${target_name} PRIVATE "${DSP_APPLICATIONS_DIR}")
	target_compile_definitions(${target_name} PRIVATE
		DSP_DEMO_OUTPUT_DIR="${DSP_DEMO_OUTPUT_DIR}")
	set_target_properties(${target_name} PROPERTIES FOLDER "${folder}")
endfunction()

# Each subdirectory contains its own CMakeLists.txt that uses
# dsp_add_application(name "Applications/<Folder>"). The folder string
# determines where the target appears in IDE Solution Explorer trees.
# See applications/README.md for the folder hierarchy.

# IIR / FIR filter demos
add_subdirectory(iir_demo)
add_subdirectory(fir_demo)

# Phase 7: Spectral methods (placeholder, not yet implemented)
# add_subdirectory(spectral_demo)

# Phase 5: Quantization (placeholder, not yet implemented)
# add_subdirectory(quantization_demo)

# Phase 9: Estimation
add_subdirectory(estimation_demo)

# Phase 10: Image processing
add_subdirectory(image_demo)

# Mixed-precision comparison (Epic #22)
add_subdirectory(mp_comparison)

# Flagship precision sweep tool (Issue #69)
add_subdirectory(precision_sweep)

# FFT trade-off analysis (Issue #74)
add_subdirectory(fft_tradeoff)

# Acquisition pipeline demo (Issue #93)
add_subdirectory(acquisition_demo)

# SDR link demo (Issue #104) — capstone for the SDR epic #85
add_subdirectory(sdr_demo)

# Scope demonstrator (Issue #152) — capstone for #133
add_subdirectory(scope_demo)

# Two-channel scope extension (Issue #173) — CrossChannelTrigger +
# ChannelAligner over the scope_demo pipeline shape.
add_subdirectory(scope_demo_2ch)

# Spectrum-analyzer demonstrator (Issue #183) — capstone for #134
add_subdirectory(spectrum_analyzer_demo)

# Multirate examples (Issue #136) — audio resampler + future rational SRC
add_subdirectory(multirate_examples)

# Phase 8: Signal conditioning (placeholder, not yet implemented)
# add_subdirectory(conditioning_demo)
