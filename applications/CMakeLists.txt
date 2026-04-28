# Applications (demonstration programs)

# Helper: create an application executable, link sw::dsp, and place it
# under a Visual Studio folder. Same pattern as dsp_add_test in
# tests/CMakeLists.txt — see tests/README.md for the broader rationale
# behind the IDE folder organization.
#
# Defined at the parent (applications/) scope so each sub-directory's
# CMakeLists.txt can call it directly. CMake function visibility flows
# from parent to child via add_subdirectory().
function(dsp_add_application target_name folder)
	add_executable(${target_name} ${target_name}.cpp)
	target_link_libraries(${target_name} PRIVATE sw::dsp)
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

# Scope demonstrator (Issue #152) — capstone for #133
add_subdirectory(scope_demo)

# Spectrum-analyzer demonstrator (Issue #183) — capstone for #134
add_subdirectory(spectrum_analyzer_demo)

# Phase 8: Signal conditioning (placeholder, not yet implemented)
# add_subdirectory(conditioning_demo)
