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
