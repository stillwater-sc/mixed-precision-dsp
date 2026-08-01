# Regression tests — large-scale / long-running correctness tests.
#
# These tests are slower than unit tests and are gated behind
# DSP_BUILD_REGRESSION_TESTS (OFF by default). Enable via the
# `ci-regression` CMake preset or `-DDSP_BUILD_REGRESSION_TESTS=ON`.
#
# CTest labels:
#   ctest -L regression         — run ONLY regression tests
#   ctest -LE regression        — run ONLY unit tests (excludes regression)
#   ctest                       — run ALL tests (unit + regression)
#
# Default timeout per regression test: 300 seconds.

# Helper: build a regression test, tag it with the "regression" CTest
# label, and give it a longer default timeout. Same shape as
# dsp_add_test() in the parent CMakeLists.txt.
function(dsp_add_regression_test test_name folder)
	add_executable(${test_name} ${test_name}.cpp)
	target_link_libraries(${test_name} PRIVATE sw::dsp)
	set_target_properties(${test_name} PROPERTIES FOLDER "Tests/Regression/${folder}")
	add_test(NAME ${test_name} COMMAND ${test_name})
	set_tests_properties(${test_name} PROPERTIES
		LABELS "regression"
		TIMEOUT 300
	)
endfunction()

# Add regression tests here as they land, e.g.:
#   dsp_add_regression_test(regression_filter_long_run "Filter")
