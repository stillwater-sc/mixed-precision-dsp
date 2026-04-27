cmake_minimum_required(VERSION 3.22)

if(NOT DEFINED DSP_MAJOR)
	set(DSP_MAJOR 0)
	set(DSP_MINOR 5)
	set(DSP_PATCH 0)
endif()

project(dsp
	DESCRIPTION "Header-only mixed-precision DSP library"
	VERSION "${DSP_MAJOR}.${DSP_MINOR}.${DSP_PATCH}"
	LANGUAGES CXX
	HOMEPAGE_URL "https://github.com/stillwater-sc/dsp"
)

set(PROJECT_VERSION_MAJOR ${DSP_MAJOR})
set(PROJECT_VERSION_MINOR ${DSP_MINOR})
set(PROJECT_VERSION_PATCH ${DSP_PATCH})

# Header-only interface library
add_library(${PROJECT_NAME} INTERFACE)
add_library(sw::dsp ALIAS ${PROJECT_NAME})

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

target_include_directories(${PROJECT_NAME} INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

####
# Dependencies
#
# Strategy: try find_package first (for system/installed copies),
# then fall back to FetchContent from GitHub (headers only).
include(FetchContent)

# Universal number arithmetic library (header-only)
find_package(universal CONFIG QUIET)
if(universal_FOUND)
	target_link_libraries(${PROJECT_NAME} INTERFACE universal::universal)
	message(STATUS "Found Universal via find_package")
else()
	message(STATUS "Universal not found via find_package, fetching headers from GitHub")
	# Header-only consumption: we want to fetch the source tree but NOT
	# call add_subdirectory on Universal's CMakeLists.txt (which would
	# build its tests, examples, and other targets we don't need).
	#
	# The classic idiom for this — FetchContent_GetProperties + manual
	# FetchContent_Populate — is deprecated as of CMake 3.30 (CMP0169).
	# The forward-compatible pattern is FetchContent_MakeAvailable with
	# SOURCE_SUBDIR pointing at a non-existent subdirectory: MakeAvailable
	# only calls add_subdirectory when the subdir contains a CMakeLists.txt,
	# so a path that doesn't exist gives us populate-without-build.
	FetchContent_Declare(universal
		GIT_REPOSITORY https://github.com/stillwater-sc/universal.git
		GIT_TAG        v4.6.10
		GIT_SHALLOW    TRUE
		SOURCE_SUBDIR  _do_not_build_
	)
	FetchContent_MakeAvailable(universal)
	# Universal headers use two include conventions:
	#   - External: #include <sw/universal/...>  (needs include/)
	#   - Internal: #include <universal/...>     (needs include/sw/)
	target_include_directories(${PROJECT_NAME} INTERFACE
		$<BUILD_INTERFACE:${universal_SOURCE_DIR}/include>
		$<BUILD_INTERFACE:${universal_SOURCE_DIR}/include/sw>)
endif()

# MTL5 linear algebra library (header-only)
find_package(MTL5 CONFIG QUIET)
if(MTL5_FOUND)
	target_link_libraries(${PROJECT_NAME} INTERFACE MTL5::mtl5)
	message(STATUS "Found MTL5 via find_package")
else()
	message(STATUS "MTL5 not found via find_package, fetching headers from GitHub")
	# Same SOURCE_SUBDIR-doesn't-exist pattern as Universal above.
	FetchContent_Declare(mtl5
		GIT_REPOSITORY https://github.com/stillwater-sc/mtl5.git
		GIT_TAG        main
		GIT_SHALLOW    TRUE
		SOURCE_SUBDIR  _do_not_build_
	)
	FetchContent_MakeAvailable(mtl5)
	target_include_directories(${PROJECT_NAME} INTERFACE
		$<BUILD_INTERFACE:${mtl5_SOURCE_DIR}/include>)
endif()

####
# Compiler-specific settings
target_compile_options(${PROJECT_NAME} INTERFACE
	$<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->
	$<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
	$<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
	$<$<CXX_COMPILER_ID:AppleClang>:-Wall -Wextra -Wpedantic>
)

####
# Build options
option(DSP_BUILD_APPLICATIONS "Build demonstration applications" ON)
option(DSP_BUILD_TESTS        "Build unit tests"                 ON)

if(DSP_BUILD_APPLICATIONS)
	add_subdirectory(applications)
endif()

if(DSP_BUILD_TESTS)
	enable_testing()
	add_subdirectory(tests)
endif()

####
# Install
include(GNUInstallDirs)

install(TARGETS ${PROJECT_NAME}
	EXPORT dspTargets
)

install(DIRECTORY include/
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
	FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT dspTargets
	FILE dspTargets.cmake
	NAMESPACE sw::
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dsp
)

# Package configuration
include(CMakePackageConfigHelpers)

configure_package_config_file(
	${CMAKE_CURRENT_SOURCE_DIR}/cmake/dspConfig.cmake.in
	${CMAKE_CURRENT_BINARY_DIR}/dspConfig.cmake
	INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dsp
)

write_basic_package_version_file(
	${CMAKE_CURRENT_BINARY_DIR}/dspConfigVersion.cmake
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY SameMajorVersion
)

install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/dspConfig.cmake
	${CMAKE_CURRENT_BINARY_DIR}/dspConfigVersion.cmake
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dsp
)
