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
	FetchContent_Declare(universal
		GIT_REPOSITORY https://github.com/stillwater-sc/universal.git
		GIT_TAG        v4.6.9
		GIT_SHALLOW    TRUE
	)
	# Header-only: fetch source but do NOT add_subdirectory (avoids building Universal's targets)
	FetchContent_GetProperties(universal)
	if(NOT universal_POPULATED)
		FetchContent_Populate(universal)
	endif()
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
	FetchContent_Declare(mtl5
		GIT_REPOSITORY https://github.com/stillwater-sc/mtl5.git
		GIT_TAG        main
		GIT_SHALLOW    TRUE
	)
	# Header-only: fetch source but do NOT add_subdirectory (avoids building MTL5's targets)
	FetchContent_GetProperties(mtl5)
	if(NOT mtl5_POPULATED)
		FetchContent_Populate(mtl5)
	endif()
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
