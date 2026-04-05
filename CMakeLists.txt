cmake_minimum_required(VERSION 3.22)

if(NOT DEFINED DSP_MAJOR)
	set(DSP_MAJOR 1)
	set(DSP_MINOR 0)
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
# Universal number arithmetic library
find_package(universal CONFIG QUIET)
if(NOT universal_FOUND)
	# Fall back to local path for development
	set(UNIVERSAL_DIR "" CACHE PATH "Path to Universal library root")
	if(EXISTS "${UNIVERSAL_DIR}/include/sw/universal")
		target_include_directories(${PROJECT_NAME} INTERFACE
			$<BUILD_INTERFACE:${UNIVERSAL_DIR}/include>)
		message(STATUS "Using Universal from: ${UNIVERSAL_DIR}")
	elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../universal/include/sw/universal")
		target_include_directories(${PROJECT_NAME} INTERFACE
			$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../universal/include>)
		message(STATUS "Using Universal from sibling directory")
	else()
		message(WARNING "Universal library not found. Some features will be unavailable.")
	endif()
else()
	target_link_libraries(${PROJECT_NAME} INTERFACE universal::universal)
	message(STATUS "Found Universal via find_package")
endif()

# MTL5 linear algebra library
find_package(MTL5 CONFIG QUIET)
if(NOT MTL5_FOUND)
	set(MTL5_DIR "" CACHE PATH "Path to MTL5 library root")
	if(EXISTS "${MTL5_DIR}/include/mtl")
		target_include_directories(${PROJECT_NAME} INTERFACE
			$<BUILD_INTERFACE:${MTL5_DIR}/include>)
		message(STATUS "Using MTL5 from: ${MTL5_DIR}")
	elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../universal/mtl5/include/mtl")
		target_include_directories(${PROJECT_NAME} INTERFACE
			$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../universal/mtl5/include>)
		message(STATUS "Using MTL5 from universal/mtl5 sibling directory")
	else()
		message(WARNING "MTL5 library not found. Some features will be unavailable.")
	endif()
else()
	target_link_libraries(${PROJECT_NAME} INTERFACE MTL5::mtl5)
	message(STATUS "Found MTL5 via find_package")
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
