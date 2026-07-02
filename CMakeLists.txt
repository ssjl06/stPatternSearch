cmake_minimum_required(VERSION 3.22)

# CUDA defaults — set BEFORE project() so LANGUAGES CUDA picks them up. Users
# can override at configure time with -DCMAKE_CUDA_ARCHITECTURES=... or
# -DCMAKE_CUDA_COMPILER=...
if(NOT DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES})
    # MX450 = compute capability 7.5
    set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA target architectures")
endif()
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

project(stPatternSearch
    VERSION 0.1.0
    DESCRIPTION "Distributed pattern-search library (USC: greedy minimum set cover for OPC segment dedup)"
    LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Match host C++ standard. nvcc 12 supports C++20 in device code.
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(STPS_BUILD_TESTS "Build unit tests (GoogleTest via FetchContent)" ON)
option(STPS_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
# Per-stage profiling in the USC solve loop (CUDA-event + host timers). Off by
# default so the production hot path carries no synchronization; enable with
# -DUSC_PROFILE=ON to get a per-iteration stage breakdown on stderr.
option(USC_PROFILE "Enable USC per-stage profiling instrumentation" OFF)

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter)
endif()
if(STPS_WARNINGS_AS_ERRORS)
    if(MSVC)
        add_compile_options(/WX)
    else()
        add_compile_options(-Werror)
    endif()
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(stComm REQUIRED)
find_package(CUDAToolkit REQUIRED)
message(STATUS "stPS: MPI backend (${MPI_CXX_COMPILER})")
message(STATUS "stPS: stComm found")
message(STATUS "stPS: CUDA toolkit ${CUDAToolkit_VERSION}, arch=${CMAKE_CUDA_ARCHITECTURES}")

include(GNUInstallDirs)

add_subdirectory(src)

if(STPS_BUILD_TESTS)
    enable_testing()
    # Use a system-installed GoogleTest. Point CMake at a custom install with
    # any of the standard hints, e.g.:
    #   -DGTEST_ROOT=/path/to/gtest        (FindGTest module hint)
    #   -DGTest_ROOT=/path/to/gtest        (<Package>_ROOT, CMP0074)
    #   -DCMAKE_PREFIX_PATH=/path/to/gtest (generic prefix)
    find_package(GTest REQUIRED)
    message(STATUS "stPS: GTest found (${GTest_DIR})")
    add_subdirectory(tests)
endif()

# ============================================================================
# Install / packaging — consumers do: find_package(stPS) +
# target_link_libraries(their_target PRIVATE stPS::stPS)
# ============================================================================

# Public headers
install(DIRECTORY include/stPS
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/stPSConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/stPSConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stPS
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/stPSConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/stPSConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/stPSConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stPS
)

install(EXPORT stPSTargets
    FILE stPSTargets.cmake
    NAMESPACE stPS::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stPS
)

message(STATUS "")
message(STATUS "========================================")
message(STATUS "stPS installation configuration:")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Headers: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/stPS")
message(STATUS "  Library: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
message(STATUS "  CMake config: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/stPS")
message(STATUS "========================================")
message(STATUS "")
