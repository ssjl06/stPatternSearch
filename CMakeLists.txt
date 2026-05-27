cmake_minimum_required(VERSION 3.22)
project(fullchipUSC
    VERSION 0.1.0
    DESCRIPTION "Distributed greedy minimum set cover for OPC segment dedup (fullchip USC)"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(FULLCHIPUSC_BUILD_TESTS "Build unit tests (GoogleTest via FetchContent)" ON)
option(FULLCHIPUSC_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter)
endif()
if(FULLCHIPUSC_WARNINGS_AS_ERRORS)
    if(MSVC)
        add_compile_options(/WX)
    else()
        add_compile_options(-Werror)
    endif()
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(stComm REQUIRED)
message(STATUS "fullchipUSC: MPI backend (${MPI_CXX_COMPILER})")
message(STATUS "fullchipUSC: stComm found")

add_subdirectory(src)

if(FULLCHIPUSC_BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.15.2
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    add_subdirectory(tests)
endif()
