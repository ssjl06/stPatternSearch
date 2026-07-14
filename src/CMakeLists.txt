add_library(stPS STATIC
    # core/ — shared, algorithm-neutral primitives + the hashed patch set
    core/bitset.cpp
    core/csr.cpp
    core/inverted_index.cpp
    core/device_buffer.cu
    core/gpu_profiler.cu
    core/host_profiler.cpp
    core/patch_set.cpp
    # usc/ — greedy minimum set-cover algorithm
    usc/usc_patch_selector_impl.cu
    usc/brute_force.cpp
    # ups/ — pattern statistics (hash counts + representative locations)
    ups/ups_pattern_stats.cu
    ups/stats_file.cpp
    # io/ — patch file readers/writers (M7)
    io/stps_binary.cpp
    # data/ — demo/test data generation + partitioning
    data/synthetic.cpp
    data/partition.cpp
)
add_library(stPS::stPS ALIAS stPS)

# -DUSC_PROFILE=ON compiles in the per-stage profiler calls (usc/usc_patch_selector_impl.cu
# guards them behind #ifdef USC_PROFILE); off by default = zero-overhead hot path.
if(USC_PROFILE)
    target_compile_definitions(stPS PRIVATE USC_PROFILE)
endif()

# usc_patch_selector_impl.cu / device_buffer.cu are compiled by nvcc — link the CUDA runtime
# so DeviceBuffer's cudaMalloc/cudaFree resolve. PIC + separable compilation let
# the static lib drop into mixed host/device consumers without surprises.
set_target_properties(stPS PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    # Resolve device symbols when the static lib is created, so the .a is
    # self-contained and plain (non-CUDA) consumers can link it without a
    # device-link step of their own.
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Public headers live in include/ (installed); internal headers under src/ stay
# private so consumers never see core/ or data/ implementation details.
target_include_directories(stPS
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(stPS PUBLIC
    MPI::MPI_CXX
    stComm::stComm
    CUDA::cudart
)

install(TARGETS stPS
    EXPORT stPSTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Demo / CLI driver. usc/main.cpp quote-includes the internal "data/synthetic.hpp",
# resolved via the src/ include root added below; it drives the selection through
# the public stPS::UscPatchSelector from <stPS/...> (carried by the link below).
add_executable(usc-patch-select usc/main.cpp)
target_include_directories(usc-patch-select PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(usc-patch-select PRIVATE stPS)

# UPS driver: top-K pattern statistics (count + representative location) via
# the public stPS::UpsPatternStats, written to a single file by all ranks in
# parallel. The src/ include root is only for the demo-data/io helpers.
add_executable(ups-pattern-stats ups/main.cpp)
target_include_directories(ups-pattern-stats PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ups-pattern-stats PRIVATE stPS)
