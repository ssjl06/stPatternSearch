add_library(stPS STATIC
    core/bitset.cpp
    core/csr.cpp
    core/inverted_index.cpp
    core/device_buffer.cu
    core/usc_solver.cu
    core/brute_force.cpp
    data/synthetic.cpp
    data/partition.cpp
)
add_library(stPS::stPS ALIAS stPS)

# usc_solver.cu / device_buffer.cu are compiled by nvcc — link the CUDA runtime
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

# Demo / CLI driver. main.cpp lives in src/, so its quote-include of the
# internal "data/synthetic.hpp" resolves relative to it; it drives the solve
# through the public stPS::Solver from <stPS/...> (carried by the link below).
add_executable(fullchipusc-solve main.cpp)
target_link_libraries(fullchipusc-solve PRIVATE stPS)
