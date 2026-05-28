add_library(fullchipusc_core STATIC
    core/bitset.cpp
    core/csr.cpp
    core/inverted_index.cpp
    core/device_buffer.cu
    core/usc_solver.cu
    core/brute_force.cpp
    data/synthetic.cpp
)
target_include_directories(fullchipusc_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# usc_solver.cu is compiled by nvcc — link the CUDA runtime so DeviceBuffer's
# cudaMalloc/cudaFree resolve. Position-independent code lets the static lib
# be linked into mixed host/device targets without surprises.
set_target_properties(fullchipusc_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(fullchipusc_core PUBLIC
    MPI::MPI_CXX
    stComm::stComm
    CUDA::cudart
)

add_executable(fullchipusc-solve main.cpp)
target_link_libraries(fullchipusc-solve PRIVATE fullchipusc_core)
