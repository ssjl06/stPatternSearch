add_library(fullchipusc_core STATIC
    core/bitset.cpp
    core/csr.cpp
    core/inverted_index.cpp
    core/usc_solver.cpp
    core/brute_force.cpp
    data/synthetic.cpp
)
target_include_directories(fullchipusc_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(fullchipusc_core PUBLIC MPI::MPI_CXX stComm::stComm)

add_executable(fullchipusc-solve main.cpp)
target_link_libraries(fullchipusc-solve PRIVATE fullchipusc_core)
