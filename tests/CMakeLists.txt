add_executable(fullchipusc_tests
    test_main.cpp
    test_bitset.cpp
    test_csr.cpp
    test_inverted_index.cpp
    test_synthetic.cpp
    test_brute_force.cpp
    test_solver_equivalence.cpp
    helpers/local_setup.cpp
)
# test_main.cpp uses cuda_runtime + NCCL via test_nccl_env.hpp.
find_package(CUDAToolkit REQUIRED)
target_link_libraries(fullchipusc_tests PRIVATE CUDA::cudart)
target_include_directories(fullchipusc_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(fullchipusc_tests PRIVATE
    fullchipusc_core
    GTest::gtest
)

include(GoogleTest)
# Tests run with one MPI rank per visible GPU (NCCL requires 1:1 process:device
# mapping). Override via -DFULLCHIPUSC_TEST_NRANKS=<n> at configure time when
# you know the host has n GPUs; the default of 2 matches our reference dual-A100
# setup. Single-component tests (bitset/csr/inv_index/...) ignore rank count;
# the solver equivalence test exercises the distributed path.
set(FULLCHIPUSC_TEST_NRANKS 2 CACHE STRING "MPI ranks for ctest (= visible GPUs)")
gtest_discover_tests(fullchipusc_tests
    EXEC_WRAPPER ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${FULLCHIPUSC_TEST_NRANKS} ${MPIEXEC_PREFLAGS}
)
