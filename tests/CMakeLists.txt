add_executable(stps_tests
    test_main.cpp
    test_bitset.cpp
    test_csr.cpp
    test_inverted_index.cpp
    test_synthetic.cpp
    test_patch_set.cpp
    test_brute_force.cpp
    test_solver_equivalence.cpp
    test_patch_io.cpp
    helpers/local_setup.cpp
)
# test_main.cpp uses cuda_runtime + NCCL via test_nccl_env.hpp.
find_package(CUDAToolkit REQUIRED)
target_link_libraries(stps_tests PRIVATE CUDA::cudart)
target_include_directories(stps_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}     # tests/  (test_nccl_env.hpp, helpers/)
    ${CMAKE_SOURCE_DIR}/src         # white-box access to internal core/ + data/
)
target_link_libraries(stps_tests PRIVATE
    stPS
    GTest::gtest
)

include(GoogleTest)
# Tests run with one MPI rank per visible GPU (NCCL requires 1:1 process:device
# mapping). Override via -DSTPS_TEST_NRANKS=<n> at configure time when
# you know the host has n GPUs; the default of 2 matches our reference dual-A100
# setup. Single-component tests (bitset/csr/inv_index/...) ignore rank count;
# the solver equivalence test exercises the distributed path.
set(STPS_TEST_NRANKS 2 CACHE STRING "MPI ranks for ctest (= visible GPUs)")
# PRE_TEST defers test discovery from build time to the first ctest run: the
# discovery step executes the test binary, which exits(2) on GPU-less hosts
# (test_main guards num_gpus >= world size) — POST_BUILD discovery would make
# `cmake --build` itself fail on such hosts (e.g. the CI compile-only runner).
gtest_discover_tests(stps_tests
    EXEC_WRAPPER ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${STPS_TEST_NRANKS} ${MPIEXEC_PREFLAGS}
    DISCOVERY_MODE PRE_TEST
)
