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
    test_hash_stats.cpp
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

# Tests run with one MPI rank per visible GPU (NCCL requires 1:1 process:device
# mapping). Override via -DSTPS_TEST_NRANKS=<n> at configure time when
# you know the host has n GPUs; the default of 2 matches our reference dual-A100
# setup. Single-component tests (bitset/csr/inv_index/...) ignore rank count;
# the solver equivalence / patch-set / hash-stats tests exercise the
# distributed path.
set(STPS_TEST_NRANKS 2 CACHE STRING "MPI ranks for ctest (= visible GPUs)")

# One ctest entry running the whole gtest suite under mpiexec. Deliberately NOT
# gtest_discover_tests: its EXEC_WRAPPER "argument" does not exist (CMake
# silently ignores unknown args), which had every test running single-process;
# and discovery itself can't be wrapped — `--gtest_list_tests` under
# `mpiexec -n 2` prints the list once per rank, corrupting the parse. One
# mpiexec launch for the suite also beats one per test (MPI+NCCL bootstrap is
# seconds per launch). gtest still reports per-test results in the output.
add_test(NAME stps_tests
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${STPS_TEST_NRANKS}
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:stps_tests>
)
