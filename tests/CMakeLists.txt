add_executable(stps_tests
    test_main.cpp
    test_bitset.cpp
    test_csr.cpp
    test_inverted_index.cpp
    test_hash_id_map.cpp
    test_synthetic.cpp
    test_patch_set.cpp
    test_brute_force.cpp
    test_solver_equivalence.cpp
    test_patch_io.cpp
    test_pattern_stats.cpp
    test_cli_support.cpp
    test_collective.cpp
    test_unique_hashes.cpp
    helpers/lazy_greedy.cpp
    helpers/local_setup.cpp
    ${CMAKE_SOURCE_DIR}/src/core/cli_support.cpp   # app-level, not in libstPS
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
enable_compiler_warnings(stps_tests)

# Host-only subset: the tests that never reach a device path, on a host-only
# Comm. No GPU needed at run time, so this runs at any rank count — including
# on GPU-less CI runners, where it is the only distributed test that executes.
add_executable(stps_host_tests
    host_test_main.cpp
    test_collective.cpp
    test_unique_hashes.cpp
)
target_include_directories(stps_host_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(stps_host_tests PRIVATE
    stPS
    GTest::gtest
)
enable_compiler_warnings(stps_host_tests)

# Rank counts to run each suite at. Every distributed test compares against a
# single-process reference that does not depend on the rank count, so running
# the same suite at several counts is the rank-invariance check.
#
# GPU suite: one MPI rank per visible GPU (NCCL requires a 1:1 process:device
# mapping). A count above the number of visible GPUs is reported as skipped,
# not failed, so the default list is safe on any box; add 4;8 for multi-node
# runs (pass the host list through MPIEXEC_PREFLAGS).
set(STPS_TEST_NRANKS "1;2" CACHE STRING "MPI rank counts for the GPU suite (each <= visible GPUs)")
# Host suite: no such limit — ranks may outnumber cores.
set(STPS_HOST_TEST_NRANKS "1;2;4;8" CACHE STRING "MPI rank counts for the host-only suite")

# One ctest entry per suite and rank count, each running the whole gtest
# binary under mpiexec. Deliberately NOT gtest_discover_tests: its EXEC_WRAPPER
# "argument" does not exist (CMake silently ignores unknown args), which had
# every test running single-process; and discovery itself can't be wrapped —
# `--gtest_list_tests` under `mpiexec -n 2` prints the list once per rank,
# corrupting the parse. One mpiexec launch for the suite also beats one per
# test (MPI+NCCL bootstrap is seconds per launch). gtest still reports
# per-test results in the output.
foreach(nranks IN LISTS STPS_TEST_NRANKS)
    add_test(NAME stps_tests_np${nranks}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${nranks}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:stps_tests>
    )
    # test_main.cpp exits 77 when there are fewer GPUs than ranks. The GPU
    # entries share the devices, so they must not overlap each other.
    set_tests_properties(stps_tests_np${nranks} PROPERTIES
        SKIP_RETURN_CODE 77
        RESOURCE_LOCK gpu
    )
endforeach()

foreach(nranks IN LISTS STPS_HOST_TEST_NRANKS)
    add_test(NAME stps_host_tests_np${nranks}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${nranks}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:stps_host_tests>
    )
    # Allow more ranks than cores. Open MPI 4 and 5 spell it differently;
    # other MPI implementations ignore both variables.
    set_tests_properties(stps_host_tests_np${nranks} PROPERTIES
        ENVIRONMENT "OMPI_MCA_rmaps_base_oversubscribe=1;PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe"
    )
endforeach()
