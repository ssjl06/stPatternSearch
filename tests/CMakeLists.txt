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
target_include_directories(fullchipusc_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(fullchipusc_tests PRIVATE
    fullchipusc_core
    GTest::gtest
)

include(GoogleTest)
# Tests run under mpirun -n 4 to exercise multi-rank paths. Single-component
# tests (bitset/csr/inv_index/...) are oblivious to rank count; the solver
# equivalence test specifically exercises the distributed path.
gtest_discover_tests(fullchipusc_tests
    EXEC_WRAPPER ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
)
