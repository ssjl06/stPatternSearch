#include <stComm/stComm.h>

#include <gtest/gtest.h>

int main(int argc, char** argv) {
    stComm::MPIComm::initialize(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    stComm::MPIComm::finalize();
    return rc;
}
