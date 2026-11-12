// main() for stps_host_tests: the same gtest setup as test_main.cpp, but the
// shared Comm is host-only (MPI, no CUDA, no NCCL). Only test files that never
// reach a device path are linked into it, which is what lets it run at any
// rank count on a machine without GPUs.

#include "test_nccl_env.hpp"

#include <stComm/stComm.h>

#include <gtest/gtest.h>

namespace {

stComm::Comm* g_comm = nullptr;

}  // namespace

stComm::Comm& stPS::test_helpers::comm() {
    return *g_comm;
}

int main(int argc, char** argv) {
    stComm::Comm::initialize(&argc, &argv);

    int rc = 0;
    {
        stComm::Comm comm;
        g_comm = &comm;

        testing::InitGoogleTest(&argc, argv);
        rc = RUN_ALL_TESTS();

        g_comm = nullptr;
    }

    stComm::Comm::finalize();
    return rc;
}
