#include "test_nccl_env.hpp"

#include <stComm/stComm.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

namespace {

// Non-owning pointer to the device-enabled Comm that lives on main's stack for
// the duration of RUN_ALL_TESTS (constructed after init, destroyed before
// finalize). Comm is non-movable, so it is held by reference, not by value.
stComm::Comm* g_comm = nullptr;

}  // namespace

stComm::Comm& fullchipusc::test_helpers::comm() {
    return *g_comm;
}

int main(int argc, char** argv) {
    stComm::Comm::initialize(&argc, &argv);

    int rc = 0;
    {
        // Probe rank/size on a host-only Comm, then guard on GPU availability.
        const int rank = stComm::Comm{}.getRank();
        const int size = stComm::Comm{}.getSize();

        int num_gpus = 0;
        cudaGetDeviceCount(&num_gpus);
        if (num_gpus < size) {
            if (rank == 0) {
                std::fprintf(stderr,
                    "fullchipusc_tests: requires >= %d GPUs but only %d visible. "
                    "Re-run with mpirun -n <=%d.\n", size, num_gpus, num_gpus);
            }
            stComm::Comm::finalize();
            return 2;
        }
        const int device_id = rank % num_gpus;
        cudaSetDevice(device_id);

        // onDevice bootstraps NCCL internally; this is the shared comm tests use.
        stComm::Comm comm = stComm::Comm::onDevice(device_id);
        g_comm = &comm;

        testing::InitGoogleTest(&argc, argv);
        rc = RUN_ALL_TESTS();

        g_comm = nullptr;  // comm destroyed at scope exit, before finalize
    }

    stComm::Comm::finalize();
    return rc;
}
