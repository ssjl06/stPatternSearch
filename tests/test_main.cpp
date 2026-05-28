#include "test_nccl_env.hpp"

#include <stComm/stComm.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <memory>

namespace {

std::shared_ptr<stComm::NCCLComm> g_nccl_comm;

}  // namespace

std::shared_ptr<stComm::NCCLComm> fullchipusc::test_helpers::nccl_comm() {
    return g_nccl_comm;
}

int main(int argc, char** argv) {
    stComm::MPIComm::initialize(&argc, &argv);

    int rc = 0;
    {
        stComm::MPIComm world;
        const int rank = world.getRank();
        const int size = world.getSize();

        int num_gpus = 0;
        cudaGetDeviceCount(&num_gpus);
        if (num_gpus < size) {
            if (rank == 0) {
                std::fprintf(stderr,
                    "fullchipusc_tests: requires >= %d GPUs but only %d visible. "
                    "Re-run with mpirun -n <=%d.\n", size, num_gpus, num_gpus);
            }
            stComm::MPIComm::finalize();
            return 2;
        }
        cudaSetDevice(rank % num_gpus);

        ncclUniqueId nccl_id;
        if (rank == 0) nccl_id = stComm::NCCLComm::getUniqueId();
        MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

        g_nccl_comm = std::make_shared<stComm::NCCLComm>();
        g_nccl_comm->initialize(rank, size, rank % num_gpus, nccl_id);

        testing::InitGoogleTest(&argc, argv);
        rc = RUN_ALL_TESTS();

        g_nccl_comm.reset();  // destroy before MPI finalize
    }

    stComm::MPIComm::finalize();
    return rc;
}
