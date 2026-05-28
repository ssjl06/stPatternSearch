#include "core/usc_solver.hpp"
#include "data/synthetic.hpp"

#include <stComm/stComm.h>

#include <cuda_runtime.h>
#include <mpi.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

namespace {

using namespace fullchipusc;

struct CliOptions {
    SyntheticParams params;
    bool print_solution = false;
};

// Per-process GPU pick. Rank-r picks GPU (r % visible_count). Caller must
// invoke this *before* the NCCL communicator is initialized — NCCL records
// the active device at ncclCommInitRank.
void pick_device_for_rank(int rank) {
    int num_gpus = 0;
    cudaError_t err = cudaGetDeviceCount(&num_gpus);
    if (err != cudaSuccess || num_gpus <= 0) {
        std::fprintf(stderr,
            "rank %d: no CUDA device available (%s). fullchipUSC is GPU-only.\n",
            rank, cudaGetErrorString(err));
        std::exit(2);
    }
    const int device_id = rank % num_gpus;
    cudaSetDevice(device_id);
}

// Bootstrap a NCCLComm shared by all ranks. Rank 0 mints the uniqueId and
// MPI-broadcasts it to everyone, then each rank calls ncclCommInitRank on
// its picked device. Returns a shared_ptr the solver takes co-ownership of.
std::shared_ptr<stComm::NCCLComm> make_nccl_comm(const stComm::MPIComm& world) {
    ncclUniqueId nccl_id;
    if (world.getRank() == 0) {
        nccl_id = stComm::NCCLComm::getUniqueId();
    }
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    int device_id = 0;
    cudaGetDevice(&device_id);  // pick_device_for_rank already set this
    auto nc = std::make_shared<stComm::NCCLComm>();
    nc->initialize(world.getRank(), world.getSize(), device_id, nccl_id);
    return nc;
}

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --N <int>           universe size hint (default %lu)\n"
        "  --M <int>           number of patches (default %lu)\n"
        "  --K <int>           average elements per patch (default %u)\n"
        "  --overlap <float>   patch overlap, 0..1 (default %.2f)\n"
        "  --seed <int>        RNG seed (default %lu)\n"
        "  --print-solution    print the selected patch IDs\n"
        "  -h, --help          show this help\n",
        prog,
        (unsigned long)CliOptions{}.params.N,
        (unsigned long)CliOptions{}.params.M,
        (unsigned)     CliOptions{}.params.K_mean,
                       CliOptions{}.params.overlap,
        (unsigned long)CliOptions{}.params.seed);
}

bool parse_args(int argc, char** argv, CliOptions& opt) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next_val = [&]() -> const char* {
            if (i + 1 >= argc) return nullptr;
            return argv[++i];
        };
        if      (a == "-h" || a == "--help") { print_usage(argv[0]); std::exit(0); }
        else if (a == "--N")        { const char* v = next_val(); if (!v) return false; opt.params.N      = std::strtoull(v, nullptr, 10); }
        else if (a == "--M")        { const char* v = next_val(); if (!v) return false; opt.params.M      = std::strtoull(v, nullptr, 10); }
        else if (a == "--K")        { const char* v = next_val(); if (!v) return false; opt.params.K_mean = static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10)); }
        else if (a == "--overlap")  { const char* v = next_val(); if (!v) return false; opt.params.overlap = std::strtod(v, nullptr); }
        else if (a == "--seed")     { const char* v = next_val(); if (!v) return false; opt.params.seed   = std::strtoull(v, nullptr, 10); }
        else if (a == "--print-solution") { opt.print_solution = true; }
        else {
            std::fprintf(stderr, "Unknown argument: %s\n", a.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    stComm::MPIComm::initialize(&argc, &argv);
    int exit_code = 0;
    {
        CliOptions opt;
        if (!parse_args(argc, argv, opt)) { exit_code = 1; }
        else {
            stComm::MPIComm world;
            pick_device_for_rank(world.getRank());
            auto nccl_comm = make_nccl_comm(world);
            USCSolver<stComm::MPIComm> solver(world, std::move(nccl_comm));

            auto t0 = std::chrono::steady_clock::now();
            auto slice = slice_patches_by_rank(
                generate_synthetic(opt.params), world.getRank(), world.getSize());
            solver.load(std::move(slice.patches), std::move(slice.global_ids));
            auto t1 = std::chrono::steady_clock::now();
            solver.setup();
            auto t2 = std::chrono::steady_clock::now();
            const auto result = solver.solve();
            auto t3 = std::chrono::steady_clock::now();

            solver.print_solution(result);
            if (world.getRank() == 0) {
                std::cout << "  M_local(rank0)=" << solver.M_local() << "\n";
                std::cout << "  timing (ms): load="
                          << std::chrono::duration<double, std::milli>(t1 - t0).count()
                          << " setup="
                          << std::chrono::duration<double, std::milli>(t2 - t1).count()
                          << " solve="
                          << std::chrono::duration<double, std::milli>(t3 - t2).count()
                          << "\n";
                if (opt.print_solution) {
                    std::cout << "  selected (" << result.selected.size() << "):";
                    const std::size_t n = std::min<std::size_t>(16, result.selected.size());
                    for (std::size_t i = 0; i < n; ++i) std::cout << ' ' << result.selected[i];
                    if (n < result.selected.size()) std::cout << " ...";
                    std::cout << "\n";
                }
            }
        }
    }
    stComm::MPIComm::finalize();
    return exit_code;
}
