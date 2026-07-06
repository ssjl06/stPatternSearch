#include <stPS/stPS.h>          // public library API (UscPatchSelector, partition, types)
#include "data/synthetic.hpp"   // internal demo data generator (this exe only)
#include "io/patch_reader.hpp"  // --input/--dump patch file support (M7)

#include <stComm/stComm.h>

#include <cuda_runtime.h>
#include <mpi.h>          // top-level error path only: MPI_Abort / rank for logging

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

namespace {

using namespace stPS;

struct CliOptions {
    SyntheticParams params;
    PartitionMode   mode = PartitionMode::ByPatch;
    bool print_solution = false;
};

// Per-process GPU pick: rank-r uses GPU (r % visible_count). Sets the active
// device for this rank's own CUDA allocations and returns the device id to hand
// to Comm::onDevice (which bootstraps NCCL on it). Exits if no GPU is visible.
int pick_device_for_rank(int rank) {
    int num_gpus = 0;
    cudaError_t err = cudaGetDeviceCount(&num_gpus);
    if (err != cudaSuccess || num_gpus <= 0) {
        std::fprintf(stderr,
            "rank %d: no CUDA device available (%s). USC is GPU-only.\n",
            rank, cudaGetErrorString(err));
        std::exit(2);
    }
    const int device_id = rank % num_gpus;
    cudaSetDevice(device_id);
    return device_id;
}

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --N <int>           universe size hint (default %lu)\n"
        "  --M <int>           number of patches (default %lu)\n"
        "  --K <int>           average elements per patch (default %u)\n"
        "  --overlap <float>   patch overlap, 0..1 (default %.2f)\n"
        "  --seed <int>        RNG seed (default %lu)\n"
        "  --partition <mode>  patch | element (default patch; element = §7.3\n"
        "                      element-sharded covered bitset, comm scales with M not N)\n"
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
        else if (a == "--partition") {
            const char* v = next_val(); if (!v) return false;
            const std::string m = v;
            if      (m == "patch")   opt.mode = PartitionMode::ByPatch;
            else if (m == "element") opt.mode = PartitionMode::ByElement;
            else { std::fprintf(stderr, "Unknown --partition mode: %s\n", v); return false; }
        }
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
    stComm::Comm::initialize(&argc, &argv);
    int exit_code = 0;
    try {
        CliOptions opt;
        if (!parse_args(argc, argv, opt)) { exit_code = 1; }
        else {
            const int rank = stComm::Comm{}.getRank();      // host probe for device pick
            const int device_id = pick_device_for_rank(rank);
            // onDevice bootstraps NCCL internally (uniqueId handshake + init).
            stComm::Comm comm = stComm::Comm::onDevice(device_id);

            // The whole library API: build a UscPatchSelector on the comm, hand
            // it this rank's patches + their global IDs, get the cover back.
            stPS::UscPatchSelector selector(comm, opt.mode);
            auto slice = stPS::slice_patches_by_rank(
                generate_synthetic(opt.params), comm.getRank(), comm.getSize());
            const std::size_t m_local = slice.patches.size();  // capture before move

            auto t0 = std::chrono::steady_clock::now();
            const auto result =
                selector.patch_select(std::move(slice.patches), std::move(slice.global_ids));
            auto t1 = std::chrono::steady_clock::now();

            if (comm.getRank() == 0) {
                std::cout << "USC patch-select\n"
                          << "  ranks=" << comm.getSize()
                          << " partition=" << (opt.mode == stPS::PartitionMode::ByElement
                                               ? "element" : "patch") << "\n"
                          << "  result: selected=" << result.selected.size()
                          << " covered=" << result.covered_count
                          << " iterations=" << result.iterations << "\n"
                          << "  M_local(rank0)=" << m_local << "\n"
                          << "  timing (ms): patch-select="
                          << std::chrono::duration<double, std::milli>(t1 - t0).count()
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
    catch (const std::exception& e) {
        // stComm now throws on a backend (CUDA/NCCL/MPI) failure. Such a fault
        // typically hits one rank mid-collective, leaving the others blocked, so
        // falling through to finalize() would deadlock the job. Log with the
        // rank for context and MPI_Abort to tear every rank down at once.
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::fprintf(stderr, "rank %d: fatal error: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;  // unreachable: MPI_Abort does not return
    }
    stComm::Comm::finalize();
    return exit_code;
}
