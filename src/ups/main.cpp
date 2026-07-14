// ups-hash-stats — distributed hash statistics over a PatchSet (UPS milestone
// 1): for every hash, how many patches contain it and where its representative
// occurrence sits (lexicographic-min (x,y)); the global top K land in a single
// text file that all ranks write in parallel.

#include "core/patch_set.hpp"    // internal: PatchSet (shared §5.1 setup)
#include "ups/hash_stats.hpp"    // internal: global_top_k / write_stats_file
#include "data/synthetic.hpp"    // internal demo data generator (this exe only)
#include "io/patch_reader.hpp"   // --input/--dump patch file support

#include <stPS/stPS.h>
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
#include <utility>

namespace {

using namespace stPS;

struct CliOptions {
    SyntheticParams params;
    std::string     input;   // read patches+coords from a .stps v2 file
    std::string     dump;    // write synthetic patches+coords to a .stps v2 file and exit
    std::string     output;  // stats file path (required unless --dump)
    std::uint64_t   output_limit = 100;
};

// Per-process GPU pick: rank-r uses GPU (r % visible_count) — same rule as
// usc-patch-select. PatchSet setup is GPU-only (device mirrors), but the
// stats path issues no device collectives, so no NCCL bootstrap: ranks may
// share a GPU, and the plain host Comm below suffices. Switch to
// Comm::onDevice once UPS grows device-side kernels that talk NCCL.
void pick_device_for_rank(int rank) {
    int num_gpus = 0;
    cudaError_t err = cudaGetDeviceCount(&num_gpus);
    if (err != cudaSuccess || num_gpus <= 0) {
        std::fprintf(stderr,
            "rank %d: no CUDA device available (%s). PatchSet setup is GPU-only.\n",
            rank, cudaGetErrorString(err));
        std::exit(2);
    }
    cudaSetDevice(rank % num_gpus);
}

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --output <path> [options]\n"
        "  --output <path>       stats file to write (required)\n"
        "  --output-limit <int>  top-K hashes to report (default %lu)\n"
        "  --N <int>             universe size hint (default %lu)\n"
        "  --M <int>             number of patches (default %lu)\n"
        "  --K <int>             average elements per patch (default %u)\n"
        "  --overlap <float>     patch overlap, 0..1 (default %.2f)\n"
        "  --seed <int>          RNG seed (default %lu)\n"
        "  --input <path>        read patches from a .stps v2 file (synthetic params\n"
        "                        ignored; needs coordinates — dump one with --dump)\n"
        "  --dump <path>         write the synthetic patches+coords to a .stps v2 file\n"
        "                        and exit (data-prep tool mode; no GPU needed)\n"
        "  -h, --help            show this help\n",
        prog,
        (unsigned long)CliOptions{}.output_limit,
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
        else if (a == "--input")  { const char* v = next_val(); if (!v) return false; opt.input  = v; }
        else if (a == "--dump")   { const char* v = next_val(); if (!v) return false; opt.dump   = v; }
        else if (a == "--output") { const char* v = next_val(); if (!v) return false; opt.output = v; }
        else if (a == "--output-limit") { const char* v = next_val(); if (!v) return false; opt.output_limit = std::strtoull(v, nullptr, 10); }
        else {
            std::fprintf(stderr, "Unknown argument: %s\n", a.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    if (!opt.input.empty() && !opt.dump.empty()) {
        std::fprintf(stderr, "--input and --dump are mutually exclusive\n");
        return false;
    }
    if (opt.dump.empty() && opt.output.empty()) {
        std::fprintf(stderr, "--output is required\n");
        print_usage(argv[0]);
        return false;
    }
    return true;
}

// This rank's contiguous patch range in file order — same split rule as
// slice_patches_by_rank. UPS needs locations, so a coordinate-less (v1) file
// is a usage error, not a silent (0,0) run.
stPS::PatchSlice read_input_slice(const std::string& path, int rank, int size) {
    auto reader = stPS::open_patch_file(path);
    const std::uint64_t M     = reader->patch_count();
    const std::uint64_t begin = (M * static_cast<std::uint64_t>(rank))     / size;
    const std::uint64_t end   = (M * (static_cast<std::uint64_t>(rank)+1)) / size;
    stPS::PatchSlice slice = reader->read_slice(begin, end);
    if (slice.coords.empty() && !slice.patches.empty()) {
        std::fprintf(stderr,
            "%s: no coordinates (v1 .stps?) — ups needs a v2 file; "
            "regenerate with ups-hash-stats --dump\n", path.c_str());
        std::exit(1);
    }
    return slice;
}

}  // namespace

int main(int argc, char** argv) {
    stComm::Comm::initialize(&argc, &argv);
    int exit_code = 0;
    try {
        CliOptions opt;
        if (!parse_args(argc, argv, opt)) { exit_code = 1; }
        else if (!opt.dump.empty()) {
            // Data-prep tool mode: write the synthetic set (v2, with coords)
            // and exit. Host-only — no GPU, no NCCL bootstrap.
            if (stComm::Comm{}.getRank() == 0) {
                const auto patches = generate_synthetic(opt.params);
                const auto coords  = generate_synthetic_coords(opt.params, patches);
                stPS::write_patch_file(opt.dump, patches, coords);
                std::cout << "dumped M=" << patches.size()
                          << " patches (+coords) to " << opt.dump << "\n";
            }
        }
        else {
            stComm::Comm comm;                       // host collectives only (see above)
            pick_device_for_rank(comm.getRank());    // device mirrors still need a GPU

            auto slice = !opt.input.empty()
                ? read_input_slice(opt.input, comm.getRank(), comm.getSize())
                : [&] {
                      auto patches = generate_synthetic(opt.params);
                      auto coords  = generate_synthetic_coords(opt.params, patches);
                      return stPS::slice_patches_by_rank(std::move(patches), std::move(coords),
                                                         comm.getRank(), comm.getSize());
                  }();

            auto t0 = std::chrono::steady_clock::now();
            // Collapse occurrences before PatchSet consumes the patches.
            const auto minloc = stPS::local_min_locations(slice.patches, slice.coords);
            stPS::PatchSet ps(comm, std::move(slice.patches));
            const auto topk = stPS::global_top_k(comm, ps, minloc, opt.output_limit);
            stPS::write_stats_file(comm, opt.output, topk);
            auto t1 = std::chrono::steady_clock::now();

            if (comm.getRank() == 0) {
                std::cout << "UPS hash-stats\n"
                          << "  ranks=" << comm.getSize()
                          << " unique_hashes=" << ps.N()
                          << " reported=" << topk.size()
                          << " (limit=" << opt.output_limit << ")\n"
                          << "  output: " << opt.output << "\n"
                          << "  timing (ms): stats="
                          << std::chrono::duration<double, std::milli>(t1 - t0).count()
                          << "\n";
            }
        }
    }
    catch (const std::exception& e) {
        // Same rationale as usc-patch-select: a backend fault mid-collective
        // leaves peers blocked — MPI_Abort tears every rank down at once.
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::fprintf(stderr, "rank %d: fatal error: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;  // unreachable: MPI_Abort does not return
    }
    stComm::Comm::finalize();
    return exit_code;
}
