#include "core/cli_support.hpp"

#include <cuda_runtime.h>
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <system_error>

namespace stPS::cli {

namespace {

int world_rank_or_minus_one() {
    int inited = 0;
    MPI_Initialized(&inited);
    int rank = -1;
    if (inited) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

[[noreturn]] void abort_job(int code) {
    std::fflush(stderr);
    int inited = 0;
    MPI_Initialized(&inited);
    if (inited) MPI_Abort(MPI_COMM_WORLD, code);
    std::abort();
}

[[noreturn]] void abort_on_terminate() {
    const int rank = world_rank_or_minus_one();
    if (const std::exception_ptr ep = std::current_exception()) {
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "rank %d: fatal error: %s\n", rank, e.what());
        } catch (...) {
            std::fprintf(stderr, "rank %d: fatal error: unknown exception\n", rank);
        }
    } else {
        std::fprintf(stderr, "rank %d: std::terminate called\n", rank);
    }
    abort_job(1);
}

}  // namespace

void install_mpi_abort_on_uncaught() {
    std::set_terminate(abort_on_terminate);
}

int pick_device_for_local_rank(const char* app_name) {
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    int local_rank = 0;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);
    const int rank = world_rank_or_minus_one();

    int num_gpus = 0;
    cudaError_t err = cudaGetDeviceCount(&num_gpus);
    if (err != cudaSuccess || num_gpus <= 0) {
        std::fprintf(stderr, "rank %d: no CUDA device available (%s). %s is GPU-only.\n",
                     rank, cudaGetErrorString(err), app_name);
        abort_job(2);
    }
    const int device_id = local_rank % num_gpus;
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "rank %d: cudaSetDevice(%d) failed: %s\n",
                     rank, device_id, cudaGetErrorString(err));
        abort_job(2);
    }
    return device_id;
}

bool same_output_path(const std::string& a, const std::string& b) {
    // absolute() first: weakly_canonical leaves a relative path whose first
    // component doesn't exist untouched, so "x" and "./x" would differ.
    const auto norm = [](const std::string& p, std::error_code& ec) {
        const auto abs = std::filesystem::absolute(p, ec);
        return ec ? abs : std::filesystem::weakly_canonical(abs, ec);
    };
    std::error_code ea, eb;
    const auto ca = norm(a, ea);
    const auto cb = norm(b, eb);
    if (ea || eb) return a == b;  // can't normalize: fall back to the literal paths
    return ca == cb;
}

}  // namespace stPS::cli
