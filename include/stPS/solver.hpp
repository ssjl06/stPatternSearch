#pragma once

#include <stPS/types.hpp>

#include <cstdint>
#include <memory>
#include <vector>

// Forward-declared so this public header pulls in no MPI/NCCL/CUDA headers.
// A device-enabled stComm::Comm (built with stComm::Comm::onDevice) drives the
// distributed solve.
namespace stComm { class Comm; }

namespace stPS {

// Result of a distributed greedy minimum-set-cover solve. Identical on every
// rank.
struct SolverResult {
    std::vector<PatchId> selected;       // chosen patch IDs, in selection order
    std::uint64_t        covered_count;  // total elements covered
    std::uint64_t        iterations;     // iterations executed
};

// Distributed greedy minimum-set-cover solver — the single entry point for the
// library. All algorithm internals (CSR, inverted index, device buffers, CUDA
// kernels) are hidden behind a pImpl, so this header stays dependency-free.
//
// GPU-only: `comm` must be device-enabled (stComm::Comm::onDevice). Construct
// once and reuse across run() calls.
class Solver {
public:
    explicit Solver(stComm::Comm& comm);
    ~Solver();

    Solver(Solver&&) noexcept;
    Solver& operator=(Solver&&) noexcept;
    Solver(const Solver&)            = delete;
    Solver& operator=(const Solver&) = delete;

    // Run the full pipeline (load → distributed setup → greedy solve) on this
    // rank's `patches` and their `global_ids`. **Collective** — every rank must
    // call together; all ranks return the same SolverResult.
    SolverResult run(std::vector<std::vector<Hash>> patches,
                     std::vector<PatchId>           global_ids);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace stPS
