#pragma once

#include "core/csr.hpp"
#include "core/device_buffer.hpp"
#include "core/inverted_index.hpp"
#include "core/types.hpp"

#include <cstdint>
#include <vector>

namespace fullchipusc {

struct SolverResult {
    std::vector<PatchId> selected;       // chosen patch IDs in order (global, identical on every rank)
    std::uint64_t        covered_count;  // total elements covered
    std::uint64_t        iterations;     // number of iterations executed
};

// Greedy minimum set-cover solver, parameterized on the communication backend.
//
// `CommT` is a duck-typed stComm-like comm object exposing:
//   - int  getRank() / getSize()
//   - allreduceMaxloc<T>(T) → std::pair<T,int>
//   - bcast<T>(T*, size_t, root) → RequestPtr (with .wait())
//   - allgatherv<T>(...)
//
// For M2 the only instantiation is USCSolver<stComm::MPIComm> (CPU + MPI host
// memory). M5 will add USCSolver<stComm::NCCLComm> for GPU + NCCL device memory.
//
// Header carries declarations only; the implementation lives in usc_solver.cpp
// behind an explicit template instantiation.
template<typename CommT>
class USCSolver {
public:
    explicit USCSolver(CommT& comm);

    // Inject this rank's local patches plus their global patch IDs.
    void load(std::vector<std::vector<Hash>> raw_patches,
              std::vector<PatchId>           patch_global_ids);

    // Distributed setup: build local PatchCsr + InvertedIndex + global universe.
    void setup();

    // Multi-rank greedy main loop. All ranks return identical SolverResult.
    SolverResult solve();

    // Print result to stdout; only the root rank actually writes anything.
    void print_solution(const SolverResult& r) const;

    // Inspection (algorithm state only; no MPI state leaked).
    std::uint64_t        N() const;
    std::uint64_t        M_local() const;
    const PatchCsr&      patches() const;
    const InvertedIndex& inverted_index() const;

private:
    CommT& comm_;

    std::vector<std::vector<Hash>> raw_patches_;       // freed after setup()
    std::vector<PatchId>           patch_global_ids_;  // local idx → global PatchId

    PatchCsr      patches_;
    InvertedIndex inv_;
    std::uint64_t N_ = 0;
    std::vector<Hash> id_to_hash_;

    // Device-side mirrors of the algorithm state. Populated at the end of
    // setup() and consumed by GPU kernels in solve(). Empty until setup runs.
    DeviceBuffer<ElementId>     d_patch_data_;
    DeviceBuffer<std::uint64_t> d_patch_offsets_;
    DeviceBuffer<ElementId>     d_inv_keys_;
    DeviceBuffer<std::uint64_t> d_inv_offsets_;
    DeviceBuffer<PatchId>       d_inv_data_;
    DeviceBuffer<Score>         d_scores_;
    DeviceBuffer<std::uint64_t> d_covered_;         // N/64 words, zero-init
    DeviceBuffer<std::uint64_t> d_newly_covered_;   // N/64 words (scratch, M4+)
};

}  // namespace fullchipusc
