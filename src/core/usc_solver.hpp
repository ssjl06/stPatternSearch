#pragma once

#include "core/csr.hpp"
#include "core/device_buffer.hpp"
#include "core/inverted_index.hpp"
#include "core/types.hpp"

#include <cstdint>
#include <memory>
#include <vector>

// Forward-declared so solve() can hold an optional pointer to an NCCL
// communicator without dragging <nccl.h> into this public header. The
// concrete type is included in usc_solver.cu where it's actually used.
namespace stComm { class NCCLComm; }

namespace fullchipusc {

struct SolverResult {
    std::vector<PatchId> selected;       // chosen patch IDs in order (global, identical on every rank)
    std::uint64_t        covered_count;  // total elements covered
    std::uint64_t        iterations;     // number of iterations executed
};

// Greedy minimum set-cover solver, parameterized on the host-side comm
// backend used for setup() and for the 16B MAXLOC + 8B winner_global metadata
// each iteration. The per-iteration newly_covered_ids payload always travels
// device-direct over NCCL, so a NCCLComm is required.
//
// `CommT` is a duck-typed stComm-like comm object exposing:
//   - int  getRank() / getSize()
//   - allreduceMaxloc<T>(T) → std::pair<T,int>
//   - bcast<T>(T*, size_t, root) → RequestPtr (with .wait())
//   - allgatherv<T>(...)
//   - alltoallv<T>(...)
//   - exscan<T>(T, op)
//
// Only USCSolver<stComm::MPIComm> is instantiated (NCCL covers the device hot
// path via the injected nccl_comm_; the host backend stays MPI because NCCL
// lacks MAXLOC/EXSCAN and setup operates on host vectors).
template<typename CommT>
class USCSolver {
public:
    // `comm` is host-side MPI for setup + tiny metadata in solve().
    // `nccl_comm` is the GPU communicator for the per-iteration
    // newly_covered_ids broadcast; must be initialized (rank/device/uniqueId)
    // before construction. Both are required.
    USCSolver(CommT& comm, std::shared_ptr<stComm::NCCLComm> nccl_comm);

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
    std::shared_ptr<stComm::NCCLComm> nccl_comm_;      // required; hot-path device bcast

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
