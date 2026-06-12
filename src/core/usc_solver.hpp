#pragma once

#include "core/csr.hpp"
#include "core/device_buffer.hpp"
#include "core/inverted_index.hpp"
#include "core/types.hpp"

#include <cstdint>
#include <memory>
#include <vector>

// Forward-declared so this public header can reference the unified comm by
// reference without dragging <mpi.h>/<nccl.h> in. The concrete type is included
// in usc_solver.cu where its members are actually called.
namespace stComm { class Comm; }

namespace fullchipusc {

struct SolverResult {
    std::vector<PatchId> selected;       // chosen patch IDs in order (global, identical on every rank)
    std::uint64_t        covered_count;  // total elements covered
    std::uint64_t        iterations;     // number of iterations executed
};

// Greedy minimum set-cover solver over a single unified stComm::Comm. The host
// (MPI) side drives setup() and the tiny per-iteration metadata (16B MAXLOC +
// 8B winner_global); the per-iteration newly_covered_ids payload travels
// device-direct over NCCL via the same Comm's Device space. The Comm must be
// device-enabled (built with Comm::onDevice) — fullchipUSC is GPU-only.
class USCSolver {
public:
    // `comm` must be a device-enabled Comm (Comm::onDevice): its Host space
    // backs setup() + solve() metadata, its Device space backs the hot-path
    // newly_covered_ids broadcast.
    explicit USCSolver(stComm::Comm& comm);

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
    stComm::Comm& comm_;   // device-enabled; Host space + Device space hot path

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
};

}  // namespace fullchipusc
