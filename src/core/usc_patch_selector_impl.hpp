#pragma once

#include "core/csr.hpp"
#include "core/device_buffer.hpp"
#include "core/inverted_index.hpp"
#include <stPS/usc_patch_selector.hpp>   // PatchSelection + the public UscPatchSelector this backs
#include <stPS/types.hpp>

#include <cstdint>
#include <memory>
#include <vector>

// Forward-declared so this internal header drags in no <mpi.h>/<nccl.h>. The
// concrete type is included in usc_patch_selector_impl.cu where its members are called.
namespace stComm { class Comm; }

namespace stPS {

// Internal implementation of the public UscPatchSelector: greedy minimum
// set-cover over a single unified stComm::Comm. The host
// (MPI) side drives setup() and the tiny per-iteration metadata (16B MAXLOC +
// 8B winner_global); the per-iteration newly_covered_ids payload travels
// device-direct over NCCL via the same Comm's Device space. The Comm must be
// device-enabled (built with Comm::onDevice) — fullchipUSC is GPU-only.
class UscPatchSelectorImpl {
public:
    // `comm` must be a device-enabled Comm (Comm::onDevice): its Host space
    // backs setup() + select() metadata, its Device space backs the hot-path
    // newly_covered_ids broadcast.
    explicit UscPatchSelectorImpl(stComm::Comm& comm);

    // Inject this rank's local patches plus their global patch IDs.
    void load(std::vector<std::vector<Hash>> raw_patches,
              std::vector<PatchId>           patch_global_ids);

    // Distributed setup: build local PatchCsr + InvertedIndex + global universe.
    void setup();

    // Multi-rank greedy main loop. All ranks return identical PatchSelection.
    PatchSelection select();

    // Print result to stdout; only the root rank actually writes anything.
    void print_selection(const PatchSelection& r) const;

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

}  // namespace stPS
