#pragma once

#include "core/device_buffer.hpp"
#include "core/patch_set.hpp"
#include <stPS/usc_patch_selector.hpp>   // PatchSelection + the public UscPatchSelector this backs
#include <stPS/types.hpp>

#include <cstdint>
#include <memory>
#include <vector>

// Forward-declared so this internal header drags in no <mpi.h>/<nccl.h>. The
// concrete type is included in usc_patch_selector_impl.cu where its members are called.
namespace stComm { class Comm; }

namespace stPS {

// Internal implementation of the public UscPatchSelector: distributed greedy
// minimum set-cover. The shared preprocessing — distributed hash→id mapping,
// PatchCsr, InvertedIndex, and their device mirrors — is delegated to PatchSet;
// this class owns only the USC-specific state (per-patch scores + cumulative
// covered bitset) and the greedy main loop.
//
// The host (MPI) side drives the tiny per-iteration metadata (16B MAXLOC + 8B
// winner_global); the per-iteration newly_covered_ids payload travels
// device-direct over NCCL via the same Comm's Device space. The Comm must be
// device-enabled (built with Comm::onDevice) — USC is GPU-only.
class UscPatchSelectorImpl {
public:
    // `comm` must be a device-enabled Comm (Comm::onDevice).
    explicit UscPatchSelectorImpl(stComm::Comm& comm);

    // Full pipeline: build the shared PatchSet from this rank's `raw_patches`,
    // then run the distributed greedy select. `global_ids` gives the global
    // PatchId of each local patch (index-aligned to raw_patches). **Collective** —
    // every rank must call together; all ranks return the same PatchSelection.
    PatchSelection patch_select(std::vector<std::vector<Hash>> raw_patches,
                                std::vector<PatchId>           global_ids);

    // Print result to stdout; only the root rank actually writes anything.
    void print_selection(const PatchSelection& r) const;

    // Inspection (algorithm state only; no MPI state leaked). Valid after
    // patch_select() has built the PatchSet.
    std::uint64_t        N() const;
    std::uint64_t        M_local() const;
    const PatchCsr&      patches() const;
    const InvertedIndex& inverted_index() const;

private:
    // Greedy main loop over the built patch_set_ + USC device state.
    PatchSelection select();

    stComm::Comm& comm_;   // device-enabled; Host space + Device space hot path

    std::unique_ptr<PatchSet> patch_set_;         // shared preprocessing result
    std::vector<PatchId>      patch_global_ids_;  // local idx → global PatchId

    // USC-specific device state. Populated at the start of patch_select() and
    // consumed by the greedy kernels. Empty until then.
    DeviceBuffer<Score>         d_scores_;     // per-patch remaining-coverage score
    DeviceBuffer<std::uint64_t> d_covered_;    // N/64 words, cumulative covered bitset
};

}  // namespace stPS
