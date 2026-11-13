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

// Wire/score type of the ByElement mode. Scores count uncovered elements per
// patch, bounded by the patch size — comfortably int32 — and the per-iteration
// AllReduce ships M_global of them, so halving the element width halves the
// mode's dominant communication (and the ArgMax scan bandwidth with it).
// Setup PROVES the fit once: initial global scores are the solve's maxima
// (scores only decrease), so validating them ≤ INT32_MAX − P at setup covers
// every later value, including the disabled floor of −P − patch_size. The
// public Score (int64) and the ByPatch mode are unchanged.
using WireScore = std::int32_t;
inline constexpr WireScore kDisabledWireScore = -1;

// Internal implementation of the public UscPatchSelector: distributed greedy
// minimum set-cover. The shared preprocessing — distributed hash→id mapping,
// PatchCsr, InvertedIndex, and their device mirrors — is delegated to PatchSet;
// this class owns only the USC-specific state (per-patch scores + cumulative
// covered bitset) and the greedy main loop.
//
// The loop is device-resident: each iteration is two small NCCL reductions (the
// winner key, then the winner's newly covered elements) between kernels, all
// enqueued in batches; the host reads 24 B of loop state per batch. The Comm
// must be device-enabled (built with Comm::onDevice) — USC is GPU-only.
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

private:
    // Greedy main loop over the built patch_set_ + USC device state.
    PatchSelection select();

    stComm::Comm& comm_;   // device-enabled; Host space + Device space hot path

    std::unique_ptr<PatchSet> patch_set_;         // shared preprocessing result
    std::vector<PatchId>      patch_global_ids_;  // local idx → global PatchId

    // USC-specific device state. Populated at the start of patch_select() and
    // consumed by the greedy kernels. Empty until then.
    DeviceBuffer<WireScore>     d_scores_;     // per-patch remaining-coverage score
    DeviceBuffer<std::uint64_t> d_covered_;    // N/64 words, cumulative covered bitset
    DeviceBuffer<PatchId>       d_global_ids_; // patch_global_ids_ on the device

    std::uint64_t M_global_          = 0;      // patches over all ranks
    std::uint64_t max_initial_score_ = 0;      // largest patch over all ranks
};

namespace detail {
// Test-only switches for the ByElement loop. The delta exchange normally runs
// only with more than one rank; forcing it (and a tiny window) lets a
// single-GPU run exercise the multi-step draining path.
struct UscElemTestHooks {
    bool          force_delta_exchange = false;
    std::uint64_t delta_window         = 0;   // 0 = sized automatically
};
UscElemTestHooks& usc_elem_test_hooks();
}  // namespace detail

// Internal implementation of the public UscPatchSelector in
// PartitionMode::ByElement — design doc §7.3, the element-partitioned covered
// bitset. The element ID space [0, N) is statically sharded across ranks
// (contiguous ranges; sample sort already balances ID density). Each rank holds
// ONLY:
//   - its shard of the cumulative covered bitset (N/(64·P) words), and
//   - for EVERY global patch, the sub-list of its elements falling in this
//     shard (redistributed once at setup as (slot, element) incidences) plus
//     the matching shard-local inverted index.
//
// Every rank keeps the FULL score of every patch (identical everywhere), so
// each rank computes the same argmax locally: no MAXLOC, no winner-payload
// broadcast, and covered_count advances by the (globally known) winner score.
// What a selection changes travels as a "delta": the patch slots whose score
// drops, sent by the rank that owns the newly covered element, through a
// fixed-size window reduced with one small ncclAllReduce per step. Over a
// whole solve every incidence is sent once — communication is O(incidences),
// not O(iterations × M_global).
//
// The loop is DEVICE-RESIDENT: the window has a fixed size and no root, so the
// host enqueues whole batches of steps on the NCCL stream without knowing any
// outcome, and reads 24 B of loop state per batch.
class UscElemPatchSelectorImpl {
public:
    // `comm` must be a device-enabled Comm (Comm::onDevice).
    explicit UscElemPatchSelectorImpl(stComm::Comm& comm);

    // Same collective contract as UscPatchSelectorImpl::patch_select.
    PatchSelection patch_select(std::vector<std::vector<Hash>> raw_patches,
                                std::vector<PatchId>           global_ids);

private:
    // Greedy main loop over the element-sharded state built by patch_select().
    PatchSelection select();

    stComm::Comm& comm_;

    // Problem geometry (identical on every rank except the shard bounds).
    std::uint64_t N_            = 0;   // global universe size
    std::uint64_t M_global_     = 0;   // total patch count across ranks
    std::uint64_t shard_base_   = 0;   // first element ID owned by this rank
    std::uint64_t shard_extent_ = 0;   // number of element IDs owned

    // Host-side views. slot_to_gid_ maps global slot → caller-provided PatchId
    // (replicated on every rank; consumed once when draining the device-side
    // selected list into the result). max_sub_len_ bounds any winner's
    // shard-local sub-patch, sizing the newly-covered scratch once up front.
    std::vector<PatchId> slot_to_gid_;
    std::uint64_t        num_inv_keys_ = 0;  // shard-local inverted index (device only)
    std::uint64_t        max_sub_len_ = 0;
    // Largest initial global score — an upper bound on every winner's score,
    // used to size the first device-resident batch.
    std::uint64_t        max_initial_score_ = 0;
    // Average patches per element (rounded up) — sizes the delta window.
    std::uint64_t        avg_degree_ = 1;

    // Device state. d_patch_data_ stores SHARD-LOCAL element IDs (global ID −
    // shard_base_), so the covered bitset and the kernels index the local shard
    // exactly like the ByPatch mode indexes the full universe. d_sub_offsets_
    // is the sub-CSR offsets mirror — the commit kernel resolves the winner's
    // range on device, so the host never needs it during the loop.
    DeviceBuffer<ElementId>     d_patch_data_;      // sub-CSR data (local IDs)
    DeviceBuffer<std::uint64_t> d_sub_offsets_;     // M_global+1 offsets
    DeviceBuffer<ElementId>     d_inv_keys_;        // shard-local inverted index
    DeviceBuffer<std::uint64_t> d_inv_offsets_;
    DeviceBuffer<PatchId>       d_inv_data_;        // values are global slots
    DeviceBuffer<WireScore>     d_scores_;          // M_global; full scores (replicated)
    DeviceBuffer<std::uint64_t> d_covered_;         // shard_extent_/64 words
};

}  // namespace stPS
