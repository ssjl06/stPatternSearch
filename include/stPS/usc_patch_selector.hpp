#pragma once

#include <stPS/types.hpp>

#include <cstdint>
#include <memory>
#include <vector>

// Forward-declared so this public header pulls in no MPI/NCCL/CUDA headers.
// A device-enabled stComm::Comm (built with stComm::Comm::onDevice) drives the
// distributed patch selection.
namespace stComm { class Comm; }

namespace stPS {

// Result of a distributed greedy minimum-set-cover patch selection. Identical on
// every rank.
struct PatchSelection {
    std::vector<PatchId> selected;       // chosen patch IDs, in selection order
    std::uint64_t        covered_count;  // total elements covered
    std::uint64_t        iterations;     // iterations executed
};

// Distribution strategy for the solve (design doc §5.2 / §7.3).
//
//   ByPatch   — patches stay on their input rank; every rank replicates the
//               full covered bitset (N bits). Per-iteration communication is a
//               MAXLOC + the winner's newly-covered ID list (M5 baseline).
//               Right default while covered (N/8 bytes) fits GPU memory.
//
//   ByElement — the element ID space [0, N) is sharded across ranks; each rank
//               holds only its shard of covered plus every patch's sub-list
//               falling in that shard (redistributed once at setup). Scores are
//               per-rank partials; one fixed-size AllReduce<SUM> of M scores per
//               iteration re-forms full scores everywhere. Communication scales
//               with M instead of N — the memory/scaling mode for huge N (§7.3).
enum class PartitionMode {
    ByPatch,
    ByElement,
};

// Distributed greedy minimum-set-cover patch selector — the single entry point
// for the library. All algorithm internals (CSR, inverted index, device buffers,
// CUDA kernels) are hidden behind a pImpl, so this header stays dependency-free.
//
// GPU-only: `comm` must be device-enabled (stComm::Comm::onDevice). Construct
// once and reuse across patch_select() calls. `mode` picks the distribution
// strategy (see PartitionMode); the input contract and the result are identical
// in both modes.
class UscPatchSelector {
public:
    explicit UscPatchSelector(stComm::Comm& comm,
                              PartitionMode mode = PartitionMode::ByPatch);
    ~UscPatchSelector();

    UscPatchSelector(UscPatchSelector&&) noexcept;
    UscPatchSelector& operator=(UscPatchSelector&&) noexcept;
    UscPatchSelector(const UscPatchSelector&)            = delete;
    UscPatchSelector& operator=(const UscPatchSelector&) = delete;

    // Run the full pipeline (load → distributed setup → greedy select) on this
    // rank's `patches` and their `global_ids`. **Collective** — every rank must
    // call together; all ranks return the same PatchSelection.
    PatchSelection patch_select(std::vector<std::vector<Hash>> patches,
                                std::vector<PatchId>           global_ids);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace stPS
