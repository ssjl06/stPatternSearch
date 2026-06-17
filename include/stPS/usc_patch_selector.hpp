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

// Distributed greedy minimum-set-cover patch selector — the single entry point
// for the library. All algorithm internals (CSR, inverted index, device buffers,
// CUDA kernels) are hidden behind a pImpl, so this header stays dependency-free.
//
// GPU-only: `comm` must be device-enabled (stComm::Comm::onDevice). Construct
// once and reuse across patch_select() calls.
class UscPatchSelector {
public:
    explicit UscPatchSelector(stComm::Comm& comm);
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
