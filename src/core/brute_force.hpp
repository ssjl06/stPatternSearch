#pragma once

#include "core/csr.hpp"
#include <stPS/usc_patch_selector.hpp>  // PatchSelection
#include <stPS/types.hpp>

#include <cstdint>

namespace stPS {

// Reference greedy set-cover implementation.
//
// Each iteration recomputes |patch[p] \ covered| for ALL patches and picks the
// argmax (tie: smaller PatchId, matching MPI_MAXLOC semantics). No inverted index,
// no incremental score maintenance. O(S · M · K) overall — slow but obviously
// correct. Used as ground truth in equivalence tests against the greedy selector.
PatchSelection brute_force_select(const PatchCsr& patches, std::uint64_t N);

}  // namespace stPS
