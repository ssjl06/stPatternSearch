#pragma once

#include "core/csr.hpp"
#include "core/inverted_index.hpp"
#include <stPS/types.hpp>

#include <cstdint>
#include <vector>

namespace stPS::test_helpers {

// Single-process reference setup. NOT used by the production algorithm
// (UscPatchSelector), only by tests that want a full-universe PatchCsr to feed
// into `brute_force_select` for equivalence comparison.
//
// Performs: hash flatten → sort+unique → ID assignment → per-patch ID list
// → build PatchCsr → build InvertedIndex. No MPI calls.
struct LocalSetupResult {
    PatchCsr      patches;
    InvertedIndex inv;
    std::uint64_t N = 0;
    std::vector<Hash> id_to_hash;
};

LocalSetupResult run_local_setup(const std::vector<std::vector<Hash>>& raw_patches);

}  // namespace stPS::test_helpers
