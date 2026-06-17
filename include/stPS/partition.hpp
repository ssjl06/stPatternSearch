#pragma once

#include <stPS/types.hpp>

#include <vector>

namespace stPS {

// Result of partitioning an M-patch list across ranks: this rank's patches plus
// the global PatchId each one carries.
struct PatchSlice {
    std::vector<std::vector<Hash>> patches;
    std::vector<PatchId>           global_ids;
};

// Partition an M-patch list contiguously across `size` ranks and return rank
// `rank`'s slice with its global patch IDs. Independent of where the patches
// came from. Rank r owns global IDs [r*M/size, (r+1)*M/size).
PatchSlice slice_patches_by_rank(std::vector<std::vector<Hash>> all_patches,
                                 int rank, int size);

}  // namespace stPS
