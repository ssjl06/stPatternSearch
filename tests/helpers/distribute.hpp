#pragma once

#include <stPS/partition.hpp>
#include <stPS/types.hpp>

#include <cstddef>
#include <vector>

namespace stPS::test_helpers {

// Uneven ways to hand a patch list to the ranks, for the cases the contiguous
// slice_patches_by_rank never produces. `coords` may be empty (USC has none).

// Everything on `owner`; every other rank holds nothing at all.
inline PatchSlice slice_all_on_rank(const std::vector<std::vector<Hash>>&  patches,
                                    const std::vector<std::vector<Point>>& coords,
                                    int owner, int rank) {
    PatchSlice out;
    if (rank != owner) return out;
    out.patches = patches;
    out.coords  = coords;
    out.global_ids.reserve(patches.size());
    for (std::size_t p = 0; p < patches.size(); ++p) out.global_ids.push_back(p);
    return out;
}

// Patch p goes to rank p % size: each rank's global IDs are interleaved with
// every other rank's instead of forming one contiguous block.
inline PatchSlice slice_round_robin(const std::vector<std::vector<Hash>>&  patches,
                                    const std::vector<std::vector<Point>>& coords,
                                    int rank, int size) {
    PatchSlice out;
    for (std::size_t p = static_cast<std::size_t>(rank); p < patches.size();
         p += static_cast<std::size_t>(size)) {
        out.patches.push_back(patches[p]);
        out.global_ids.push_back(p);
        if (!coords.empty()) out.coords.push_back(coords[p]);
    }
    return out;
}

}  // namespace stPS::test_helpers
