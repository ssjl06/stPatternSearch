#pragma once

#include <stPS/types.hpp>

#include <cstdint>
#include <vector>

namespace stPS {

// Result of partitioning an M-patch list across ranks: this rank's patches plus
// the global PatchId each one carries. `coords`, when non-empty, mirrors
// `patches` shape exactly — coords[p][i] is where occurrence patches[p][i]
// sits. Empty coords = the source carried no locations (USC never needs them).
struct PatchSlice {
    std::vector<std::vector<Hash>>  patches;
    std::vector<PatchId>            global_ids;
    std::vector<std::vector<Point>> coords;
};

// The contiguous split rule used everywhere a list is divided across ranks
// (patches, file slices, output lines): rank r of `size` owns items
// [r*count/size, (r+1)*count/size). Throws std::invalid_argument unless
// 0 <= rank < size.
struct RankRange {
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};
RankRange rank_range(std::uint64_t count, int rank, int size);

// Partition an M-patch list contiguously across `size` ranks and return rank
// `rank`'s slice with its global patch IDs. Independent of where the patches
// came from. Rank r owns global IDs [r*M/size, (r+1)*M/size).
PatchSlice slice_patches_by_rank(std::vector<std::vector<Hash>> all_patches,
                                 int rank, int size);

// Same split, carrying per-occurrence coordinates alongside. Throws
// std::invalid_argument if coords does not have one entry per patch.
PatchSlice slice_patches_by_rank(std::vector<std::vector<Hash>> all_patches,
                                 std::vector<std::vector<Point>> coords,
                                 int rank, int size);

}  // namespace stPS
