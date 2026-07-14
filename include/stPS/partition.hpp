#pragma once

#include <stPS/types.hpp>

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

// Partition an M-patch list contiguously across `size` ranks and return rank
// `rank`'s slice with its global patch IDs. Independent of where the patches
// came from. Rank r owns global IDs [r*M/size, (r+1)*M/size).
PatchSlice slice_patches_by_rank(std::vector<std::vector<Hash>> all_patches,
                                 int rank, int size);

// Same split, carrying per-occurrence coordinates alongside (coords must match
// all_patches shape).
PatchSlice slice_patches_by_rank(std::vector<std::vector<Hash>> all_patches,
                                 std::vector<std::vector<Point>> coords,
                                 int rank, int size);

}  // namespace stPS
