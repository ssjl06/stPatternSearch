#pragma once

#include "core/types.hpp"

#include <cstdint>
#include <vector>

namespace stPS {

struct SyntheticParams {
    std::uint64_t N            = 1000;   // target universe size (distinct hash count)
    std::uint64_t M            = 100;    // number of patches
    std::uint32_t K_mean       = 30;     // average elements per patch
    double        overlap      = 0.3;    // 0 = patches barely share; 1 = patches mostly share
    std::uint64_t seed         = 1;
};

// Deterministic synthetic patch generator.
//
// Conceptually: generate N unique uint64 hashes (the "hash pool"), then for each
// patch sample K elements with cluster-correlated bias so that `overlap` controls
// how much patches share hashes.
//
//   overlap = 0  → each patch's elements are independent uniform samples
//   overlap = 1  → all patches draw from a single narrow cluster (max sharing)
//
// Returns one vector per patch, each holding raw uint64 hash values (not yet
// converted to IDs). Patch sizes vary mildly around K_mean.
std::vector<std::vector<Hash>> generate_synthetic(const SyntheticParams& p);

// Data distribution utility — independent of where patches came from (synthetic
// or future real loaders). Partitions an M-patch list contiguously across `size`
// ranks and returns rank `rank`'s slice with its corresponding global patch IDs.
//
//   rank r owns global IDs [r*M/size, (r+1)*M/size).
struct PatchSlice {
    std::vector<std::vector<Hash>> patches;
    std::vector<PatchId>           global_ids;
};
PatchSlice slice_patches_by_rank(std::vector<std::vector<Hash>> all_patches,
                                  int rank, int size);

}  // namespace stPS
