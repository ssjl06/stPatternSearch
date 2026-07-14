#pragma once

#include <stPS/types.hpp>

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

// Deterministic per-occurrence coordinates for a generated patch set: the
// result mirrors `patches` shape (coords[p][i] belongs to patches[p][i]),
// derived from p.seed only — same params + patches always give the same
// coordinates. Kept separate from generate_synthetic so the USC path (which
// has no use for locations) stays untouched.
std::vector<std::vector<Point>> generate_synthetic_coords(
    const SyntheticParams& p, const std::vector<std::vector<Hash>>& patches);

}  // namespace stPS
