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

// The same content in flat (CSR) form — one array of hashes, patch after
// patch, instead of one vector per patch. This is how a patch file is laid
// out, so a slice read as FlatPatches goes from disk to the algorithms without
// a per-patch allocation or a second copy; every entry point takes either form.
//
//   patch p = hashes[offsets[p] .. offsets[p + 1])
//
// `offsets` has patch_count() + 1 entries, starts at 0, never decreases and
// ends at hashes.size(); it may be left empty for "no patches". `coords` is
// empty or holds one Point per hash (coords[i] is where hashes[i] sits).
struct FlatPatches {
    std::vector<Hash>          hashes;
    std::vector<std::uint64_t> offsets;
    std::vector<PatchId>       global_ids;
    std::vector<Point>         coords;

    std::uint64_t patch_count() const noexcept {
        return offsets.empty() ? 0 : offsets.size() - 1;
    }
};

// Flatten a slice. Each patch's storage is released as it is copied, so the
// peak is one extra copy of the largest patch, not of the whole slice.
FlatPatches flatten_patches(PatchSlice slice);

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
