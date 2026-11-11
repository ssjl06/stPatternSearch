#include <stPS/partition.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace stPS {

RankRange rank_range(std::uint64_t count, int rank, int size) {
    if (size <= 0 || rank < 0 || rank >= size) {
        throw std::invalid_argument("rank_range: rank " + std::to_string(rank) +
                                    " is not in [0, " + std::to_string(size) + ")");
    }
    const std::uint64_t rk64 = static_cast<std::uint64_t>(rank);
    const std::uint64_t sz64 = static_cast<std::uint64_t>(size);
    return { (count * rk64) / sz64, (count * (rk64 + 1)) / sz64 };
}

PatchSlice slice_patches_by_rank(std::vector<std::vector<Hash>> all_patches,
                                 int rank, int size) {
    const RankRange mine = rank_range(all_patches.size(), rank, size);

    PatchSlice slice;
    slice.patches.reserve(mine.end - mine.begin);
    slice.global_ids.reserve(mine.end - mine.begin);
    for (std::uint64_t p = mine.begin; p < mine.end; ++p) {
        slice.patches.push_back(std::move(all_patches[p]));
        slice.global_ids.push_back(p);
    }
    return slice;
}

PatchSlice slice_patches_by_rank(std::vector<std::vector<Hash>> all_patches,
                                 std::vector<std::vector<Point>> coords,
                                 int rank, int size) {
    // Checked, not asserted: a short coords vector would otherwise be read out
    // of bounds below in a release build.
    if (coords.size() != all_patches.size()) {
        throw std::invalid_argument(
            "slice_patches_by_rank: coords has " + std::to_string(coords.size()) +
            " entries for " + std::to_string(all_patches.size()) + " patches");
    }
    const std::uint64_t my_begin = rank_range(all_patches.size(), rank, size).begin;

    PatchSlice slice = slice_patches_by_rank(std::move(all_patches), rank, size);
    slice.coords.reserve(slice.patches.size());
    for (std::uint64_t p = 0; p < slice.patches.size(); ++p) {
        slice.coords.push_back(std::move(coords[my_begin + p]));
    }
    return slice;
}

}  // namespace stPS
