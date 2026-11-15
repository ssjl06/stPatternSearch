#include <stPS/partition.hpp>
#include "core/flat_patches.hpp"

#include <cstddef>
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

FlatPatches flatten_patches(PatchSlice slice) {
    FlatPatches out;
    const std::size_t M = slice.patches.size();
    const bool with_coords = !slice.coords.empty();
    if (with_coords && slice.coords.size() != M) {
        throw std::invalid_argument("flatten_patches: coords must have one entry per patch");
    }
    out.offsets.resize(M + 1);
    out.offsets[0] = 0;
    for (std::size_t p = 0; p < M; ++p) {
        if (with_coords && slice.coords[p].size() != slice.patches[p].size()) {
            throw std::invalid_argument(
                "flatten_patches: coords/patches shape mismatch at patch " + std::to_string(p));
        }
        out.offsets[p + 1] = out.offsets[p] + slice.patches[p].size();
    }
    out.hashes.reserve(out.offsets[M]);
    if (with_coords) out.coords.reserve(out.offsets[M]);
    for (std::size_t p = 0; p < M; ++p) {
        out.hashes.insert(out.hashes.end(), slice.patches[p].begin(), slice.patches[p].end());
        std::vector<Hash>().swap(slice.patches[p]);
        if (with_coords) {
            out.coords.insert(out.coords.end(), slice.coords[p].begin(), slice.coords[p].end());
            std::vector<Point>().swap(slice.coords[p]);
        }
    }
    out.global_ids = std::move(slice.global_ids);
    return out;
}

std::string flat_patches_error(const FlatPatches& patches) {
    const auto& off = patches.offsets;
    if (off.empty()) {
        if (!patches.hashes.empty()) return "hashes given without offsets";
    } else {
        if (off.front() != 0) return "offsets must start at 0";
        for (std::size_t p = 0; p + 1 < off.size(); ++p) {
            if (off[p] > off[p + 1]) {
                return "offsets decrease at patch " + std::to_string(p);
            }
        }
        if (off.back() != patches.hashes.size()) {
            return "offsets end at " + std::to_string(off.back()) + " but there are " +
                   std::to_string(patches.hashes.size()) + " hashes";
        }
    }
    if (!patches.coords.empty() && patches.coords.size() != patches.hashes.size()) {
        return "coords must be empty or hold one point per hash";
    }
    return {};
}

}  // namespace stPS
