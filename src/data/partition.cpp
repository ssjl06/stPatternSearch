#include <stPS/partition.hpp>

#include <cstdint>
#include <utility>

namespace stPS {

PatchSlice slice_patches_by_rank(std::vector<std::vector<Hash>> all_patches,
                                 int rank, int size) {
    const std::uint64_t M    = all_patches.size();
    const std::uint64_t rk64 = static_cast<std::uint64_t>(rank);
    const std::uint64_t sz64 = static_cast<std::uint64_t>(size);
    const std::uint64_t my_begin = (M * rk64)       / sz64;
    const std::uint64_t my_end   = (M * (rk64 + 1)) / sz64;

    PatchSlice slice;
    slice.patches.reserve(my_end - my_begin);
    slice.global_ids.reserve(my_end - my_begin);
    for (std::uint64_t p = my_begin; p < my_end; ++p) {
        slice.patches.push_back(std::move(all_patches[p]));
        slice.global_ids.push_back(static_cast<PatchId>(p));
    }
    return slice;
}

}  // namespace stPS
