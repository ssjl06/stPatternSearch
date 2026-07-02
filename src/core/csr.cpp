#include "core/csr.hpp"

namespace stPS {

PatchCsr build_patch_csr(const std::vector<std::vector<ElementId>>& sorted_id_lists) {
    PatchCsr csr;
    const std::uint64_t M = sorted_id_lists.size();
    csr.offsets.resize(M + 1);
    csr.offsets[0] = 0;

    std::uint64_t total = 0;
    for (std::uint64_t p = 0; p < M; ++p) {
        total += sorted_id_lists[p].size();
        csr.offsets[p + 1] = total;
    }

    csr.data.resize(total);
    for (std::uint64_t p = 0; p < M; ++p) {
        const auto& src = sorted_id_lists[p];
        const std::uint64_t off = csr.offsets[p];
        for (std::size_t i = 0; i < src.size(); ++i) {
            csr.data[off + i] = src[i];
        }
    }
    return csr;
}

}  // namespace stPS
