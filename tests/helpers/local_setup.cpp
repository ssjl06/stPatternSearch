#include "helpers/local_setup.hpp"

#include <algorithm>
#include <cstddef>

namespace stPS::test_helpers {

LocalSetupResult run_local_setup(const std::vector<std::vector<Hash>>& raw_patches) {
    LocalSetupResult out;

    std::size_t total = 0;
    for (const auto& p : raw_patches) total += p.size();
    std::vector<Hash> all_hashes;
    all_hashes.reserve(total);
    for (const auto& p : raw_patches) {
        all_hashes.insert(all_hashes.end(), p.begin(), p.end());
    }
    std::sort(all_hashes.begin(), all_hashes.end());
    all_hashes.erase(std::unique(all_hashes.begin(), all_hashes.end()), all_hashes.end());
    out.id_to_hash = std::move(all_hashes);
    out.N = out.id_to_hash.size();

    std::vector<std::vector<ElementId>> id_patches(raw_patches.size());
    for (std::size_t p = 0; p < raw_patches.size(); ++p) {
        const auto& src = raw_patches[p];
        auto& dst = id_patches[p];
        dst.resize(src.size());
        for (std::size_t i = 0; i < src.size(); ++i) {
            const auto it = std::lower_bound(out.id_to_hash.begin(), out.id_to_hash.end(), src[i]);
            dst[i] = static_cast<ElementId>(it - out.id_to_hash.begin());
        }
        std::sort(dst.begin(), dst.end());
        dst.erase(std::unique(dst.begin(), dst.end()), dst.end());
    }
    out.patches = build_patch_csr(id_patches);
    out.inv     = build_inverted_index(out.patches, out.N);
    return out;
}

}  // namespace stPS::test_helpers
