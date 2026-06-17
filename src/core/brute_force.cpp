#include "core/brute_force.hpp"

#include "core/bitset.hpp"

namespace stPS {

PatchSelection brute_force_select(const PatchCsr& patches, std::uint64_t N) {
    const std::uint64_t M = patches.M();
    PatchSelection result;
    result.covered_count = 0;
    result.iterations    = 0;

    DenseBitset covered(N);
    std::vector<bool> selected_flag(M, false);

    while (result.covered_count < N) {
        // Recompute |patch[p] \ covered| for every patch.
        std::int64_t best_score = 0;
        PatchId      best_patch = 0;
        bool         found = false;
        for (PatchId p = 0; p < M; ++p) {
            if (selected_flag[p]) continue;
            std::int64_t s = 0;
            for (ElementId e : patches.patch(p)) {
                if (!covered.test(e)) ++s;
            }
            // Tie-break: smaller PatchId wins. The strict > guarantees this since
            // we iterate p in ascending order.
            if (s > best_score) {
                best_score = s;
                best_patch = p;
                found      = true;
            }
        }
        if (!found || best_score <= 0) break;

        // Apply: set bits for the chosen patch's uncovered elements.
        for (ElementId e : patches.patch(best_patch)) {
            if (!covered.test(e)) {
                covered.set(e);
                ++result.covered_count;
            }
        }
        selected_flag[best_patch] = true;
        result.selected.push_back(best_patch);
        ++result.iterations;
    }

    return result;
}

}  // namespace stPS
