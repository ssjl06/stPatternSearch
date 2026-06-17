#include "core/inverted_index.hpp"

#include <algorithm>
#include <cassert>

namespace stPS {

std::span<const PatchId> InvertedIndex::patches_of(ElementId e) const noexcept {
    const auto it = std::lower_bound(keys.begin(), keys.end(), e);
    if (it == keys.end() || *it != e) {
        return {};
    }
    const std::uint64_t k = static_cast<std::uint64_t>(it - keys.begin());
    const std::uint64_t b = offsets[k];
    const std::uint64_t n = offsets[k + 1] - b;
    return { data.data() + b, static_cast<std::size_t>(n) };
}

InvertedIndex build_inverted_index(const PatchCsr& patches, std::uint64_t N) {
    InvertedIndex inv;
    const std::uint64_t total = patches.data.size();
    if (total == 0) {
        inv.offsets.assign(1, 0);
        return inv;
    }

    // Pass 1: collect (element, patch) pairs. Use a parallel-arrays representation
    // so we can sort by element ID and use a stable group structure afterward.
    // For M1 we work in-memory; the design doc's distributed variant (§5.1) replaces
    // this with sample sort across ranks.
    std::vector<ElementId> elems(total);
    std::vector<PatchId>   owners(total);
    {
        const std::uint64_t M = patches.M();
        std::uint64_t cursor = 0;
        for (PatchId p = 0; p < M; ++p) {
            const std::uint64_t b = patches.offsets[p];
            const std::uint64_t e = patches.offsets[p + 1];
            for (std::uint64_t i = b; i < e; ++i) {
                elems[cursor]  = patches.data[i];
                owners[cursor] = p;
                ++cursor;
            }
        }
        assert(cursor == total);
    }

    // Indirect sort by element ID, keeping (elem, owner) paired.
    std::vector<std::uint64_t> order(total);
    for (std::uint64_t i = 0; i < total; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](std::uint64_t a, std::uint64_t b) {
                  if (elems[a] != elems[b]) return elems[a] < elems[b];
                  return owners[a] < owners[b];  // deterministic tie-break
              });

    // Count unique element IDs to size keys/offsets.
    std::uint64_t num_keys = 0;
    for (std::uint64_t i = 0; i < total; ) {
        const ElementId e = elems[order[i]];
        ++num_keys;
        std::uint64_t j = i + 1;
        while (j < total && elems[order[j]] == e) ++j;
        i = j;
    }

    inv.keys.resize(num_keys);
    inv.offsets.assign(num_keys + 1, 0);
    inv.data.resize(total);

    // Fill keys + offsets + data in one pass.
    std::uint64_t k = 0;
    std::uint64_t write = 0;
    for (std::uint64_t i = 0; i < total; ) {
        const ElementId e = elems[order[i]];
        inv.keys[k]    = e;
        inv.offsets[k] = write;
        std::uint64_t j = i;
        while (j < total && elems[order[j]] == e) {
            inv.data[write++] = owners[order[j]];
            ++j;
        }
        ++k;
        i = j;
    }
    inv.offsets[num_keys] = write;
    assert(write == total);
    assert(k == num_keys);

    // Sanity: all element IDs fit in [0, N).
    if (!inv.keys.empty()) {
        assert(inv.keys.back() < N);
    }
    (void)N;
    return inv;
}

}  // namespace stPS
