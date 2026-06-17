#pragma once

#include "core/csr.hpp"
#include <stPS/types.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace stPS {

// Sparse CSR transpose of PatchCsr (design doc §4.4).
//   keys[k]     : the k-th element ID that actually appears in this rank's patches
//                 (sorted ascending; size = local unique element count, NOT N)
//   offsets[k]  : where patches containing keys[k] start in data; size = keys.size() + 1
//   data[...]   : patch IDs, grouped by element
//
// Lookup: patches_of(e) binary-searches keys for e. If found, returns the
// corresponding patch-ID range; otherwise returns an empty span.
//
// Memory cost: O(local_unique_elements + total_patch_data) — does NOT scale with N.
struct InvertedIndex {
    std::vector<ElementId>     keys;
    std::vector<std::uint64_t> offsets;  // size = keys.size() + 1
    std::vector<PatchId>       data;

    std::uint64_t num_keys() const noexcept { return keys.size(); }

    std::span<const PatchId> patches_of(ElementId e) const noexcept;
};

// Build a sparse inverted index by transposing a PatchCsr.
// N is the global universe size (used only for sanity assertions; the index itself
// stores only elements that actually appear in `patches`).
InvertedIndex build_inverted_index(const PatchCsr& patches, std::uint64_t N);

}  // namespace stPS
