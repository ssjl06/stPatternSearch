#pragma once

#include "core/types.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace fullchipusc {

// Compressed Sparse Row layout: for each patch p,
//   patch p's element IDs = data[offsets[p] .. offsets[p+1])
// data is the concatenation of all per-patch ID lists (each list sorted ascending).
struct PatchCsr {
    std::vector<ElementId>     data;
    std::vector<std::uint64_t> offsets;  // size = M + 1

    std::uint64_t M() const noexcept {
        return offsets.empty() ? 0 : offsets.size() - 1;
    }

    std::uint64_t patch_size(PatchId p) const noexcept {
        return offsets[p + 1] - offsets[p];
    }

    std::span<const ElementId> patch(PatchId p) const noexcept {
        return { data.data() + offsets[p], static_cast<std::size_t>(patch_size(p)) };
    }
};

// Build a CSR from a list of per-patch ID lists. Each inner list MUST already be
// sorted ascending and contain unique IDs (setup ensures this).
PatchCsr build_patch_csr(const std::vector<std::vector<ElementId>>& sorted_id_lists);

}  // namespace fullchipusc
