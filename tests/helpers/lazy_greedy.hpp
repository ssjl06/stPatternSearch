#pragma once

#include "core/csr.hpp"
#include <stPS/usc_patch_selector.hpp>  // PatchSelection
#include <stPS/types.hpp>

#include <cstdint>

namespace stPS::test_helpers {

// Exact greedy set cover with lazy score evaluation — the same answer as
// brute_force_select (argmax of |patch \ covered|, tie: smaller PatchId), in
// roughly O(incidences · log M) instead of O(S · M · K), so it stays usable as
// ground truth at sizes where brute force does not finish.
//
// A max-heap holds (score, patch) with possibly stale scores. Scores only ever
// drop, so a popped entry whose recomputed score still equals its stored one
// beats every other patch's true score; among equal stored scores the heap
// yields the smaller PatchId first, which is the tie rule.
PatchSelection lazy_greedy_select(const PatchCsr& patches, std::uint64_t N);

}  // namespace stPS::test_helpers
