#pragma once

#include "core/patch_set.hpp"
#include <stPS/types.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace stComm { class Comm; }

namespace stPS {

// One hash's global statistics: how many patches contain it (per-patch
// deduped, summed across ranks) and its representative location — the
// lexicographically smallest (x first, then y) coordinate over every
// occurrence anywhere.
struct HashStat {
    Hash          hash  = 0;
    std::uint64_t count = 0;
    Point         rep;
};

// Local pre-pass (no comm): collapse this rank's occurrences to one
// lexicographic-min location per unique hash, sorted by hash. coords must
// mirror patches shape (coords[p][i] belongs to patches[p][i]). Run this
// BEFORE handing `patches` to PatchSet — the PatchSet constructor consumes
// them, and global_top_k relies on this hash-sorted order aligning 1:1 with
// the PatchSet's inverted-index keys.
std::vector<std::pair<Hash, Point>> local_min_locations(
    const std::vector<std::vector<Hash>>& patches,
    const std::vector<std::vector<Point>>& coords);

// Collective: global top-k hash statistics (count desc, hash asc tiebreak),
// identical on every rank. `minloc` is this rank's local_min_locations result
// for the same patches the PatchSet was built from. k is clamped to the
// global unique-hash count N.
std::vector<HashStat> global_top_k(stComm::Comm& comm, const PatchSet& ps,
                                   const std::vector<std::pair<Hash, Point>>& minloc,
                                   std::uint64_t k);

// Collective: write `topk` (identical on every rank, from global_top_k) to a
// single text file, every rank writing its own contiguous line range in
// parallel. Fixed-width records make every byte offset a pure function of
// the line index — no collectives, just POSIX pwrite on a shared filesystem.
// Overwrites `path`; all ranks have completed (and the file has its final
// size) when this returns.
void write_stats_file(stComm::Comm& comm, const std::string& path,
                      const std::vector<HashStat>& topk);

}  // namespace stPS
