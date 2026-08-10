#pragma once

#include <stPS/types.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Forward-declared so this public header pulls in no MPI/NCCL/CUDA headers.
namespace stComm { class Comm; }

namespace stPS {

// One pattern's global statistics: how many patches contain its hash
// (per-patch deduped, summed across ranks) and its representative location —
// the lexicographically smallest (x first, then y) coordinate over every
// occurrence on any rank. Identical on every rank.
struct PatternStat {
    Hash          hash  = 0;
    std::uint64_t count = 0;
    Point         rep;
};

// Distributed pattern statistics over a hashed patch set — the UPS entry
// point. Internals (the shared PatchSet setup, device buffers, kernels) are
// hidden behind a pImpl, so this header stays dependency-free.
//
// GPU-only (the setup builds device mirrors). Construct once and reuse across
// pattern_stats() calls.
class UpsPatternStats {
public:
    explicit UpsPatternStats(stComm::Comm& comm);
    ~UpsPatternStats();

    UpsPatternStats(UpsPatternStats&&) noexcept;
    UpsPatternStats& operator=(UpsPatternStats&&) noexcept;
    UpsPatternStats(const UpsPatternStats&)            = delete;
    UpsPatternStats& operator=(const UpsPatternStats&) = delete;

    // Run the full pipeline (distributed §5.1 setup + counting + top-k merge)
    // on this rank's `patches` and their per-occurrence `coords` (must mirror
    // patches shape: coords[p][i] is where occurrence patches[p][i] sits).
    // **Collective** — every rank must call together; all ranks return the
    // identical top-k list (count desc, hash asc; size = min(k, global unique
    // hash count)). Throws std::invalid_argument on a coords shape mismatch.
    std::vector<PatternStat> pattern_stats(std::vector<std::vector<Hash>>  patches,
                                           std::vector<std::vector<Point>> coords,
                                           std::uint64_t k);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Collective: write `stats` (identical on every rank, from pattern_stats) to a
// single text file, every rank writing its own contiguous line range in
// parallel (fixed-width records → offsets are pure functions of the line
// index; assumes a POSIX-coherent shared filesystem). Overwrites `path`; all
// ranks have completed — and the file has its final size — when this returns.
void write_pattern_stats_file(stComm::Comm& comm, const std::string& path,
                              const std::vector<PatternStat>& stats);

}  // namespace stPS
