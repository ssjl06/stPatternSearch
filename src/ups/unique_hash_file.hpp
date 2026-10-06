#pragma once

#include <stPS/types.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace stComm { class Comm; }

namespace stPS {

// Collective: write the unique-hash file (format in <stPS/unique_hashes.hpp>)
// from the §5.1 shards already in hand — `shard_hashes` is this rank's sorted
// slice of the global unique set starting at global index `shard_start`, and
// `N` the global count. Shared by write_unique_hashes_file and
// UpsPatternStats::pattern_stats (which reuses its PatchSet's shards, so the
// hash sort runs once).
void write_unique_hash_shards(stComm::Comm& comm, const std::string& path,
                              const std::vector<Hash>& shard_hashes,
                              std::uint64_t shard_start, std::uint64_t N);

}  // namespace stPS
