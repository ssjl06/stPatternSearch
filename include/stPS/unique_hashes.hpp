#pragma once

#include <stPS/types.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace stComm { class Comm; }

namespace stPS {

// Unique-hash file: every distinct hash across all ranks' patches, ascending,
// no duplicates — the full set, unlike the top-k stats file.
//
//   magic   8 B        "STPSUHS1"
//   N       8 B        uint64 hash count
//   hashes  N × 8 B    uint64, strictly ascending
//
// Host byte order (little-endian on x86-64/aarch64), same as .stps. Reading
// it back in numpy: np.fromfile(path, dtype=np.uint64, offset=16).
inline constexpr char        kUniqueHashMagic[8]     = {'S','T','P','S','U','H','S','1'};
inline constexpr std::size_t kUniqueHashHeaderBytes  = 16;

// Collective: write the unique-hash file for every rank's `patches` (each
// rank passes its own slice) and return the global unique count N, identical
// on every rank. Host-only — no GPU or NCCL needed; it runs the distributed
// hash sort (design doc §5.1) and each rank writes its own sorted shard in
// parallel, so the output is the same for any rank count.
//
// `path` must be one file for every rank (a shared filesystem across nodes).
// If any rank's I/O fails, every rank throws std::runtime_error.
std::uint64_t write_unique_hashes_file(stComm::Comm& comm, const std::string& path,
                                       std::vector<std::vector<Hash>> patches);

}  // namespace stPS
