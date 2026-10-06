#include <stPS/unique_hashes.hpp>

#include "core/patch_set.hpp"
#include "io/parallel_file.hpp"
#include "ups/unique_hash_file.hpp"

#include <sys/types.h>

#include <cstring>
#include <utility>

namespace stPS {

void write_unique_hash_shards(stComm::Comm& comm, const std::string& path,
                              const std::vector<Hash>& shard_hashes,
                              std::uint64_t shard_start, std::uint64_t N) {
    // §5.1 shards are contiguous, rank-ordered and each sorted, so this
    // rank's hashes land at a fixed offset and the file is globally sorted
    // with no further exchange.
    unsigned char header[kUniqueHashHeaderBytes];
    std::memcpy(header, kUniqueHashMagic, sizeof(kUniqueHashMagic));
    std::memcpy(header + sizeof(kUniqueHashMagic), &N, sizeof(N));

    const off_t offset = static_cast<off_t>(kUniqueHashHeaderBytes + shard_start * sizeof(Hash));
    const off_t total  = static_cast<off_t>(kUniqueHashHeaderBytes + N * sizeof(Hash));
    io::write_shared_file(comm, path, "unique-hashes",
                          shard_hashes.data(), shard_hashes.size() * sizeof(Hash),
                          offset, total, header, sizeof(header));
}

std::uint64_t write_unique_hashes_file(stComm::Comm& comm, const std::string& path,
                                       std::vector<std::vector<Hash>> patches) {
    // Only the hash sort's shards are needed; the hash → ID translation the
    // mapping also does is unused here (cheap next to the sort itself).
    const IdMappedPatches mapped = map_hashes_to_element_ids(comm, std::move(patches));
    write_unique_hash_shards(comm, path, mapped.shard_hashes, mapped.shard_start, mapped.N);
    return mapped.N;
}

}  // namespace stPS
