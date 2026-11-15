#pragma once

#include "core/device_buffer.hpp"
#include <stPS/types.hpp>

#include <cstdint>
#include <vector>

namespace stComm { class Comm; }

namespace stPS {

// IdMappedPatches (see core/patch_set.hpp) with the translated patches kept on
// the device as a CSR: patch p's element IDs, sorted and unique, are
// data[offsets[p] .. offsets[p+1]). Local patch order is the input order.
struct DeviceIdMappedPatches {
    DeviceBuffer<ElementId>     data;
    DeviceBuffer<std::uint64_t> offsets;  // M_local + 1 entries
    std::uint64_t               M_local = 0;
    std::uint64_t               N = 0;

    // Same meaning as in IdMappedPatches; small (unique / ranks) and read on
    // the host by its consumers, so it is copied back.
    std::vector<Hash> shard_hashes;
    std::uint64_t     shard_start = 0;
};

// Collective: the distributed hash → ID mapping of map_hashes_to_element_ids,
// run on the GPU: sorting, de-duplication, the rank-to-rank exchanges
// (device-direct) and the per-patch translation all happen in device memory. Only scalars and the splitter samples cross to the
// host. Needs a device-enabled Comm.
//
// IDs, N and the shards are identical to the host mapping's.
//
// Peak device memory: 32 B per hash occurrence (the pair sort that orders each
// patch), on top of this rank's distinct hashes.
//
// `d_hashes` holds this rank's hash occurrences patch after patch and
// `d_offsets` their CSR offsets (M + 1 entries, the first 0, the last
// d_hashes.size()); both are consumed.
DeviceIdMappedPatches map_hashes_to_element_ids_device(
    stComm::Comm& comm, DeviceBuffer<Hash> d_hashes,
    DeviceBuffer<std::uint64_t> d_offsets, std::uint64_t M);

// Same, uploading a flat host slice first (`offsets` may be empty for "no
// patches").
DeviceIdMappedPatches map_hashes_to_element_ids_device(
    stComm::Comm& comm, const std::vector<Hash>& hashes,
    const std::vector<std::uint64_t>& offsets);

// Same, from one vector per patch: flattened on the host, then uploaded.
DeviceIdMappedPatches map_hashes_to_element_ids_device(
    stComm::Comm& comm, std::vector<std::vector<Hash>> raw_patches);

}  // namespace stPS
