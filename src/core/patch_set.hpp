#pragma once

#include "core/csr.hpp"
#include "core/device_buffer.hpp"
#include <stPS/types.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

// Forward-declared so this internal header drags in no <mpi.h>/<nccl.h>. The
// concrete type is included in patch_set.cpp where its members are called.
namespace stComm { class Comm; }

namespace stPS {

// Result of the distributed hash → dense-ID mapping (design doc §5.1): this
// rank's patches translated into ElementId space (each patch sorted + unique,
// local input order preserved), plus the global universe size N. Shared by both
// partitioning strategies — PatchSet builds its patch-partitioned structures on
// top of it, and the element-partitioned selector (§7.3) redistributes it by
// element shard instead.
struct IdMappedPatches {
    std::vector<std::vector<ElementId>> id_patches;
    std::uint64_t                       N = 0;

    // This rank's element shard from the temporary hash partition (§5.1 steps
    // 5–6), kept as the ElementId → Hash reverse map: shard_hashes[i] is the
    // original hash of global ID (shard_start + i), sorted ascending. Shards
    // are contiguous and ordered by rank: rank r owns
    // [shard_start_r, shard_start_r + shard_hashes_r.size()). O(unique/size)
    // per rank — consumers that report hashes (UPS stats) read it back here.
    std::vector<Hash> shard_hashes;
    std::uint64_t     shard_start = 0;
};

// Collective: distributed sample-sort hash → ID mapping (§5.1 steps 1–11).
// Every rank must call together with its own local patches.
IdMappedPatches map_hashes_to_element_ids(stComm::Comm& comm,
                                          std::vector<std::vector<Hash>> raw_patches);

namespace detail {

// Sample-sort splitter selection, shared by the host and device mappings.
inline constexpr int kSplitterSamplesPerRank = 128;

// Evenly spaced positions in a sorted array of n values to sample: exactly
// kSplitterSamplesPerRank of them, or none when n == 0.
std::vector<std::size_t> splitter_sample_indices(std::size_t n);

// Collective: gather every rank's sample (the values at the positions above;
// empty for a rank without hashes) and return the size-1 splitters, identical
// on every rank. A hash h belongs to rank upper_bound(splitters, h).
std::vector<Hash> splitters_from_sample(stComm::Comm& comm, std::vector<Hash> local_sample);

}  // namespace detail

// Distributed, algorithm-neutral "hashed patch set" — the shared preprocessing
// every patch algorithm (USC greedy set-cover, UPS pattern counting, ...) builds
// on. Given each rank's patches as lists of Hash, it performs the distributed
// hash → dense ElementId mapping (sample-sort, design doc §5.1) and builds the
// structures the algorithms consume: a PatchCsr (patches in ElementId space) and
// its sparse InvertedIndex, plus their device mirrors and the global universe N.
//
// It holds NO algorithm state (no scores/covered) and NO per-patch application
// metadata: the local patch index [0, M_local) is preserved in input order, so
// each algorithm layer keeps its own per-patch data (USC: global PatchIds; UPS:
// spatial locations) indexed identically.
//
// GPU-only, move-only. `comm` is used only during construction (the collective
// setup) and is not retained.

// Which host structures get a device mirror. Every incidence costs 16 B of GPU
// memory in the full set (patch CSR data + inverted-index values), so an
// algorithm whose kernels never read those asks for the layout only.
enum class PatchSetMirrors {
    All,                  // patch data + inverted-index keys/offsets/values (USC)
    InvertedIndexLayout,  // inverted-index keys + offsets only (UPS)
};

class PatchSet {
public:
    // Collective: every rank must construct together with its own local patches.
    PatchSet(stComm::Comm& comm, std::vector<std::vector<Hash>> raw_patches,
             PatchSetMirrors mirrors = PatchSetMirrors::All);

    // Host-side views.
    std::uint64_t        N() const noexcept { return N_; }
    std::uint64_t        M_local() const noexcept { return patches_.M(); }
    const PatchCsr&      patches() const noexcept { return patches_; }
    // Sorted element IDs that occur in this rank's patches — the inverted
    // index keys. The index itself is built and kept on the device (see the
    // d_inv_* mirrors); only its keys are copied back.
    const std::vector<ElementId>& inv_keys() const noexcept { return inv_keys_; }

    // ElementId → Hash reverse map for this rank's element shard (see
    // IdMappedPatches::shard_hashes): shard_hashes()[i] is the hash of global
    // ID (shard_start() + i). Shards are contiguous and rank-ordered.
    const std::vector<Hash>& shard_hashes() const noexcept { return shard_hashes_; }
    std::uint64_t            shard_start()  const noexcept { return shard_start_; }

    // Device mirrors (populated at construction) — raw const pointers for
    // kernels. d_patch_data / d_patch_offsets / d_inv_data are null unless PatchSetMirrors::All.
    const ElementId*     d_patch_data()    const noexcept { return d_patch_data_.data(); }
    const std::uint64_t* d_patch_offsets() const noexcept { return d_patch_offsets_.data(); }
    const ElementId*     d_inv_keys()      const noexcept { return d_inv_keys_.data(); }
    const std::uint64_t* d_inv_offsets()   const noexcept { return d_inv_offsets_.data(); }
    const PatchId*       d_inv_data()      const noexcept { return d_inv_data_.data(); }

private:
    PatchCsr      patches_;
    std::vector<ElementId> inv_keys_;
    std::uint64_t N_ = 0;

    std::vector<Hash> shard_hashes_;
    std::uint64_t     shard_start_ = 0;

    DeviceBuffer<ElementId>     d_patch_data_;
    DeviceBuffer<std::uint64_t> d_patch_offsets_;
    DeviceBuffer<ElementId>     d_inv_keys_;
    DeviceBuffer<std::uint64_t> d_inv_offsets_;
    DeviceBuffer<PatchId>       d_inv_data_;
};

}  // namespace stPS
