#include "core/patch_set.hpp"

#include "core/collective.hpp"
#include "core/csr.hpp"
#include "core/hash_id_map_device.hpp"
#include "core/host_profiler.hpp"
#include "core/inverted_index_device.hpp"

#include <stComm/stComm.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace stPS {

namespace detail {

std::vector<std::size_t> splitter_sample_indices(std::size_t n) {
    std::vector<std::size_t> idx;
    if (n == 0) return idx;
    const auto k = static_cast<std::size_t>(kSplitterSamplesPerRank);
    idx.resize(k);
    for (std::size_t i = 0; i < k; ++i) {
        // Fewer hashes than samples: repeat the last one, so every rank still
        // contributes the same fixed count.
        idx[i] = n <= k ? std::min(i, n - 1) : (i * (n - 1)) / (k - 1);
    }
    return idx;
}

std::vector<Hash> splitters_from_sample(stComm::Comm& comm, std::vector<Hash> local_sample) {
    const int size = comm.getSize();
    if (size <= 1) return {};
    // A rank without hashes sends zeros as placeholders — they are sorted with
    // the real samples; the bias is minimal.
    if (local_sample.empty()) local_sample.assign(kSplitterSamplesPerRank, Hash{0});

    // Uniform recvcounts: every rank sends exactly kSplitterSamplesPerRank.
    const std::size_t total_samples = static_cast<std::size_t>(size) * kSplitterSamplesPerRank;
    std::vector<Hash> all_samples(total_samples);
    std::vector<int>  recvcounts(static_cast<std::size_t>(size), kSplitterSamplesPerRank);
    comm.allgatherv<stComm::Space::Host, Hash>(local_sample.data(), kSplitterSamplesPerRank,
                                   all_samples.data(), recvcounts.data())->wait();

    // Every rank sorts the same samples, so every rank picks the same splitters.
    std::sort(all_samples.begin(), all_samples.end());
    std::vector<Hash> splitters;
    splitters.reserve(static_cast<std::size_t>(size - 1));
    for (int r = 1; r < size; ++r) {
        splitters.push_back(
            all_samples[static_cast<std::size_t>(r) * kSplitterSamplesPerRank]);
    }
    return splitters;
}

}  // namespace detail

namespace {

// Distributed sample sort splitter selection. Every rank computes the same
// (size-1) splitter values from gathered samples; afterwards each hash is
// routed to its target rank via upper_bound over splitters. This guarantees
// balanced bucket sizes regardless of hash distribution (unlike `hash % size`
// which assumes uniform distribution).
//
// Returns a sorted vector of size (size-1) splitter Hash values. For size == 1
// returns empty (no partitioning needed).
std::vector<Hash> compute_splitters(const std::vector<Hash>& local_uniq_sorted,
                                    stComm::Comm& comm) {
    std::vector<Hash> local_sample;
    for (std::size_t idx : detail::splitter_sample_indices(local_uniq_sorted.size())) {
        local_sample.push_back(local_uniq_sorted[idx]);
    }
    return detail::splitters_from_sample(comm, std::move(local_sample));
}

// Map a hash to its target rank via splitter binary search.
// For size=1 returns 0 unconditionally.
inline std::size_t target_rank_for_hash(Hash h, const std::vector<Hash>& splitters) {
    return static_cast<std::size_t>(
        std::upper_bound(splitters.begin(), splitters.end(), h) - splitters.begin());
}

}  // namespace

IdMappedPatches map_hashes_to_element_ids(stComm::Comm& comm,
                                          std::vector<std::vector<Hash>> raw_patches) {
    const int size = comm.getSize();
    const int rank = comm.getRank();
    const auto usize = static_cast<std::size_t>(size);
    using W = detail::PhaseTimer::Where;
    detail::PhaseTimer phases;

    // Design doc §5.1 distributed setup. Patches are already partitioned across
    // ranks (permanent). Within setup we perform a *temporary* hash partition:
    // each unique hash is routed to its owner rank, the owner assigns a
    // contiguous block of global IDs to its bucket, then each rank receives back
    // the IDs for the hashes it originally held. The hash partition is discarded;
    // what survives is each patch translated into dense ElementId space.

    // Step 1: flatten + dedupe this rank's local hash set.
    std::vector<Hash> local_uniq;
    {
        std::size_t total = 0;
        for (const auto& p : raw_patches) total += p.size();
        local_uniq.reserve(total);
        for (const auto& p : raw_patches) {
            local_uniq.insert(local_uniq.end(), p.begin(), p.end());
        }
        std::sort(local_uniq.begin(), local_uniq.end());
        local_uniq.erase(std::unique(local_uniq.begin(), local_uniq.end()), local_uniq.end());
    }

    phases.mark("flatten + sort + unique", W::Host);

    // Step 2: sample-sort based partitioning. Every rank computes the same
    // (size-1) splitters from gathered samples, then routes each hash to its
    // target rank via splitter binary search. local_uniq is already sorted.
    const std::vector<Hash> splitters = compute_splitters(local_uniq, comm);

    // local_uniq is sorted and the splitters are sorted, so target ranks are
    // non-decreasing along it: local_uniq already IS the send buffer, laid out
    // in target-rank order. Only the per-target counts are needed.
    std::vector<std::size_t> sendcounts(usize, 0);
    for (Hash h : local_uniq) ++sendcounts[target_rank_for_hash(h, splitters)];

    // Step 3: exchange the counts so every rank learns its recv counts.
    const ExchangeCounts counts = exchange_counts(comm, std::move(sendcounts));

    // Step 4: alltoallv — every hash flies to its owner rank. size_t counts:
    // stComm's unlimited-size exchange, so a rank's unique-hash set may be any
    // size (the int-count collectives stop at INT_MAX elements).
    const std::size_t total_recv =
        std::accumulate(counts.recv.begin(), counts.recv.end(), std::size_t{0});
    std::vector<Hash> recvbuf(total_recv);
    comm.alltoallv<stComm::Space::Host, Hash>(local_uniq.data(), counts.send.data(),
                                   recvbuf.data(), counts.recv.data())->wait();

    phases.mark("splitters + hash exchange", W::HostComm);

    // Step 5: form this rank's bucket universe — sorted unique view of recvbuf.
    // Keep recvbuf intact for the per-arrival-slot lookup in Step 8.
    std::vector<Hash> bucket(recvbuf);
    std::sort(bucket.begin(), bucket.end());
    bucket.erase(std::unique(bucket.begin(), bucket.end()), bucket.end());

    // Step 6: Exscan over bucket sizes → this rank's first global ID.
    const std::uint64_t local_shard_size = bucket.size();
    std::uint64_t global_id_start = 0;
    comm.exscan<stComm::Space::Host>(local_shard_size, &global_id_start,
                                     stComm::ReduceOp::Sum)->wait();

    // Step 7: only the last rank knows N at this point; broadcast it.
    std::uint64_t N_total = (rank == size - 1) ? (global_id_start + local_shard_size) : 0;
    comm.bcast<stComm::Space::Host, std::uint64_t>(&N_total, 1, size - 1)->wait();

    // Step 8: for every slot in recvbuf (arrival order, possibly with dupes),
    // produce the corresponding global ID. The reverse alltoallv expects responses
    // in the same slot order as recvbuf — that's how it routes back to senders.
    std::vector<std::uint64_t> id_response(total_recv);
    for (std::size_t i = 0; i < recvbuf.size(); ++i) {
        const auto it = std::lower_bound(bucket.begin(), bucket.end(), recvbuf[i]);
        id_response[i] = global_id_start + static_cast<std::uint64_t>(it - bucket.begin());
    }
    recvbuf.clear(); recvbuf.shrink_to_fit();

    phases.mark("shard sort + id assignment", W::Host);

    // Step 9: reverse alltoallv — IDs travel back to original senders. The
    // send and recv counts swap roles relative to Step 4, so id_back[i] is the
    // global ID of local_uniq[i].
    std::vector<std::uint64_t> id_back(local_uniq.size());
    comm.alltoallv<stComm::Space::Host, std::uint64_t>(id_response.data(), counts.recv.data(),
                                            id_back.data(),     counts.send.data())->wait();
    id_response.clear(); id_response.shrink_to_fit();

    phases.mark("id return exchange", W::HostComm);

    // Step 10: local_uniq[i] ↔ id_back[i] are paired by construction. Build
    // this rank's hash → id map. A hash table on purpose: a binary search over
    // local_uniq needs no extra memory but measured ~2.5× slower for the whole
    // setup (one cache-missing probe chain per occurrence).
    std::unordered_map<Hash, ElementId> hash_to_id;
    hash_to_id.reserve(local_uniq.size() * 2);
    for (std::size_t i = 0; i < local_uniq.size(); ++i) {
        hash_to_id.emplace(local_uniq[i], id_back[i]);
    }
    local_uniq.clear(); local_uniq.shrink_to_fit();
    id_back.clear();    id_back.shrink_to_fit();

    // Step 11: translate raw_patches into ID-space, sort + dedupe per patch. The
    // local patch index [0, M_local) is preserved in input order. The bucket
    // (this rank's element shard, sorted) survives as the ElementId → Hash
    // reverse map: bucket[i] ↔ global ID (global_id_start + i) by Step 8.
    IdMappedPatches out;
    out.N            = N_total;
    out.shard_hashes = std::move(bucket);
    out.shard_start  = global_id_start;
    out.id_patches.resize(raw_patches.size());
    for (std::size_t p = 0; p < raw_patches.size(); ++p) {
        std::vector<Hash>& src = raw_patches[p];
        auto& dst = out.id_patches[p];
        dst.resize(src.size());
        for (std::size_t i = 0; i < src.size(); ++i) {
            dst[i] = hash_to_id.at(src[i]);
        }
        std::sort(dst.begin(), dst.end());
        dst.erase(std::unique(dst.begin(), dst.end()), dst.end());
        std::vector<Hash>().swap(src);  // release the hashes as we go
    }
    phases.mark("hash table + translate patches", W::Host);
    phases.report(stderr, "[stps-profile hash->id map]", rank);
    return out;
}

PatchSet::PatchSet(stComm::Comm& comm, std::vector<std::vector<Hash>> raw_patches,
                   PatchSetMirrors mirrors) {
    // Shared §5.1 hash → ID mapping, then the patch-partitioned structures:
    // patch CSR + inverted index stay partitioned by patch ID.
    using W = detail::PhaseTimer::Where;
    detail::PhaseTimer phases;
    DeviceIdMappedPatches mapped =
        map_hashes_to_element_ids_device(comm, std::move(raw_patches));
    N_            = mapped.N;
    shard_hashes_ = std::move(mapped.shard_hashes);
    shard_start_  = mapped.shard_start;
    phases.mark("hash->id map (detailed above)", W::Device);

    // Inverted index: built on the device from the device CSR. UPS reads only
    // its layout (keys + offsets), so the values are dropped there.
    const std::uint64_t M     = mapped.M_local;
    const std::uint64_t total = mapped.data.size();
    DeviceInvertedIndex inv = build_inverted_index_device(
        mapped.data.data(), mapped.offsets.data(), M, total, N_);
    d_inv_keys_    = std::move(inv.keys);
    d_inv_offsets_ = std::move(inv.offsets);
    if (mirrors == PatchSetMirrors::All) d_inv_data_ = std::move(inv.data);
    phases.mark("inverted index build", W::Device);

    // Host views: the patch CSR and the index keys.
    patches_.offsets.resize(M + 1);
    mapped.offsets.copy_to_host(patches_.offsets.data(), M + 1);
    patches_.data.resize(total);
    if (total) mapped.data.copy_to_host(patches_.data.data(), total);
    inv_keys_.resize(inv.num_keys);
    if (inv.num_keys > 0) d_inv_keys_.copy_to_host(inv_keys_.data(), inv_keys_.size());
    if (mirrors == PatchSetMirrors::All) {
        d_patch_data_    = std::move(mapped.data);
        d_patch_offsets_ = std::move(mapped.offsets);
    }
    phases.mark("host views download", W::Transfer);
    phases.report(stderr, "[stps-profile PatchSet]", comm.getRank());
}

}  // namespace stPS
