#include "core/patch_set.hpp"

#include "core/csr.hpp"
#include "core/inverted_index.hpp"

#include <stComm/stComm.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace stPS {

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
    constexpr int kSamplesPerRank = 128;
    const int size = comm.getSize();
    if (size <= 1) return {};

    // Step 1: pick kSamplesPerRank evenly-spaced indices from local_uniq_sorted.
    // If we have fewer hashes than samples, pad by repeating the last element
    // so every rank contributes the same fixed count (simplifies allgather).
    std::vector<Hash> local_sample(kSamplesPerRank);
    const std::size_t n = local_uniq_sorted.size();
    if (n == 0) {
        // Edge case: rank has no hashes. Use 0 as a placeholder — these samples
        // will be sorted with the real ones; bias is minimal since other ranks
        // contribute meaningful samples.
        std::fill(local_sample.begin(), local_sample.end(), Hash{0});
    } else if (n <= static_cast<std::size_t>(kSamplesPerRank)) {
        for (int i = 0; i < kSamplesPerRank; ++i) {
            local_sample[i] = local_uniq_sorted[std::min<std::size_t>(i, n - 1)];
        }
    } else {
        for (int i = 0; i < kSamplesPerRank; ++i) {
            const std::size_t idx =
                (static_cast<std::size_t>(i) * (n - 1)) / (kSamplesPerRank - 1);
            local_sample[i] = local_uniq_sorted[idx];
        }
    }

    // Step 2: allgatherv all samples — uniform recvcounts since every rank
    // sends exactly kSamplesPerRank hashes.
    const std::size_t total_samples = static_cast<std::size_t>(size) * kSamplesPerRank;
    std::vector<Hash> all_samples(total_samples);
    std::vector<int>  recvcounts(static_cast<std::size_t>(size), kSamplesPerRank);
    comm.allgatherv<stComm::Space::Host, Hash>(local_sample.data(), kSamplesPerRank,
                                   all_samples.data(), recvcounts.data())->wait();

    // Step 3: sort all samples (every rank computes identical splitters).
    std::sort(all_samples.begin(), all_samples.end());

    // Step 4: pick (size-1) splitters at positions kSamplesPerRank, 2*kSamplesPerRank, ...
    std::vector<Hash> splitters;
    splitters.reserve(static_cast<std::size_t>(size - 1));
    for (int r = 1; r < size; ++r) {
        splitters.push_back(
            all_samples[static_cast<std::size_t>(r) * kSamplesPerRank]);
    }
    return splitters;
}

// Map a hash to its target rank via splitter binary search.
// For size=1 returns 0 unconditionally.
inline int target_rank_for_hash(Hash h, const std::vector<Hash>& splitters) {
    return static_cast<int>(
        std::upper_bound(splitters.begin(), splitters.end(), h) - splitters.begin());
}

}  // namespace

PatchSet::PatchSet(stComm::Comm& comm, std::vector<std::vector<Hash>> raw_patches) {
    const int size = comm.getSize();
    const int rank = comm.getRank();

    // Design doc §5.1 distributed setup. Patches are already partitioned across
    // ranks (permanent). Within setup we perform a *temporary* hash partition:
    // each unique hash is routed to its owner rank, the owner assigns a
    // contiguous block of global IDs to its bucket, then each rank receives back
    // the IDs for the hashes it originally held. The hash partition is discarded;
    // the resulting patch CSR + inverted index stay partitioned by patch ID.

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

    // Step 2: sample-sort based partitioning. Every rank computes the same
    // (size-1) splitters from gathered samples, then routes each hash to its
    // target rank via splitter binary search. local_uniq is already sorted.
    const std::vector<Hash> splitters = compute_splitters(local_uniq, comm);

    std::vector<int>  sendcounts(static_cast<std::size_t>(size), 0);
    for (Hash h : local_uniq) ++sendcounts[target_rank_for_hash(h, splitters)];

    std::vector<int>  send_offsets(static_cast<std::size_t>(size), 0);
    for (int r = 1; r < size; ++r) send_offsets[r] = send_offsets[r - 1] + sendcounts[r - 1];

    std::vector<Hash> sendbuf(local_uniq.size());
    {
        std::vector<int> cursor = send_offsets;  // per-target write head
        for (Hash h : local_uniq) {
            const int target = target_rank_for_hash(h, splitters);
            sendbuf[cursor[target]++] = h;
        }
    }

    // Step 3: exchange sendcounts so every rank learns its own recv counts.
    // stComm has no plain allgather; emulate via allgatherv with uniform
    // recv length = size (each rank contributes its full sendcounts row).
    std::vector<int> all_sendcounts(static_cast<std::size_t>(size) * size);
    {
        std::vector<int> uniform(static_cast<std::size_t>(size), size);
        comm.allgatherv<stComm::Space::Host, int>(sendcounts.data(), size,
                                       all_sendcounts.data(), uniform.data())->wait();
    }
    std::vector<int> recvcounts(static_cast<std::size_t>(size), 0);
    for (int s = 0; s < size; ++s) recvcounts[s] = all_sendcounts[s * size + rank];

    // Step 4: alltoallv — every hash flies to its owner rank.
    const int total_recv = std::accumulate(recvcounts.begin(), recvcounts.end(), 0);
    std::vector<Hash> recvbuf(static_cast<std::size_t>(total_recv));
    comm.alltoallv<stComm::Space::Host, Hash>(sendbuf.data(), sendcounts.data(),
                                   recvbuf.data(), recvcounts.data())->wait();

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
    N_ = N_total;

    // Step 8: for every slot in recvbuf (arrival order, possibly with dupes),
    // produce the corresponding global ID. The reverse alltoallv expects responses
    // in the same slot order as recvbuf — that's how it routes back to senders.
    std::vector<std::uint64_t> id_response(static_cast<std::size_t>(total_recv));
    for (std::size_t i = 0; i < recvbuf.size(); ++i) {
        const auto it = std::lower_bound(bucket.begin(), bucket.end(), recvbuf[i]);
        id_response[i] = global_id_start + static_cast<std::uint64_t>(it - bucket.begin());
    }

    // Step 9: reverse alltoallv — IDs travel back to original senders. The
    // sendcounts and recvcounts arrays swap roles relative to Step 4.
    std::vector<std::uint64_t> id_back(sendbuf.size());
    comm.alltoallv<stComm::Space::Host, std::uint64_t>(id_response.data(), recvcounts.data(),
                                            id_back.data(),     sendcounts.data())->wait();

    // Step 10: sendbuf[i] ↔ id_back[i] are paired by construction. Build this
    // rank's hash → id map.
    std::unordered_map<Hash, ElementId> hash_to_id;
    hash_to_id.reserve(sendbuf.size() * 2);
    for (std::size_t i = 0; i < sendbuf.size(); ++i) {
        hash_to_id.emplace(sendbuf[i], static_cast<ElementId>(id_back[i]));
    }

    // Step 11: translate raw_patches into ID-space, sort + dedupe per patch. The
    // local patch index [0, M_local) is preserved in input order.
    std::vector<std::vector<ElementId>> id_patches(raw_patches.size());
    for (std::size_t p = 0; p < raw_patches.size(); ++p) {
        const auto& src = raw_patches[p];
        auto& dst = id_patches[p];
        dst.resize(src.size());
        for (std::size_t i = 0; i < src.size(); ++i) {
            dst[i] = hash_to_id.at(src[i]);
        }
        std::sort(dst.begin(), dst.end());
        dst.erase(std::unique(dst.begin(), dst.end()), dst.end());
    }

    patches_ = build_patch_csr(id_patches);
    inv_     = build_inverted_index(patches_, N_);

    // Mirror host state to device — consumed by the algorithm layers' kernels.
    const std::size_t patch_data_n = patches_.data.size();
    const std::size_t patch_offs_n = patches_.offsets.size();
    const std::size_t inv_keys_n   = inv_.keys.size();
    const std::size_t inv_offs_n   = inv_.offsets.size();
    const std::size_t inv_data_n   = inv_.data.size();

    d_patch_data_.resize(patch_data_n);
    d_patch_offsets_.resize(patch_offs_n);
    d_inv_keys_.resize(inv_keys_n);
    d_inv_offsets_.resize(inv_offs_n);
    d_inv_data_.resize(inv_data_n);

    if (patch_data_n) d_patch_data_.copy_from_host(patches_.data.data(),      patch_data_n);
    if (patch_offs_n) d_patch_offsets_.copy_from_host(patches_.offsets.data(), patch_offs_n);
    if (inv_keys_n)   d_inv_keys_.copy_from_host(inv_.keys.data(),            inv_keys_n);
    if (inv_offs_n)   d_inv_offsets_.copy_from_host(inv_.offsets.data(),      inv_offs_n);
    if (inv_data_n)   d_inv_data_.copy_from_host(inv_.data.data(),            inv_data_n);
}

}  // namespace stPS
