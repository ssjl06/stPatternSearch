#include "core/usc_solver.hpp"

#include "core/bitset.hpp"

#include <stComm/stComm.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <unordered_map>

namespace fullchipusc {

namespace {

// Distributed sample sort splitter selection. Every rank computes the same
// (size-1) splitter values from gathered samples; afterwards each hash is
// routed to its target rank via upper_bound over splitters. This guarantees
// balanced bucket sizes regardless of hash distribution (unlike `hash % size`
// which assumes uniform distribution).
//
// Returns a sorted vector of size (size-1) splitter Hash values. For size == 1
// returns empty (no partitioning needed).
template<typename CommT>
std::vector<Hash> compute_splitters(const std::vector<Hash>& local_uniq_sorted,
                                    CommT& comm) {
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
    comm.template allgatherv<Hash>(local_sample.data(), kSamplesPerRank,
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

template<typename CommT>
USCSolver<CommT>::USCSolver(CommT& comm) : comm_(comm) {}

template<typename CommT>
void USCSolver<CommT>::load(std::vector<std::vector<Hash>> raw_patches,
                            std::vector<PatchId>           patch_global_ids) {
    raw_patches_      = std::move(raw_patches);
    patch_global_ids_ = std::move(patch_global_ids);
}

template<typename CommT>
void USCSolver<CommT>::setup() {
    const int size = comm_.getSize();
    const int rank = comm_.getRank();

    // Stage 2 (M3) — design doc §5.1 distributed setup, hash % size partitioning.
    //
    // Patches are already partitioned across ranks (permanent). Within setup we
    // perform a *temporary* hash partition: each unique hash is routed to its
    // owner rank (owner = hash % size), the owner assigns a contiguous block of
    // global IDs to its bucket, then each rank receives back the IDs for the
    // hashes it originally held. The hash partition is discarded; solve() uses
    // only the resulting patch CSR + inverted index, both still partitioned by
    // patch ID. See [[reference-design-doc]] §5.2 for the two-partition model.

    // Step 1: flatten + dedupe this rank's local hash set.
    std::vector<Hash> local_uniq;
    {
        std::size_t total = 0;
        for (const auto& p : raw_patches_) total += p.size();
        local_uniq.reserve(total);
        for (const auto& p : raw_patches_) {
            local_uniq.insert(local_uniq.end(), p.begin(), p.end());
        }
        std::sort(local_uniq.begin(), local_uniq.end());
        local_uniq.erase(std::unique(local_uniq.begin(), local_uniq.end()), local_uniq.end());
    }

    // Step 2: sample-sort based partitioning. Every rank computes the same
    // (size-1) splitters from gathered samples, then routes each hash to its
    // target rank via splitter binary search. This balances buckets regardless
    // of hash distribution. local_uniq is already sorted from Step 1.
    const std::vector<Hash> splitters = compute_splitters(local_uniq, comm_);

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
        comm_.template allgatherv<int>(sendcounts.data(), size,
                                       all_sendcounts.data(), uniform.data())->wait();
    }
    std::vector<int> recvcounts(static_cast<std::size_t>(size), 0);
    for (int s = 0; s < size; ++s) recvcounts[s] = all_sendcounts[s * size + rank];

    // Step 4: alltoallv — every hash flies to its owner rank.
    const int total_recv = std::accumulate(recvcounts.begin(), recvcounts.end(), 0);
    std::vector<Hash> recvbuf(static_cast<std::size_t>(total_recv));
    comm_.template alltoallv<Hash>(sendbuf.data(), sendcounts.data(),
                                   recvbuf.data(), recvcounts.data())->wait();

    // Step 5: form this rank's bucket universe — sorted unique view of recvbuf.
    // Keep recvbuf intact for the per-arrival-slot lookup in Step 8.
    std::vector<Hash> bucket(recvbuf);
    std::sort(bucket.begin(), bucket.end());
    bucket.erase(std::unique(bucket.begin(), bucket.end()), bucket.end());

    // Step 6: Exscan over bucket sizes → this rank's first global ID.
    const std::uint64_t local_shard_size = bucket.size();
    const std::uint64_t global_id_start  =
        comm_.template exscan<std::uint64_t>(local_shard_size, MPI_SUM);

    // Step 7: only the last rank knows N at this point; broadcast it.
    std::uint64_t N_total = (rank == size - 1) ? (global_id_start + local_shard_size) : 0;
    comm_.template bcast<std::uint64_t>(&N_total, 1, size - 1)->wait();
    N_ = N_total;

    // Step 8: for every slot in recvbuf (in arrival order, possibly with dupes),
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
    comm_.template alltoallv<std::uint64_t>(id_response.data(), recvcounts.data(),
                                            id_back.data(),     sendcounts.data())->wait();

    // Step 10: sendbuf[i] ↔ id_back[i] are paired by construction. Build this
    // rank's hash → id map.
    std::unordered_map<Hash, ElementId> hash_to_id;
    hash_to_id.reserve(sendbuf.size() * 2);
    for (std::size_t i = 0; i < sendbuf.size(); ++i) {
        hash_to_id.emplace(sendbuf[i], static_cast<ElementId>(id_back[i]));
    }

    // Step 11: translate raw_patches_ into ID-space, sort + dedupe per patch.
    std::vector<std::vector<ElementId>> id_patches(raw_patches_.size());
    for (std::size_t p = 0; p < raw_patches_.size(); ++p) {
        const auto& src = raw_patches_[p];
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

    // id_to_hash_ used to hold the full universe (Stage 1). With distributed
    // setup it can only meaningfully hold this rank's bucket — keep it for
    // optional debug inspection, but no algorithm path depends on it.
    id_to_hash_ = std::move(bucket);

    raw_patches_.clear();
    raw_patches_.shrink_to_fit();
}

template<typename CommT>
SolverResult USCSolver<CommT>::solve() {
    const std::uint64_t M_loc = patches_.M();
    SolverResult result;
    result.covered_count = 0;
    result.iterations    = 0;

    std::vector<Score> scores(M_loc);
    for (std::uint64_t p = 0; p < M_loc; ++p) {
        scores[p] = static_cast<Score>(patches_.patch_size(p));
    }

    DenseBitset covered(N_);
    DenseBitset nc_mask(N_);
    std::vector<ElementId>    newly_covered_ids;
    std::vector<std::uint8_t> dirty(M_loc, 0);
    std::vector<PatchId>      affected;

    while (result.covered_count < N_) {
        Score   local_best_score = kDisabledScore;
        PatchId local_best_patch = 0;
        for (std::uint64_t p = 0; p < M_loc; ++p) {
            if (scores[p] > local_best_score) {
                local_best_score = scores[p];
                local_best_patch = static_cast<PatchId>(p);
            }
        }

        auto maxloc = comm_.template allreduceMaxloc<std::int64_t>(local_best_score);
        const std::int64_t winner_score = maxloc.first;
        const int          winner_rank  = maxloc.second;
        if (winner_score <= 0) break;

        PatchId winner_global = 0;
        if (comm_.getRank() == winner_rank) {
            winner_global = patch_global_ids_[local_best_patch];
        }
        comm_.template bcast<PatchId>(&winner_global, 1, winner_rank)->wait();

        newly_covered_ids.resize(static_cast<std::size_t>(winner_score));
        if (comm_.getRank() == winner_rank) {
            std::size_t i = 0;
            for (ElementId e : patches_.patch(local_best_patch)) {
                if (!covered.test(e)) newly_covered_ids[i++] = e;
            }
        }
        comm_.template bcast<ElementId>(newly_covered_ids.data(),
                                        static_cast<std::size_t>(winner_score),
                                        winner_rank)->wait();

        for (ElementId e : newly_covered_ids) covered.set(e);
        result.covered_count += static_cast<std::uint64_t>(winner_score);
        result.selected.push_back(winner_global);

        nc_mask.clear();
        for (ElementId e : newly_covered_ids) nc_mask.set(e);

        affected.clear();
        for (ElementId e : newly_covered_ids) {
            for (PatchId q : inv_.patches_of(e)) {
                if (!dirty[q]) { dirty[q] = 1; affected.push_back(q); }
            }
        }
        for (PatchId q : affected) {
            dirty[q] = 0;
            if (scores[q] <= 0) continue;
            const auto delta = nc_mask.popcount_intersect_with_ids(patches_.patch(q));
            scores[q] -= static_cast<Score>(delta);
        }

        if (comm_.getRank() == winner_rank) {
            scores[local_best_patch] = kDisabledScore;
        }

        ++result.iterations;
    }

    return result;
}

template<typename CommT>
void USCSolver<CommT>::print_solution(const SolverResult& r) const {
    if (comm_.getRank() != 0) return;
    std::cout << "fullchipUSC solve\n"
              << "  universe N=" << N_
              << " ranks=" << comm_.getSize() << "\n"
              << "  result: selected=" << r.selected.size()
              << " covered=" << r.covered_count << "/" << N_
              << " iterations=" << r.iterations << "\n";
}

template<typename CommT> std::uint64_t        USCSolver<CommT>::N() const            { return N_; }
template<typename CommT> std::uint64_t        USCSolver<CommT>::M_local() const      { return patches_.M(); }
template<typename CommT> const PatchCsr&      USCSolver<CommT>::patches() const      { return patches_; }
template<typename CommT> const InvertedIndex& USCSolver<CommT>::inverted_index() const { return inv_; }

// Explicit instantiation for the M2 backend.
template class USCSolver<stComm::MPIComm>;

}  // namespace fullchipusc
