#include <stPS/stPS.h>          // public API under test: UscPatchSelector, slice, types
#include "usc/brute_force.hpp"  // white-box reference (brute_force_select)
#include "data/synthetic.hpp"
#include "helpers/chunk_guard.hpp"
#include "helpers/distribute.hpp"
#include "helpers/lazy_greedy.hpp"
#include "helpers/local_setup.hpp"
#include "helpers/test_scale.hpp"
#include "test_nccl_env.hpp"

#include <stComm/stComm.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace stPS;

namespace {

using Patches = std::vector<std::vector<Hash>>;

// brute_force_select is the obviously-correct reference; lazy_greedy_select
// gives the same answer fast enough for the sizes brute force cannot reach
// (LazyGreedy.MatchesBruteForce ties the two together).
enum class Reference { BruteForce, LazyGreedy };

void expect_match(const PatchSelection& selection, const PatchSelection& ref,
                  const std::string& tag) {
    EXPECT_EQ(selection.selected.size(), ref.selected.size())
        << "[" << tag << "] selected size mismatch";
    EXPECT_EQ(selection.covered_count, ref.covered_count)
        << "[" << tag << "] covered_count mismatch";
    const std::size_t n = std::min(selection.selected.size(), ref.selected.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (selection.selected[i] != ref.selected[i]) {
            ADD_FAILURE() << "[" << tag << "] first mismatch at step " << i << ": got "
                          << selection.selected[i] << ", want " << ref.selected[i];
            break;  // everything after the first divergence differs too
        }
    }
}

std::string make_tag(const std::string& what, stComm::Comm& comm, PartitionMode mode) {
    return what + " ranks=" + std::to_string(comm.getSize()) +
           " mode=" + (mode == PartitionMode::ByElement ? "element" : "patch");
}

// Distributed path: this rank hands `mine` (its share of `full`, with the
// global IDs = indices into `full`) to the public UscPatchSelector on the
// shared device-enabled Comm.
//
// Reference path: every rank runs the single-process setup + reference solver
// on the full data. Identical work on every rank, and independent of the rank
// count — so passing at several rank counts shows the result does not depend
// on how the patches were distributed.
//
// `full` must list the patches in rank-major order (rank 0's patches in the
// order it passes them, then rank 1's, ...): that position is what breaks
// ties. `position_to_gid`, when given, maps such a position to the global ID
// the owning rank attached; without it the position is the global ID.
void check_solve(const Patches& full, PatchSlice mine, PartitionMode mode,
                 Reference reference, const std::string& what,
                 const std::vector<PatchId>* position_to_gid = nullptr) {
    stComm::Comm& comm = test_helpers::comm();
    const std::string tag = make_tag(what, comm, mode);

    UscPatchSelector selector(comm, mode);
    const auto multi = selector.patch_select(std::move(mine.patches), std::move(mine.global_ids));

    const auto local = test_helpers::run_local_setup(full);
    auto ref = reference == Reference::BruteForce
        ? brute_force_select(local.patches, local.N)
        : test_helpers::lazy_greedy_select(local.patches, local.N);
    if (position_to_gid != nullptr) {
        for (PatchId& p : ref.selected) p = (*position_to_gid)[p];
    }
    expect_match(multi, ref, tag);

    EXPECT_EQ(multi.covered_count, local.N) << "[" << tag << "] coverage incomplete";
    EXPECT_EQ(multi.iterations, multi.selected.size()) << "[" << tag << "]";
    std::vector<PatchId> sorted_sel = multi.selected;
    std::sort(sorted_sel.begin(), sorted_sel.end());
    const auto uniq_end = std::unique(sorted_sel.begin(), sorted_sel.end());
    EXPECT_EQ(uniq_end, sorted_sel.end()) << "[" << tag << "] duplicate selection";
}

Patches synthetic(std::uint64_t N, std::uint64_t M, std::uint32_t K,
                  double overlap, std::uint64_t seed) {
    SyntheticParams params;
    params.N = N; params.M = M; params.K_mean = K; params.overlap = overlap; params.seed = seed;
    return generate_synthetic(params);
}

PatchSlice contiguous_slice(const Patches& full) {
    stComm::Comm& comm = test_helpers::comm();
    return slice_patches_by_rank(full, comm.getRank(), comm.getSize());
}

void run_case(std::uint64_t N, std::uint64_t M, std::uint32_t K,
              double overlap, std::uint64_t seed,
              PartitionMode mode = PartitionMode::ByPatch) {
    char what[128];
    std::snprintf(what, sizeof(what), "N=%lu M=%lu K=%u overlap=%.2f seed=%lu",
                  N, M, K, overlap, seed);
    const auto full = synthetic(N, M, K, overlap, seed);
    check_solve(full, contiguous_slice(full), mode, Reference::BruteForce, what);
}

constexpr PartitionMode kBothModes[] = {PartitionMode::ByPatch, PartitionMode::ByElement};

}  // namespace

TEST(SolverEquivalence, SmallLowOverlap) {
    for (std::uint64_t seed : {1u, 2u, 3u}) run_case(100, 50, 10, 0.3, seed);
}

TEST(SolverEquivalence, MediumLowOverlap) {
    for (std::uint64_t seed : {1u, 2u, 3u}) run_case(1000, 200, 30, 0.1, seed);
}

TEST(SolverEquivalence, MediumHighOverlap) {
    for (std::uint64_t seed : {1u, 2u, 3u}) run_case(1000, 200, 30, 0.7, seed);
}

TEST(SolverEquivalence, LargerMidOverlap) {
    run_case(10000, 500, 50, 0.5, 1);
}

// PartitionMode::ByElement (§7.3) must select the exact same cover as the
// brute-force reference (and therefore as ByPatch) on every workload shape.
TEST(SolverEquivalenceElem, SmallLowOverlap) {
    for (std::uint64_t seed : {1u, 2u, 3u}) run_case(100, 50, 10, 0.3, seed, PartitionMode::ByElement);
}

TEST(SolverEquivalenceElem, MediumLowOverlap) {
    for (std::uint64_t seed : {1u, 2u, 3u}) run_case(1000, 200, 30, 0.1, seed, PartitionMode::ByElement);
}

TEST(SolverEquivalenceElem, MediumHighOverlap) {
    for (std::uint64_t seed : {1u, 2u, 3u}) run_case(1000, 200, 30, 0.7, seed, PartitionMode::ByElement);
}

TEST(SolverEquivalenceElem, LargerMidOverlap) {
    run_case(10000, 500, 50, 0.5, 1, PartitionMode::ByElement);
}

TEST(UscPatchSelectorApi, RejectsGlobalIdCountMismatch) {
    // Every rank passes one ID short, so all throw before any collective.
    stComm::Comm& comm = test_helpers::comm();
    for (PartitionMode mode : {PartitionMode::ByPatch, PartitionMode::ByElement}) {
        UscPatchSelector selector(comm, mode);
        std::vector<std::vector<Hash>> patches = {{1, 2}, {2, 3}};
        std::vector<PatchId>           ids     = {0};
        EXPECT_THROW(selector.patch_select(std::move(patches), std::move(ids)),
                     std::invalid_argument);
    }
}

// ---- Reference cross-check ---------------------------------------------------

TEST(LazyGreedy, MatchesBruteForce) {
    // Host-only: the fast reference used at scale must agree with brute force,
    // including the tie rule (high overlap produces many equal scores).
    for (double overlap : {0.1, 0.5, 0.9}) {
        for (std::uint64_t seed : {1u, 2u, 3u}) {
            const auto local = test_helpers::run_local_setup(synthetic(2000, 300, 25, overlap, seed));
            const auto bf   = brute_force_select(local.patches, local.N);
            const auto lazy = test_helpers::lazy_greedy_select(local.patches, local.N);
            EXPECT_EQ(lazy.selected, bf.selected) << "overlap=" << overlap << " seed=" << seed;
            EXPECT_EQ(lazy.covered_count, bf.covered_count);
            EXPECT_EQ(lazy.iterations, bf.iterations);
        }
    }
}

// ---- Uneven distributions, degenerate inputs, scale -------------------------

TEST(SolverEdge, AllPatchesOnOneRank) {
    // Every other rank has no patches at all, yet owns an element shard and
    // takes part in every collective.
    stComm::Comm& comm = test_helpers::comm();
    const auto full = synthetic(1000, 200, 30, 0.5, 7);
    for (PartitionMode mode : kBothModes) {
        for (int owner : {0, comm.getSize() - 1}) {
            check_solve(full, test_helpers::slice_all_on_rank(full, {}, owner, comm.getRank()),
                        mode, Reference::BruteForce,
                        "all patches on rank " + std::to_string(owner));
        }
    }
}

TEST(SolverEdge, RoundRobinGlobalIds) {
    // Each rank's global IDs interleave with the others' instead of forming a
    // contiguous block. Ties go to the lowest rank, then the earliest patch on
    // it — not to the smaller global ID — so the reference runs on the patches
    // laid out in that rank-major order, and its picks are mapped back to the
    // IDs the ranks attached.
    stComm::Comm& comm = test_helpers::comm();
    const auto full = synthetic(1000, 200, 30, 0.7, 11);

    Patches rank_major;
    std::vector<PatchId> position_to_gid;
    for (int r = 0; r < comm.getSize(); ++r) {
        const auto part = test_helpers::slice_round_robin(full, {}, r, comm.getSize());
        rank_major.insert(rank_major.end(), part.patches.begin(), part.patches.end());
        position_to_gid.insert(position_to_gid.end(),
                               part.global_ids.begin(), part.global_ids.end());
    }
    for (PartitionMode mode : kBothModes) {
        check_solve(rank_major,
                    test_helpers::slice_round_robin(full, {}, comm.getRank(), comm.getSize()),
                    mode, Reference::BruteForce, "round-robin ids", &position_to_gid);
    }
}

TEST(SolverEdge, FewerPatchesThanRanks) {
    const Patches full = {{3, 1, 4, 1, 5}};
    for (PartitionMode mode : kBothModes) {
        check_solve(full, contiguous_slice(full), mode, Reference::BruteForce, "single patch");
    }
}

TEST(SolverEdge, EmptyInput) {
    for (PartitionMode mode : kBothModes) {
        check_solve({}, PatchSlice{}, mode, Reference::BruteForce, "no patches");
    }
}

TEST(SolverEdge, DegeneratePatches) {
    // Empty patches, repeated hashes inside a patch, identical patches (pure
    // ties), a subset that never gets picked, and one patch covering a lot.
    Patches full = {
        {},
        {10, 20, 30},
        {10, 20, 30},              // identical to patch 1: the tie goes to 1
        {20, 20, 20, 20},          // one distinct element, repeated
        {10, 30},                  // subset of patch 1, never needed
        {},
        {40, 50, 60, 70, 80, 90},  // biggest, picked first
        {90, 100},
        {100},
    };
    for (PartitionMode mode : kBothModes) {
        check_solve(full, contiguous_slice(full), mode, Reference::BruteForce, "degenerate");
    }
}

TEST(SolverEdge, FewDistinctHashes) {
    // Hundreds of patches over a handful of hashes: the sample-sort splitters
    // collapse and most ranks own an empty element shard.
    Patches full;
    for (Hash p = 0; p < 300; ++p) full.push_back({7, 7 + p % 4, 1000 + p % 3});
    for (PartitionMode mode : kBothModes) {
        check_solve(full, contiguous_slice(full), mode, Reference::BruteForce, "few distinct");
    }
}

TEST(SolverEdge, SmallExchangeChunks) {
    // Host exchanges split into many tiny point-to-point chunks (odd size, so
    // boundaries fall inside elements).
    const test_helpers::ChunkBytesGuard guard(29);
    const auto full = synthetic(1000, 200, 30, 0.5, 13);
    for (PartitionMode mode : kBothModes) {
        check_solve(full, contiguous_slice(full), mode, Reference::BruteForce, "small chunks");
    }
}

TEST(SolverScale, MatchesLazyGreedy) {
    // Size from STPS_TEST_SCALE_M (see helpers/test_scale.hpp). The exact
    // selection sequence is compared, not just the cover properties.
    const std::uint64_t M = test_helpers::scale_patch_count(20000);
    const auto full = synthetic(M * 8, M, 32, 0.5, 17);
    for (PartitionMode mode : kBothModes) {
        check_solve(full, contiguous_slice(full), mode, Reference::LazyGreedy,
                    "scale M=" + std::to_string(M));
    }
}
