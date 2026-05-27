#include "core/brute_force.hpp"
#include "core/csr.hpp"

#include <gtest/gtest.h>

using namespace fullchipusc;

TEST(BruteForce, TrivialFullCover) {
    // One patch covers everything.
    auto csr = build_patch_csr({{0, 1, 2, 3, 4}});
    auto r = solve_brute_force(csr, 5);
    EXPECT_EQ(r.selected.size(), 1u);
    EXPECT_EQ(r.selected[0], 0u);
    EXPECT_EQ(r.covered_count, 5u);
}

TEST(BruteForce, TwoDisjointPatches) {
    auto csr = build_patch_csr({{0, 1}, {2, 3}});
    auto r = solve_brute_force(csr, 4);
    EXPECT_EQ(r.selected.size(), 2u);
    EXPECT_EQ(r.covered_count, 4u);
    // Tie on initial scores (both 2) — smaller PatchId wins first.
    EXPECT_EQ(r.selected[0], 0u);
    EXPECT_EQ(r.selected[1], 1u);
}

TEST(BruteForce, GreedyPicksLargestFirst) {
    // p0 covers {0,1}, p1 covers {0,1,2,3,4}, p2 covers {4,5}
    // Greedy: pick p1 first (covers 5), then p2 (covers 4,5 → 5 already covered → adds 5)
    // After p1: covered={0..4}. p2 \ covered = {5}, score=1. p0 \ covered = {}, score=0.
    auto csr = build_patch_csr({{0, 1}, {0, 1, 2, 3, 4}, {4, 5}});
    auto r = solve_brute_force(csr, 6);
    EXPECT_EQ(r.selected.size(), 2u);
    EXPECT_EQ(r.selected[0], 1u);
    EXPECT_EQ(r.selected[1], 2u);
    EXPECT_EQ(r.covered_count, 6u);
}

TEST(BruteForce, NoCoverPossibleWhenPatchesEmpty) {
    auto csr = build_patch_csr({{}, {}});
    auto r = solve_brute_force(csr, 0);
    EXPECT_EQ(r.selected.size(), 0u);
    EXPECT_EQ(r.covered_count, 0u);
}
