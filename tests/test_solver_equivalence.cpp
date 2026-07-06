#include <stPS/stPS.h>          // public API under test: UscPatchSelector, slice, types
#include "usc/brute_force.hpp"  // white-box reference (brute_force_select)
#include "data/synthetic.hpp"
#include "helpers/local_setup.hpp"
#include "test_nccl_env.hpp"

#include <stComm/stComm.h>

#include <gtest/gtest.h>

#include <cstdio>

using namespace stPS;

namespace {

void expect_match(const PatchSelection& selection, const PatchSelection& bf,
                  const std::string& tag) {
    EXPECT_EQ(selection.selected.size(), bf.selected.size())
        << "[" << tag << "] selected size mismatch";
    EXPECT_EQ(selection.covered_count, bf.covered_count)
        << "[" << tag << "] covered_count mismatch";
    const std::size_t n = std::min(selection.selected.size(), bf.selected.size());
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(selection.selected[i], bf.selected[i])
            << "[" << tag << "] mismatch at step " << i;
    }
}

void run_case(std::uint64_t N, std::uint64_t M, std::uint32_t K,
              double overlap, std::uint64_t seed) {
    SyntheticParams params;
    params.N = N; params.M = M; params.K_mean = K; params.overlap = overlap; params.seed = seed;

    // Distributed path: every rank gets its slice and runs the public
    // UscPatchSelector on the shared device-enabled Comm — one patch_select()
    // does load + setup + select.
    stComm::Comm& comm = test_helpers::comm();
    UscPatchSelector selector(comm);
    auto raw_full = generate_synthetic(params);
    auto slice    = slice_patches_by_rank(raw_full, comm.getRank(), comm.getSize());
    const auto multi = selector.patch_select(std::move(slice.patches), std::move(slice.global_ids));

    // Reference path: every rank runs single-process setup + brute_force on the
    // full deterministic data. Identical work on every rank, used only as ground
    // truth for the comparison below.
    auto local = test_helpers::run_local_setup(raw_full);
    auto bf    = brute_force_select(local.patches, local.N);

    char tag[128];
    std::snprintf(tag, sizeof(tag),
                  "N=%lu M=%lu K=%u overlap=%.2f seed=%lu ranks=%d",
                  (unsigned long)N, (unsigned long)M, (unsigned)K, overlap,
                  (unsigned long)seed, comm.getSize());
    expect_match(multi, bf, tag);

    EXPECT_EQ(multi.covered_count, local.N) << "[" << tag << "] coverage incomplete";
    std::vector<PatchId> sorted_sel = multi.selected;
    std::sort(sorted_sel.begin(), sorted_sel.end());
    const auto uniq_end = std::unique(sorted_sel.begin(), sorted_sel.end());
    EXPECT_EQ(uniq_end, sorted_sel.end()) << "[" << tag << "] duplicate selection";
}

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
