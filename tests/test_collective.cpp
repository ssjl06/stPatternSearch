// Tests for the collective agreement helpers (core/collective.hpp) and the
// shared split rule (stPS/partition.hpp).

#include "core/collective.hpp"
#include "test_nccl_env.hpp"

#include <stPS/partition.hpp>
#include <stComm/stComm.h>

#include <gtest/gtest.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace stPS;

TEST(Collective, AllRanksOkNeedsEveryRank) {
    stComm::Comm& comm = test_helpers::comm();
    EXPECT_TRUE(all_ranks_ok(comm, true));
    // One dissenting rank is enough — and every rank sees the same verdict.
    EXPECT_FALSE(all_ranks_ok(comm, comm.getRank() != 0));
}

TEST(Collective, ExchangeCountsMirrorSendAndRecv) {
    stComm::Comm& comm = test_helpers::comm();
    const auto size = static_cast<std::size_t>(comm.getSize());
    const auto rank = static_cast<std::uint64_t>(comm.getRank());

    // Rank s sends (s + 1) * (r + 2) elements to rank r.
    std::vector<std::uint64_t> send(size);
    for (std::size_t r = 0; r < size; ++r) send[r] = (rank + 1) * (r + 2);

    const ExchangeCounts counts = agree_exchange_counts(comm, send, "test exchange");
    ASSERT_EQ(counts.send.size(), size);
    ASSERT_EQ(counts.recv.size(), size);
    for (std::size_t r = 0; r < size; ++r) {
        EXPECT_EQ(counts.send[r], static_cast<int>((rank + 1) * (r + 2)));
        EXPECT_EQ(counts.recv[r], static_cast<int>((r + 1) * (rank + 2)));
    }
}

TEST(Collective, ExchangeCountsRejectOversizeOnEveryRank) {
    stComm::Comm& comm = test_helpers::comm();
    const auto size = static_cast<std::size_t>(comm.getSize());

    // Only rank 0 is over the limit; the verdict comes from the gathered
    // matrix, so every rank must throw — none may be left to enter the
    // exchange alone.
    std::vector<std::uint64_t> send(size, 1);
    if (comm.getRank() == 0) send[0] = static_cast<std::uint64_t>(INT_MAX) + 1;
    EXPECT_THROW(agree_exchange_counts(comm, send, "test exchange"), std::length_error);

    // A custom (smaller) limit applies to the per-rank totals, both ways.
    std::vector<std::uint64_t> small(size, 4);
    EXPECT_THROW(agree_exchange_counts(comm, small, "test exchange", 4 * size - 1),
                 std::length_error);
    EXPECT_NO_THROW(agree_exchange_counts(comm, small, "test exchange", 4 * size));

    // A count that would wrap a 64-bit running sum is still caught.
    std::vector<std::uint64_t> wrap(size, UINT64_MAX);
    EXPECT_THROW(agree_exchange_counts(comm, wrap, "test exchange"), std::length_error);
}

TEST(Partition, RankRangeTilesTheList) {
    for (std::uint64_t count : {0u, 1u, 7u, 100u}) {
        for (int size : {1, 2, 3, 8}) {
            std::uint64_t expect_begin = 0;
            for (int r = 0; r < size; ++r) {
                const RankRange range = rank_range(count, r, size);
                EXPECT_EQ(range.begin, expect_begin);
                EXPECT_LE(range.begin, range.end);
                expect_begin = range.end;
            }
            EXPECT_EQ(expect_begin, count);
        }
    }
    EXPECT_THROW(rank_range(10, 2, 2), std::invalid_argument);
    EXPECT_THROW(rank_range(10, -1, 2), std::invalid_argument);
    EXPECT_THROW(rank_range(10, 0, 0), std::invalid_argument);
}

TEST(Partition, SliceWithCoordsRejectsShapeMismatch) {
    // Used to be an assert: in a release build the short coords vector was
    // read out of bounds.
    std::vector<std::vector<Hash>>  patches = {{1, 2}, {3}, {4, 5}};
    std::vector<std::vector<Point>> coords  = {{{0, 0}, {1, 1}}, {{2, 2}}};
    EXPECT_THROW(slice_patches_by_rank(patches, coords, 0, 1), std::invalid_argument);

    coords.push_back({{3, 3}, {4, 4}});
    const PatchSlice slice = slice_patches_by_rank(patches, coords, 1, 3);
    ASSERT_EQ(slice.patches.size(), 1u);
    EXPECT_EQ(slice.patches[0], (std::vector<Hash>{3}));
    ASSERT_EQ(slice.coords.size(), 1u);
    EXPECT_EQ(slice.global_ids, (std::vector<PatchId>{1}));
}
