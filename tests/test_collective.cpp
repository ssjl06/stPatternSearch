// Tests for the collective agreement helpers (core/collective.hpp) and the
// shared split rule (stPS/partition.hpp).

#include "core/collective.hpp"
#include "test_nccl_env.hpp"

#include <stPS/partition.hpp>
#include <stComm/stComm.h>

#include <gtest/gtest.h>

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
    const auto rank = static_cast<std::size_t>(comm.getRank());

    // Rank s sends (s + 1) * (r + 2) elements to rank r — plus a count far
    // past INT_MAX on the diagonal: counts are 64-bit end to end, never capped.
    const std::size_t huge = (std::size_t{1} << 40) + rank;
    std::vector<std::size_t> send(size);
    for (std::size_t r = 0; r < size; ++r) send[r] = (rank + 1) * (r + 2);
    send[rank] = huge;

    const ExchangeCounts counts = exchange_counts(comm, send);
    ASSERT_EQ(counts.send.size(), size);
    ASSERT_EQ(counts.recv.size(), size);
    for (std::size_t r = 0; r < size; ++r) {
        if (r == rank) continue;
        EXPECT_EQ(counts.send[r], (rank + 1) * (r + 2));
        EXPECT_EQ(counts.recv[r], (r + 1) * (rank + 2));
    }
    EXPECT_EQ(counts.send[rank], huge);
    EXPECT_EQ(counts.recv[rank], huge);
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
