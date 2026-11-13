// The device hash → ID mapping must give exactly what the host mapping gives:
// the same IDs for every patch, the same N and the same shards. Both are
// collectives run on this job's ranks, so the comparison holds at any rank
// count.

#include "core/hash_id_map_device.hpp"
#include "core/patch_set.hpp"
#include "data/synthetic.hpp"
#include "helpers/distribute.hpp"
#include "helpers/test_scale.hpp"
#include "test_nccl_env.hpp"

#include <stPS/partition.hpp>
#include <stComm/stComm.h>

#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

using namespace stPS;

namespace {

void expect_same_as_host(const std::vector<std::vector<Hash>>& mine) {
    stComm::Comm& comm = test_helpers::comm();
    const IdMappedPatches host = map_hashes_to_element_ids(comm, mine);
    const DeviceIdMappedPatches dev = map_hashes_to_element_ids_device(comm, mine);

    EXPECT_EQ(dev.N, host.N);
    EXPECT_EQ(dev.shard_start, host.shard_start);
    EXPECT_EQ(dev.shard_hashes, host.shard_hashes);
    ASSERT_EQ(dev.M_local, host.id_patches.size());

    std::vector<std::uint64_t> offsets(dev.M_local + 1);
    dev.offsets.copy_to_host(offsets.data(), offsets.size());
    std::vector<ElementId> data(dev.data.size());
    if (!data.empty()) dev.data.copy_to_host(data.data(), data.size());

    ASSERT_EQ(offsets.front(), 0u);
    ASSERT_EQ(offsets.back(), data.size());
    for (std::size_t p = 0; p < host.id_patches.size(); ++p) {
        ASSERT_LE(offsets[p], offsets[p + 1]);
        const std::vector<ElementId> got(
            data.begin() + static_cast<std::ptrdiff_t>(offsets[p]),
            data.begin() + static_cast<std::ptrdiff_t>(offsets[p + 1]));
        ASSERT_EQ(got, host.id_patches[p]) << "patch " << p;
    }
}

std::vector<std::vector<Hash>> my_slice(const std::vector<std::vector<Hash>>& all) {
    stComm::Comm& comm = test_helpers::comm();
    return slice_patches_by_rank(all, comm.getRank(), comm.getSize()).patches;
}

}  // namespace

TEST(HashIdMapDevice, MatchesHostOnSynthetic) {
    SyntheticParams sp;
    sp.N = 20000; sp.M = test_helpers::scale_patch_count(3000); sp.K_mean = 24;
    sp.overlap = 0.5; sp.seed = 11;
    expect_same_as_host(my_slice(generate_synthetic(sp)));
}

TEST(HashIdMapDevice, EmptyInput) {
    expect_same_as_host({});
}

TEST(HashIdMapDevice, EmptyAndRepeatedHashesInsideAPatch) {
    const Hash big = std::numeric_limits<Hash>::max();
    const std::vector<std::vector<Hash>> all = {
        {},
        {7, 7, 7},
        {big, 0, big, 5, 0},
        {},
        {5},
        {9, 3, 9, 3, 1},
        {},
    };
    expect_same_as_host(my_slice(all));
}

TEST(HashIdMapDevice, AllPatchesOnOneRank) {
    SyntheticParams sp;
    sp.N = 3000; sp.M = 400; sp.K_mean = 16; sp.overlap = 0.4; sp.seed = 5;
    const auto all = generate_synthetic(sp);
    stComm::Comm& comm = test_helpers::comm();
    expect_same_as_host(
        test_helpers::slice_all_on_rank(all, {}, comm.getSize() - 1, comm.getRank()).patches);
}

TEST(HashIdMapDevice, FewDistinctHashes) {
    // Far fewer distinct hashes than splitter samples.
    std::vector<std::vector<Hash>> all(200);
    for (std::size_t p = 0; p < all.size(); ++p) {
        all[p] = { Hash{p % 3}, Hash{(p + 1) % 3}, Hash{1000} };
    }
    expect_same_as_host(my_slice(all));
}
