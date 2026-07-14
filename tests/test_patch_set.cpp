// Tests for PatchSet's element-shard hash retention (core/patch_set.hpp): the
// §5.1 setup keeps each rank's shard of the temporary hash partition as the
// ElementId → Hash reverse map. Because splitters route hashes to ranks in
// sorted order and each shard is itself sorted, concatenating the shards in
// rank order must reproduce the globally sorted unique hash list — which is
// exactly what the single-process reference (run_local_setup) computes.

#include "core/patch_set.hpp"
#include "data/synthetic.hpp"
#include "helpers/local_setup.hpp"
#include "test_nccl_env.hpp"

#include <stPS/stPS.h>
#include <stComm/stComm.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

using namespace stPS;

namespace {

std::vector<std::vector<Hash>> small_synthetic() {
    SyntheticParams p;
    p.N = 600; p.M = 50; p.K_mean = 10; p.overlap = 0.3; p.seed = 11;
    return generate_synthetic(p);
}

// Gather every rank's shard_hashes in rank order (variable-length allgatherv).
std::vector<Hash> gather_shards(stComm::Comm& comm, const std::vector<Hash>& shard) {
    const int size = comm.getSize();

    const std::uint64_t local_n = shard.size();
    std::vector<std::uint64_t> sizes(static_cast<std::size_t>(size));
    std::vector<int> ones(static_cast<std::size_t>(size), 1);
    comm.allgatherv<stComm::Space::Host, std::uint64_t>(&local_n, 1,
                                            sizes.data(), ones.data())->wait();

    std::vector<int> counts(static_cast<std::size_t>(size));
    std::uint64_t total = 0;
    for (int r = 0; r < size; ++r) { counts[r] = static_cast<int>(sizes[r]); total += sizes[r]; }

    std::vector<Hash> all(total);
    comm.allgatherv<stComm::Space::Host, Hash>(shard.data(), static_cast<int>(local_n),
                                   all.data(), counts.data())->wait();
    return all;
}

}  // namespace

TEST(PatchSetShard, ReverseMapIsGlobalSortedUniqueHashes) {
    auto& comm = test_helpers::comm();
    const auto full  = small_synthetic();
    auto slice = slice_patches_by_rank(full, comm.getRank(), comm.getSize());

    PatchSet ps(comm, std::move(slice.patches));

    // Reference: single-process setup over the full (deterministic) set.
    const auto ref = test_helpers::run_local_setup(full);
    ASSERT_EQ(ps.N(), ref.N);

    // Each shard is sorted, and shard_start is the exclusive prefix sum of
    // shard sizes in rank order (contiguous global ID blocks).
    const auto& shard = ps.shard_hashes();
    EXPECT_TRUE(std::is_sorted(shard.begin(), shard.end()));

    const auto all = gather_shards(comm, shard);
    ASSERT_EQ(all.size(), ref.id_to_hash.size());
    EXPECT_EQ(all, ref.id_to_hash);

    // shard_start: recompute this rank's block offset from the gathered sizes.
    std::uint64_t expect_start = 0;
    {
        // gather_shards already validated the concatenation; recompute offsets
        // locally from the reference by locating this shard's first hash.
        if (!shard.empty()) {
            const auto it = std::lower_bound(ref.id_to_hash.begin(), ref.id_to_hash.end(),
                                             shard.front());
            expect_start = static_cast<std::uint64_t>(it - ref.id_to_hash.begin());
        }
        if (shard.empty()) expect_start = ps.shard_start();  // no constraint
    }
    EXPECT_EQ(ps.shard_start(), expect_start);
}

TEST(PatchSetShard, ReverseMapTranslatesCsrBackToInputHashes) {
    auto& comm = test_helpers::comm();
    const auto full  = small_synthetic();
    auto slice = slice_patches_by_rank(full, comm.getRank(), comm.getSize());
    const auto raw = slice.patches;  // keep a copy; PatchSet consumes the input

    PatchSet ps(comm, std::move(slice.patches));
    const auto all = gather_shards(comm, ps.shard_hashes());  // global id → hash

    // Every CSR row, translated back through the reverse map, must equal the
    // sorted+deduped input patch (local patch order is preserved).
    ASSERT_EQ(ps.M_local(), raw.size());
    for (std::size_t p = 0; p < raw.size(); ++p) {
        std::vector<Hash> expect(raw[p]);
        std::sort(expect.begin(), expect.end());
        expect.erase(std::unique(expect.begin(), expect.end()), expect.end());

        const auto row = ps.patches().patch(static_cast<PatchId>(p));
        ASSERT_EQ(row.size(), expect.size()) << "patch " << p;
        for (std::size_t i = 0; i < row.size(); ++i) {
            ASSERT_LT(row[i], all.size());
            EXPECT_EQ(all[row[i]], expect[i]) << "patch " << p << " elem " << i;
        }
    }
}
