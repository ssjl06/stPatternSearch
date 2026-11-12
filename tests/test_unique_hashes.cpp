// Tests for the unique-hash file (public stPS::write_unique_hashes_file): the
// distributed hash sort (design doc §5.1) plus the parallel write. Host-only —
// nothing here touches a GPU, so this file is also built into
// stps_host_tests, which runs at several rank counts on machines (and CI
// runners) that have no GPU at all.
//
// Every case is checked against a single-process reference (sort + unique over
// the full input), which does not depend on the rank count: passing at 1, 2, 4
// and 8 ranks is what shows the output is the same for any of them.

#include "data/synthetic.hpp"
#include "helpers/chunk_guard.hpp"
#include "helpers/distribute.hpp"
#include "helpers/test_scale.hpp"
#include "helpers/unique_hash_file.hpp"
#include "test_nccl_env.hpp"

#include <stPS/stPS.h>
#include <stComm/stComm.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace stPS;
using test_helpers::read_unique_hash_file;

namespace {

using Patches = std::vector<std::vector<Hash>>;

Patches synthetic(std::uint64_t M, std::uint64_t seed) {
    SyntheticParams p;
    p.N = M * 8; p.M = M; p.K_mean = 15; p.overlap = 0.5; p.seed = seed;
    return generate_synthetic(p);
}

std::vector<Hash> all_unique(const Patches& patches) {
    std::vector<Hash> all;
    for (const auto& p : patches) all.insert(all.end(), p.begin(), p.end());
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    return all;
}

// ctest runs the same binary at several rank counts, possibly in parallel:
// keep their files apart.
std::string temp_path(stComm::Comm& comm, const std::string& name) {
    return ::testing::TempDir() + name + "_np" + std::to_string(comm.getSize()) + ".bin";
}

// Write `mine` (this rank's share of `full`) collectively, then have every
// rank read the file back and compare it with the reference over `full`.
void expect_file_matches(stComm::Comm& comm, const std::string& name,
                         const Patches& full, Patches mine) {
    const auto want = all_unique(full);
    const std::string path = temp_path(comm, name);

    const std::uint64_t n = write_unique_hashes_file(comm, path, std::move(mine));
    EXPECT_EQ(n, want.size());

    const auto [n_field, hashes] = read_unique_hash_file(path);
    EXPECT_EQ(n_field, want.size());
    EXPECT_TRUE(hashes == want) << name << ": file content differs from the reference ("
                                << hashes.size() << " vs " << want.size() << " hashes)";

    comm.barrier();  // no rank may still be reading when rank 0 unlinks
    if (comm.getRank() == 0) std::remove(path.c_str());
}

Patches contiguous_slice(stComm::Comm& comm, const Patches& full) {
    return slice_patches_by_rank(full, comm.getRank(), comm.getSize()).patches;
}

}  // namespace

TEST(UniqueHashes, FileHoldsEverySortedDistinctHash) {
    auto& comm = test_helpers::comm();
    const auto full = synthetic(60, 59);
    expect_file_matches(comm, "unique_hashes_shared", full, contiguous_slice(comm, full));
}

TEST(UniqueHashes, EmptyInputWritesHeaderOnly) {
    auto& comm = test_helpers::comm();
    expect_file_matches(comm, "unique_hashes_empty", {}, {});
}

TEST(UniqueHashes, IoErrorThrowsOnEveryRank) {
    auto& comm = test_helpers::comm();
    EXPECT_THROW(write_unique_hashes_file(comm, ::testing::TempDir() + "no_such_dir/u.bin",
                                          contiguous_slice(comm, synthetic(60, 67))),
                 std::runtime_error);
}

TEST(UniqueHashes, AllPatchesOnOneRank) {
    // Every other rank contributes nothing to the sample sort and still
    // receives its share of the hash space.
    auto& comm = test_helpers::comm();
    const auto full = synthetic(80, 71);
    for (int owner : {0, comm.getSize() - 1}) {
        expect_file_matches(comm, "unique_hashes_one_rank", full,
            test_helpers::slice_all_on_rank(full, {}, owner, comm.getRank()).patches);
    }
}

TEST(UniqueHashes, FewerPatchesThanRanks) {
    auto& comm = test_helpers::comm();
    const Patches full = {{42, 7, 42, 99}};
    expect_file_matches(comm, "unique_hashes_single_patch", full, contiguous_slice(comm, full));
}

TEST(UniqueHashes, FewDistinctHashes) {
    // Far fewer distinct hashes than ranks × samples: the splitters collapse
    // onto the same few values and most ranks end up owning an empty shard.
    auto& comm = test_helpers::comm();
    Patches full(200, std::vector<Hash>{5, 5, 9, 1000003});
    full[17] = {};                 // an empty patch in the middle
    full[101].push_back(77);       // one hash that occurs exactly once
    expect_file_matches(comm, "unique_hashes_few_distinct", full, contiguous_slice(comm, full));
    expect_file_matches(comm, "unique_hashes_few_distinct_rr", full,
        test_helpers::slice_round_robin(full, {}, comm.getRank(), comm.getSize()).patches);
}

TEST(UniqueHashes, ClusteredHashesAndExtremes) {
    // One dense run of consecutive values plus both ends of the hash range.
    auto& comm = test_helpers::comm();
    const Hash max = std::numeric_limits<Hash>::max();
    Patches full;
    for (Hash p = 0; p < 300; ++p) {
        std::vector<Hash> patch;
        for (Hash i = 0; i < 20; ++i) patch.push_back((Hash{1} << 40) + p * 7 + i);
        if (p % 50 == 0) { patch.push_back(0); patch.push_back(max); patch.push_back(max - 1); }
        full.push_back(std::move(patch));
    }
    expect_file_matches(comm, "unique_hashes_clustered", full, contiguous_slice(comm, full));
}

TEST(UniqueHashes, SmallExchangeChunks) {
    // Force the hash exchange through many tiny point-to-point chunks (an odd
    // size, so chunk boundaries fall inside elements).
    auto& comm = test_helpers::comm();
    const test_helpers::ChunkBytesGuard guard(29);
    const auto full = synthetic(400, 73);
    expect_file_matches(comm, "unique_hashes_chunked", full, contiguous_slice(comm, full));
}

TEST(UniqueHashes, MediumScale) {
    auto& comm = test_helpers::comm();
    const auto full = synthetic(test_helpers::scale_patch_count(20000), 79);
    expect_file_matches(comm, "unique_hashes_scale", full, contiguous_slice(comm, full));
}
