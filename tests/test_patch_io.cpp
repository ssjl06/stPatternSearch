// Tests for the .stps binary patch file (io/patch_reader.hpp, M7).
//
// Host-only tests write per-rank files (every rank runs every test under the
// mpiexec wrapper — distinct paths avoid write races). The distributed test
// shares one file: rank 0 writes it, everyone reads their own slice, and the
// solve must match the synthetic-path solve bit for bit.

#include "io/patch_reader.hpp"
#include "data/synthetic.hpp"
#include "test_nccl_env.hpp"

#include <stPS/stPS.h>
#include <stComm/stComm.h>

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace stPS;

namespace {

std::vector<std::vector<Hash>> small_synthetic(std::uint64_t seed = 7) {
    SyntheticParams p;
    p.N = 500; p.M = 40; p.K_mean = 12; p.overlap = 0.4; p.seed = seed;
    return generate_synthetic(p);
}

// Per-rank temp path for host-only tests (each rank writes its own copy).
std::string rank_path(const std::string& name) {
    return ::testing::TempDir() + "stps_io_" + name + "_r" +
           std::to_string(test_helpers::comm().getRank()) + ".stps";
}

struct FileGuard {  // unlink on scope exit
    std::string path;
    ~FileGuard() { std::remove(path.c_str()); }
};

}  // namespace

TEST(PatchIo, RoundTrip) {
    const auto patches = small_synthetic();
    FileGuard g{rank_path("roundtrip")};
    write_patch_file(g.path, patches);

    auto reader = open_patch_file(g.path);
    ASSERT_EQ(reader->patch_count(), patches.size());

    const PatchSlice all = reader->read_slice(0, patches.size());
    ASSERT_EQ(all.patches.size(), patches.size());
    for (std::size_t p = 0; p < patches.size(); ++p) {
        EXPECT_EQ(all.patches[p], patches[p]) << "patch " << p;
        EXPECT_EQ(all.global_ids[p], static_cast<PatchId>(p));
    }
}

TEST(PatchIo, SliceMatchesRankSlicer) {
    const auto patches = small_synthetic(11);
    FileGuard g{rank_path("slices")};
    write_patch_file(g.path, patches);
    auto reader = open_patch_file(g.path);
    const std::uint64_t M = reader->patch_count();

    // Every (rank, size) split a driver could ask for must equal the in-memory
    // slicer — that equivalence is what makes --input solves bit-identical to
    // synthetic solves.
    for (int size : {1, 2, 3, 7}) {
        for (int rank = 0; rank < size; ++rank) {
            const std::uint64_t b = (M * static_cast<std::uint64_t>(rank))     / size;
            const std::uint64_t e = (M * (static_cast<std::uint64_t>(rank)+1)) / size;
            const PatchSlice got = reader->read_slice(b, e);
            const PatchSlice ref = slice_patches_by_rank(patches, rank, size);
            EXPECT_EQ(got.patches,    ref.patches)    << "size=" << size << " rank=" << rank;
            EXPECT_EQ(got.global_ids, ref.global_ids) << "size=" << size << " rank=" << rank;
        }
    }
}

TEST(PatchIo, EmptyAndOutOfRange) {
    const auto patches = small_synthetic(13);
    FileGuard g{rank_path("edges")};
    write_patch_file(g.path, patches);
    auto reader = open_patch_file(g.path);
    const std::uint64_t M = reader->patch_count();

    const PatchSlice empty = reader->read_slice(M, M);  // legal empty slice
    EXPECT_TRUE(empty.patches.empty());
    EXPECT_THROW(reader->read_slice(0, M + 1), std::runtime_error);
    EXPECT_THROW(reader->read_slice(2, 1),     std::runtime_error);
}

TEST(PatchIo, RejectsCorruptFiles) {
    // Missing file.
    EXPECT_THROW(open_patch_file(rank_path("does_not_exist")), std::runtime_error);

    // Bad magic.
    FileGuard bad{rank_path("badmagic")};
    { std::ofstream f(bad.path, std::ios::binary); f << "NOTSTPS!garbagegarbage"; }
    EXPECT_THROW(open_patch_file(bad.path), std::runtime_error);

    // Truncated: valid file cut short must fail the open-time size check.
    const auto patches = small_synthetic(17);
    FileGuard cut{rank_path("truncated")};
    write_patch_file(cut.path, patches);
    {
        std::ifstream in(cut.path, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
        bytes.resize(bytes.size() - 16);
        std::ofstream out(cut.path, std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    EXPECT_THROW(open_patch_file(cut.path), std::runtime_error);
}

// End-to-end: a solve fed from the file must match the synthetic-path solve
// bit for bit — the file is just a different transport for the same patches.
TEST(PatchIo, DistributedFileSolveMatchesSynthetic) {
    stComm::Comm& comm = test_helpers::comm();
    const int rank = comm.getRank();
    const int size = comm.getSize();

    SyntheticParams params;
    params.N = 1000; params.M = 200; params.K_mean = 30; params.overlap = 0.5; params.seed = 3;
    const auto raw_full = generate_synthetic(params);

    // One shared file: rank 0 writes, everyone reads its own slice.
    const std::string path = ::testing::TempDir() + "stps_io_dist_shared.stps";
    if (rank == 0) write_patch_file(path, raw_full);
    comm.barrier();

    auto reader = open_patch_file(path);
    const std::uint64_t M = reader->patch_count();
    const std::uint64_t b = (M * static_cast<std::uint64_t>(rank))     / size;
    const std::uint64_t e = (M * (static_cast<std::uint64_t>(rank)+1)) / size;
    PatchSlice from_file = reader->read_slice(b, e);

    UscPatchSelector file_sel(comm);
    const auto file_result =
        file_sel.patch_select(std::move(from_file.patches), std::move(from_file.global_ids));

    PatchSlice from_mem = slice_patches_by_rank(raw_full, rank, size);
    UscPatchSelector mem_sel(comm);
    const auto mem_result =
        mem_sel.patch_select(std::move(from_mem.patches), std::move(from_mem.global_ids));

    EXPECT_EQ(file_result.selected,      mem_result.selected);
    EXPECT_EQ(file_result.covered_count, mem_result.covered_count);
    EXPECT_EQ(file_result.iterations,    mem_result.iterations);

    comm.barrier();  // no rank may still be reading when rank 0 unlinks
    if (rank == 0) std::remove(path.c_str());
}
