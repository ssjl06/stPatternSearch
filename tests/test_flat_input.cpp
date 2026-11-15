// The flat (CSR) input form must behave exactly like the one-vector-per-patch
// form at every entry point: same selection, same statistics, same unique-hash
// file — and a malformed flat slice must be rejected, not read out of bounds.

#include "data/synthetic.hpp"
#include "helpers/unique_hash_file.hpp"
#include "io/patch_reader.hpp"
#include "test_nccl_env.hpp"

#include <stPS/stPS.h>
#include <stComm/stComm.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace stPS;

namespace {

struct Data {
    std::vector<std::vector<Hash>>  patches;
    std::vector<std::vector<Point>> coords;
};

Data synthetic(std::uint64_t N, std::uint64_t M, std::uint32_t K, double overlap,
               std::uint64_t seed) {
    SyntheticParams p;
    p.N = N; p.M = M; p.K_mean = K; p.overlap = overlap; p.seed = seed;
    Data d;
    d.patches = generate_synthetic(p);
    d.coords  = generate_synthetic_coords(p, d.patches);
    return d;
}

PatchSlice my_slice(const Data& d) {
    stComm::Comm& comm = test_helpers::comm();
    return slice_patches_by_rank(d.patches, d.coords, comm.getRank(), comm.getSize());
}

std::string temp_path(const std::string& name) {
    stComm::Comm& comm = test_helpers::comm();
    return ::testing::TempDir() + name + "_" + test_helpers::suite_name() + "_np" +
           std::to_string(comm.getSize());
}

}  // namespace

TEST(FlatInput, FlattenKeepsEveryPatchInPlace) {
    PatchSlice s;
    s.patches    = {{5, 3}, {}, {9}, {1, 1, 2}};
    s.coords     = {{{0, 1}, {2, 3}}, {}, {{4, 5}}, {{6, 7}, {8, 9}, {10, 11}}};
    s.global_ids = {40, 41, 42, 43};
    const FlatPatches f = flatten_patches(s);
    EXPECT_EQ(f.patch_count(), 4u);
    EXPECT_EQ(f.offsets, (std::vector<std::uint64_t>{0, 2, 2, 3, 6}));
    EXPECT_EQ(f.hashes, (std::vector<Hash>{5, 3, 9, 1, 1, 2}));
    EXPECT_EQ(f.global_ids, s.global_ids);
    ASSERT_EQ(f.coords.size(), 6u);
    EXPECT_DOUBLE_EQ(f.coords[2].x, 4);
    EXPECT_DOUBLE_EQ(f.coords[5].y, 11);

    EXPECT_EQ(flatten_patches(PatchSlice{}).patch_count(), 0u);
    PatchSlice bad = s;
    bad.coords.pop_back();
    EXPECT_THROW(flatten_patches(bad), std::invalid_argument);
}

TEST(FlatInput, UscMatchesNested) {
    stComm::Comm& comm = test_helpers::comm();
    const Data d = synthetic(3000, 600, 20, 0.5, 7);
    for (PartitionMode mode : {PartitionMode::ByPatch, PartitionMode::ByElement}) {
        PatchSlice s = my_slice(d);
        const FlatPatches flat = flatten_patches(s);
        UscPatchSelector nested_sel(comm, mode), flat_sel(comm, mode);
        const PatchSelection a = nested_sel.patch_select(s.patches, s.global_ids);
        const PatchSelection b = flat_sel.patch_select(flat);
        EXPECT_EQ(a.selected, b.selected);
        EXPECT_EQ(a.covered_count, b.covered_count);
        EXPECT_GT(b.selected.size(), 0u);
    }
}

TEST(FlatInput, UscAcceptsNoPatches) {
    stComm::Comm& comm = test_helpers::comm();
    for (PartitionMode mode : {PartitionMode::ByPatch, PartitionMode::ByElement}) {
        UscPatchSelector sel(comm, mode);
        EXPECT_TRUE(sel.patch_select(FlatPatches{}).selected.empty());
    }
}

TEST(FlatInput, UpsMatchesNested) {
    stComm::Comm& comm = test_helpers::comm();
    const Data d = synthetic(2000, 400, 18, 0.6, 9);
    PatchSlice s = my_slice(d);
    const FlatPatches flat = flatten_patches(s);
    UpsPatternStats ups(comm);
    const auto a = ups.pattern_stats(s.patches, s.coords, 200);
    const auto b = ups.pattern_stats(flat, 200);
    ASSERT_EQ(a.size(), b.size());
    ASSERT_GT(b.size(), 0u);
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].hash, b[i].hash) << i;
        EXPECT_EQ(a[i].count, b[i].count) << i;
        EXPECT_DOUBLE_EQ(a[i].rep.x, b[i].rep.x) << i;
        EXPECT_DOUBLE_EQ(a[i].rep.y, b[i].rep.y) << i;
    }
}

TEST(FlatInput, UniqueHashFileMatchesNested) {
    stComm::Comm& comm = test_helpers::comm();
    const Data d = synthetic(2000, 400, 18, 0.6, 3);
    PatchSlice s = my_slice(d);
    FlatPatches flat = flatten_patches(s);
    const std::string nested_path = temp_path("flat_uh_nested") + ".bin";
    const std::string flat_path   = temp_path("flat_uh_flat") + ".bin";
    const std::uint64_t n1 = write_unique_hashes_file(comm, nested_path, s.patches);
    const std::uint64_t n2 = write_unique_hashes_file(comm, flat_path, std::move(flat.hashes));
    EXPECT_EQ(n1, n2);
    EXPECT_TRUE(test_helpers::read_unique_hash_file(nested_path) ==
                test_helpers::read_unique_hash_file(flat_path));
    comm.barrier();  // no rank may still be reading when rank 0 unlinks
    if (comm.getRank() == 0) {
        std::remove(nested_path.c_str());
        std::remove(flat_path.c_str());
    }
}

// Every rank passes the same malformed slice, so every rank throws.
TEST(FlatInput, RejectsMalformedSlices) {
    stComm::Comm& comm = test_helpers::comm();
    FlatPatches good;
    good.hashes     = {1, 2, 3, 4};
    good.offsets    = {0, 2, 4};
    good.global_ids = {0, 1};
    good.coords     = {{0, 0}, {1, 1}, {2, 2}, {3, 3}};

    const auto usc_throws = [&](FlatPatches p) {
        UscPatchSelector sel(comm);
        EXPECT_THROW(sel.patch_select(std::move(p)), std::invalid_argument);
    };
    { FlatPatches p = good; p.offsets[0] = 1;          usc_throws(p); }   // not starting at 0
    { FlatPatches p = good; p.offsets = {0, 3, 2};     usc_throws(p); }   // decreasing
    { FlatPatches p = good; p.offsets.back() = 9;      usc_throws(p); }   // past the hashes
    { FlatPatches p = good; p.offsets.clear();         usc_throws(p); }   // hashes, no offsets
    { FlatPatches p = good; p.global_ids.pop_back();   usc_throws(p); }   // one ID short

    const auto ups_throws = [&](FlatPatches p) {
        UpsPatternStats ups(comm);
        EXPECT_THROW(ups.pattern_stats(std::move(p), 10), std::invalid_argument);
    };
    { FlatPatches p = good; p.coords.pop_back();       ups_throws(p); }   // a point short
    { FlatPatches p = good; p.coords.clear();          ups_throws(p); }   // no points at all
    { FlatPatches p = good; p.offsets.back() = 9;      ups_throws(p); }
}

TEST(FlatInput, FileReadsFlatLikeNested) {
    const Data d = synthetic(1500, 120, 14, 0.4, 5);
    const std::string path = temp_path("flat_reader") + "_r" +
                             std::to_string(test_helpers::comm().getRank()) + ".stps";
    write_patch_file(path, d.patches, d.coords);
    {
        auto reader = open_patch_file(path);
        for (const auto& [begin, end] : std::vector<std::pair<std::uint64_t, std::uint64_t>>{
                 {0, 120}, {17, 53}, {40, 40}, {119, 120}}) {
            const PatchSlice  nested = reader->read_slice(begin, end);
            const FlatPatches flat   = reader->read_flat(begin, end);
            const FlatPatches want   = flatten_patches(nested);
            EXPECT_EQ(flat.offsets, want.offsets);
            EXPECT_EQ(flat.hashes, want.hashes);
            EXPECT_EQ(flat.global_ids, want.global_ids);
            ASSERT_EQ(flat.coords.size(), want.coords.size());
            for (std::size_t i = 0; i < flat.coords.size(); ++i) {
                EXPECT_DOUBLE_EQ(flat.coords[i].x, want.coords[i].x);
                EXPECT_DOUBLE_EQ(flat.coords[i].y, want.coords[i].y);
            }
        }
    }
    std::remove(path.c_str());
}
