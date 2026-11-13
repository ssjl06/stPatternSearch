#include "core/csr.hpp"
#include "core/inverted_index.hpp"
#include "core/inverted_index_device.hpp"
#include "data/synthetic.hpp"
#include "helpers/local_setup.hpp"

#include <algorithm>
#include <gtest/gtest.h>

using namespace stPS;

namespace {
std::vector<PatchId> as_vec(std::span<const PatchId> sp) {
    return { sp.begin(), sp.end() };
}
}  // namespace

TEST(InvertedIndex, TransposeDesignDocExample) {
    // Same as design doc §3.2.
    std::vector<std::vector<ElementId>> patches = {
        {0, 2},
        {1, 2, 3},
        {0, 4},
        {3, 4},
    };
    auto csr = build_patch_csr(patches);
    auto inv = build_inverted_index(csr, /*N=*/5);

    EXPECT_EQ(inv.num_keys(), 5u);
    // Element → expected patch list
    EXPECT_EQ(as_vec(inv.patches_of(0)), (std::vector<PatchId>{0, 2}));
    EXPECT_EQ(as_vec(inv.patches_of(1)), (std::vector<PatchId>{1}));
    EXPECT_EQ(as_vec(inv.patches_of(2)), (std::vector<PatchId>{0, 1}));
    EXPECT_EQ(as_vec(inv.patches_of(3)), (std::vector<PatchId>{1, 3}));
    EXPECT_EQ(as_vec(inv.patches_of(4)), (std::vector<PatchId>{2, 3}));
}

TEST(InvertedIndex, MissingElementReturnsEmpty) {
    std::vector<std::vector<ElementId>> patches = {{0, 2}, {2, 4}};
    auto csr = build_patch_csr(patches);
    auto inv = build_inverted_index(csr, /*N=*/10);

    EXPECT_TRUE(inv.patches_of(1).empty());
    EXPECT_TRUE(inv.patches_of(3).empty());
    EXPECT_TRUE(inv.patches_of(9).empty());
    EXPECT_EQ(inv.num_keys(), 3u);  // only 0, 2, 4
}

TEST(InvertedIndex, KeysAreSorted) {
    std::vector<std::vector<ElementId>> patches = {{5, 1}, {3, 5}, {7}};
    // patch IDs are inputs to build_patch_csr but we want to test invariant on
    // index keys, so manually sort each patch first.
    for (auto& p : patches) std::sort(p.begin(), p.end());
    auto csr = build_patch_csr(patches);
    auto inv = build_inverted_index(csr, /*N=*/8);

    for (std::size_t i = 1; i < inv.keys.size(); ++i) {
        EXPECT_LT(inv.keys[i - 1], inv.keys[i]);
    }
}

// ---- Device build: must reproduce the host build exactly ----------------------

namespace {

template <typename T>
std::vector<T> download(const DeviceBuffer<T>& d, std::size_t count) {
    std::vector<T> host(count);
    if (count > 0) d.copy_to_host(host.data(), count);
    return host;
}

void expect_device_matches_host(const PatchCsr& csr, std::uint64_t N) {
    const InvertedIndex       want = build_inverted_index(csr, N);
    const DeviceInvertedIndex got  = build_inverted_index_device(csr, N);
    ASSERT_EQ(got.num_keys, want.num_keys());
    EXPECT_EQ(download(got.keys, want.keys.size()), want.keys);
    EXPECT_EQ(download(got.offsets, want.offsets.size()), want.offsets);
    EXPECT_EQ(download(got.data, want.data.size()), want.data);
}

}  // namespace

TEST(InvertedIndexDevice, MatchesHostOnDesignDocExample) {
    expect_device_matches_host(build_patch_csr({{0, 2}, {1, 2, 3}, {0, 4}, {3, 4}}), 5);
}

TEST(InvertedIndexDevice, EmptyAndDegenerateInputs) {
    expect_device_matches_host(build_patch_csr({}), 0);
    expect_device_matches_host(build_patch_csr({{}, {}, {}}), 4);
    expect_device_matches_host(build_patch_csr({{7}}), 8);
    // Empty patches at the front, in the middle and at the end; one element
    // shared by every non-empty patch; IDs at both ends of [0, N).
    expect_device_matches_host(
        build_patch_csr({{}, {0, 5, 1023}, {}, {}, {5}, {5, 1023}, {0, 5}, {}}), 1024);
    // N far above the IDs in use, and N == 1.
    expect_device_matches_host(build_patch_csr({{1, 2}, {2}}), std::uint64_t{1} << 40);
    expect_device_matches_host(build_patch_csr({{0}, {0}, {0}}), 1);
}

TEST(InvertedIndexDevice, MatchesHostOnSynthetic) {
    for (double overlap : {0.1, 0.9}) {
        SyntheticParams p;
        p.N = 40000; p.M = 5000; p.K_mean = 30; p.overlap = overlap; p.seed = 5;
        const auto local = test_helpers::run_local_setup(generate_synthetic(p));
        expect_device_matches_host(local.patches, local.N);
    }
}
