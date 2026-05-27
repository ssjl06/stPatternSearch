#include "core/csr.hpp"
#include "core/inverted_index.hpp"

#include <algorithm>
#include <gtest/gtest.h>

using namespace fullchipusc;

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
