#include "core/csr.hpp"

#include <gtest/gtest.h>

using namespace fullchipusc;

TEST(PatchCsr, BuildAndQuery) {
    // Design doc §3.2 example.
    std::vector<std::vector<ElementId>> patches = {
        {0, 2},
        {1, 2, 3},
        {0, 4},
        {3, 4},
    };
    auto csr = build_patch_csr(patches);
    EXPECT_EQ(csr.M(), 4u);
    EXPECT_EQ(csr.data.size(), 9u);
    std::vector<ElementId> expected_data = {0, 2, 1, 2, 3, 0, 4, 3, 4};
    EXPECT_EQ(csr.data, expected_data);
    std::vector<std::uint64_t> expected_offsets = {0, 2, 5, 7, 9};
    EXPECT_EQ(csr.offsets, expected_offsets);

    for (PatchId p = 0; p < csr.M(); ++p) {
        auto span = csr.patch(p);
        std::vector<ElementId> got(span.begin(), span.end());
        EXPECT_EQ(got, patches[p]);
        EXPECT_EQ(csr.patch_size(p), patches[p].size());
    }
}

TEST(PatchCsr, EmptyPatches) {
    auto csr = build_patch_csr({});
    EXPECT_EQ(csr.M(), 0u);
}

TEST(PatchCsr, PatchWithSingleElement) {
    auto csr = build_patch_csr({{42}});
    EXPECT_EQ(csr.M(), 1u);
    EXPECT_EQ(csr.patch_size(0), 1u);
    auto sp = csr.patch(0);
    EXPECT_EQ(sp.size(), 1u);
    EXPECT_EQ(sp[0], 42u);
}
