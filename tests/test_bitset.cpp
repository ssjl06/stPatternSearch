#include "core/bitset.hpp"

#include <gtest/gtest.h>

using namespace fullchipusc;

TEST(DenseBitset, EmptyBitset) {
    DenseBitset b(0);
    EXPECT_EQ(b.num_bits(), 0u);
    EXPECT_EQ(b.popcount(), 0u);
}

TEST(DenseBitset, SetAndTest) {
    DenseBitset b(128);
    EXPECT_FALSE(b.test(0));
    EXPECT_FALSE(b.test(63));
    EXPECT_FALSE(b.test(64));
    EXPECT_FALSE(b.test(127));
    b.set(0);
    b.set(63);
    b.set(64);
    b.set(127);
    EXPECT_TRUE(b.test(0));
    EXPECT_TRUE(b.test(63));
    EXPECT_TRUE(b.test(64));
    EXPECT_TRUE(b.test(127));
    EXPECT_FALSE(b.test(1));
    EXPECT_FALSE(b.test(62));
    EXPECT_FALSE(b.test(65));
    EXPECT_EQ(b.popcount(), 4u);
}

TEST(DenseBitset, BoundaryWords) {
    // 129 bits → 3 words; last word has 1 active bit
    DenseBitset b(129);
    EXPECT_EQ(b.num_words(), 3u);
    b.set(128);
    EXPECT_TRUE(b.test(128));
    EXPECT_EQ(b.popcount(), 1u);
}

TEST(DenseBitset, OrInplace) {
    DenseBitset a(100), b(100);
    a.set(5);
    a.set(50);
    b.set(50);
    b.set(99);
    a.or_inplace(b);
    EXPECT_TRUE(a.test(5));
    EXPECT_TRUE(a.test(50));
    EXPECT_TRUE(a.test(99));
    EXPECT_EQ(a.popcount(), 3u);
}

TEST(DenseBitset, IntersectWithIds) {
    DenseBitset b(200);
    for (ElementId id : {3u, 7u, 64u, 65u, 128u, 199u}) b.set(id);
    std::vector<ElementId> probe = {3, 4, 7, 64, 100, 128, 199, 0};
    EXPECT_EQ(b.popcount_intersect_with_ids(probe), 5u);  // 3,7,64,128,199
}

TEST(DenseBitset, IntersectDenseDense) {
    DenseBitset a(256), c(256);
    a.set(1); a.set(100); a.set(255);
    c.set(100); c.set(255);
    EXPECT_EQ(a.popcount_intersect(c), 2u);
}

TEST(DenseBitset, CollectSetIds) {
    DenseBitset b(200);
    for (ElementId id : {1u, 63u, 64u, 100u, 199u}) b.set(id);
    std::vector<ElementId> ids;
    b.collect_set_ids(ids);
    std::vector<ElementId> expected = {1, 63, 64, 100, 199};
    EXPECT_EQ(ids, expected);
}

TEST(DenseBitset, ResizeClearsBits) {
    DenseBitset b(100);
    b.set(7);
    b.resize(200);
    EXPECT_FALSE(b.test(7));
    EXPECT_EQ(b.popcount(), 0u);
    EXPECT_EQ(b.num_bits(), 200u);
}
