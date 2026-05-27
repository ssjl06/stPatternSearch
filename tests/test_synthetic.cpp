#include "data/synthetic.hpp"
#include "helpers/local_setup.hpp"

#include <gtest/gtest.h>

using namespace fullchipusc;

TEST(Synthetic, DeterministicForSameSeed) {
    SyntheticParams p;
    p.N = 200; p.M = 30; p.K_mean = 10; p.overlap = 0.3; p.seed = 42;
    const auto a = generate_synthetic(p);
    const auto b = generate_synthetic(p);
    EXPECT_EQ(a, b);
}

TEST(Synthetic, DifferentSeedsDiffer) {
    SyntheticParams p1, p2;
    p1.N = 200; p1.M = 30; p1.K_mean = 10; p1.overlap = 0.3; p1.seed = 1;
    p2 = p1; p2.seed = 2;
    EXPECT_NE(generate_synthetic(p1), generate_synthetic(p2));
}

TEST(Synthetic, PatchCountMatches) {
    SyntheticParams p;
    p.N = 500; p.M = 50; p.K_mean = 20; p.overlap = 0.5; p.seed = 7;
    const auto patches = generate_synthetic(p);
    EXPECT_EQ(patches.size(), p.M);
}

TEST(Synthetic, UniverseScalesWithOverlap) {
    // Generator contract: universe size ≈ region_size = N*(1-overlap) + K*overlap.
    // We sample enough patches that the region should be (near-)saturated.
    SyntheticParams p;
    p.N = 1000; p.M = 200; p.K_mean = 50; p.overlap = 0.0; p.seed = 1;
    const auto s0 = test_helpers::run_local_setup(generate_synthetic(p));
    // overlap=0 → region = N → universe close to N
    EXPECT_GT(s0.N, p.N * 9 / 10);
    EXPECT_LE(s0.N, p.N);

    p.overlap = 0.5;
    const auto s_mid = test_helpers::run_local_setup(generate_synthetic(p));
    // overlap=0.5 → region ≈ 525 → universe close to 525
    EXPECT_GT(s_mid.N, 450u);
    EXPECT_LE(s_mid.N, 600u);
}

TEST(Synthetic, HighOverlapShrinksUniverse) {
    SyntheticParams p_low, p_high;
    p_low.N = 1000; p_low.M = 50; p_low.K_mean = 20; p_low.overlap = 0.0; p_low.seed = 3;
    p_high = p_low; p_high.overlap = 0.95;
    const auto u_low  = test_helpers::run_local_setup(generate_synthetic(p_low)).N;
    const auto u_high = test_helpers::run_local_setup(generate_synthetic(p_high)).N;
    // High overlap → patches cluster → fewer distinct elements seen.
    EXPECT_LT(u_high, u_low);
}
