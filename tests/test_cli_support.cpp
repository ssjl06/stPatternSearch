// Tests for the CLI drivers' app-level helpers (core/cli_support).

#include "core/cli_support.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using stPS::cli::same_output_path;
using stPS::cli::OptionParse;
using stPS::cli::parse_synthetic_option;

TEST(CliSupport, SameOutputPathNormalizes) {
    // Two outputs aimed at one file: ups-pattern-stats rejects these, or the
    // stats writer would silently overwrite the unique-hash export.
    EXPECT_TRUE(same_output_path("result.bin", "result.bin"));
    EXPECT_TRUE(same_output_path("result.bin", "./result.bin"));
    EXPECT_TRUE(same_output_path("sub/../result.bin", "result.bin"));
    EXPECT_FALSE(same_output_path("stats.txt", "hashes.bin"));
    EXPECT_FALSE(same_output_path("a/result.bin", "b/result.bin"));
}

TEST(CliSupport, SameOutputPathCatchesHardLinks) {
    // Different names, one inode: writing both would still overwrite.
    const std::string dir  = ::testing::TempDir();
    const std::string a    = dir + "cli_hardlink_a.bin";
    const std::string b    = dir + "cli_hardlink_b.bin";
    const std::string other = dir + "cli_hardlink_other.bin";
    std::remove(b.c_str());
    { std::ofstream(a) << "x"; std::ofstream(other) << "y"; }
    std::error_code ec;
    std::filesystem::create_hard_link(a, b, ec);
    if (ec) GTEST_SKIP() << "hard links unsupported here: " << ec.message();

    EXPECT_TRUE(same_output_path(a, b));
    EXPECT_FALSE(same_output_path(a, other));
    std::remove(a.c_str()); std::remove(b.c_str()); std::remove(other.c_str());
}

TEST(CliSupport, ParseSyntheticOption) {
    char a0[] = "prog", a1[] = "--N", a2[] = "500", a3[] = "--overlap", a4[] = "0.25",
         a5[] = "--output", a6[] = "--seed";
    char* argv[] = {a0, a1, a2, a3, a4, a5, a6};
    const int argc = 7;
    stPS::SyntheticParams params;

    int i = 1;
    EXPECT_EQ(parse_synthetic_option(argc, argv, i, params), OptionParse::Consumed);
    EXPECT_EQ(i, 2);  // now on the value
    EXPECT_EQ(params.N, 500u);

    i = 3;
    EXPECT_EQ(parse_synthetic_option(argc, argv, i, params), OptionParse::Consumed);
    EXPECT_DOUBLE_EQ(params.overlap, 0.25);

    i = 5;  // a driver-specific option: left alone
    EXPECT_EQ(parse_synthetic_option(argc, argv, i, params), OptionParse::NotMine);
    EXPECT_EQ(i, 5);

    i = 6;  // last argument, no value after it
    EXPECT_EQ(parse_synthetic_option(argc, argv, i, params), OptionParse::MissingValue);
}
