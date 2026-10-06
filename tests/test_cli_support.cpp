// Tests for the CLI drivers' app-level helpers (core/cli_support).

#include "core/cli_support.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using stPS::cli::same_output_path;

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
