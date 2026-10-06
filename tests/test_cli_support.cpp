// Tests for the CLI drivers' app-level helpers (core/cli_support).

#include "core/cli_support.hpp"

#include <gtest/gtest.h>

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
