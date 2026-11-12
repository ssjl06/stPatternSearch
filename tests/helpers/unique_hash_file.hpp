#pragma once

#include <stPS/unique_hashes.hpp>
#include <stPS/types.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace stPS::test_helpers {

// Parse a unique-hash file: checks the magic, returns {N field, hash array}.
inline std::pair<std::uint64_t, std::vector<Hash>> read_unique_hash_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    EXPECT_TRUE(in.good()) << path;
    char magic[8] = {};
    std::uint64_t n = 0;
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    EXPECT_EQ(std::string(magic, 8), std::string(kUniqueHashMagic, 8));
    std::vector<Hash> hashes;
    Hash h = 0;
    while (in.read(reinterpret_cast<char*>(&h), sizeof(h))) hashes.push_back(h);
    return {n, hashes};
}

}  // namespace stPS::test_helpers
