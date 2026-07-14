#pragma once

#include <cstdint>

namespace stPS {

using ElementId = std::uint64_t;
using PatchId   = std::uint64_t;
using Score     = std::int64_t;
using Hash      = std::uint64_t;

// Spatial location of one hash occurrence (layout coordinates). Ordering is
// lexicographic — smaller x first, then smaller y — used wherever a single
// representative location must be chosen deterministically.
struct Point {
    double x = 0.0;
    double y = 0.0;

    friend bool operator<(const Point& a, const Point& b) noexcept {
        return a.x != b.x ? a.x < b.x : a.y < b.y;
    }
    friend bool operator==(const Point& a, const Point& b) noexcept {
        return a.x == b.x && a.y == b.y;
    }
};

inline constexpr Score kDisabledScore = -1;

}  // namespace stPS
