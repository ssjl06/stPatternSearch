#pragma once

#include <cstdint>
#include <tuple>

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

    // std::tie compares with operator< only, and the defaulted == is
    // memberwise — exact equality is intended here, without tripping
    // -Wfloat-equal in our own code.
    friend bool operator<(const Point& a, const Point& b) noexcept {
        return std::tie(a.x, a.y) < std::tie(b.x, b.y);
    }
    friend bool operator==(const Point&, const Point&) noexcept = default;
};

inline constexpr Score kDisabledScore = -1;

}  // namespace stPS
