#pragma once

#include <cstdint>
#include <cstdlib>

namespace stPS::test_helpers {

// Patch count for the medium-scale tests. Small by default so the suite stays
// quick on a laptop GPU; on real hardware raise it, e.g.
//   STPS_TEST_SCALE_M=5000000 ctest
// Every rank generates the full synthetic set and builds the host reference,
// so host memory per rank grows with it (roughly 100 B per occurrence).
inline std::uint64_t scale_patch_count(std::uint64_t fallback) {
    const char* env = std::getenv("STPS_TEST_SCALE_M");
    if (env == nullptr || *env == '\0') return fallback;
    const std::uint64_t value = std::strtoull(env, nullptr, 10);
    return value > 0 ? value : fallback;
}

}  // namespace stPS::test_helpers
