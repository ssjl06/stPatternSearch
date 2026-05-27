#pragma once

#include <cstdint>

namespace fullchipusc {

using ElementId = std::uint64_t;
using PatchId   = std::uint64_t;
using Score     = std::int64_t;
using Hash      = std::uint64_t;

inline constexpr Score kDisabledScore = -1;

}  // namespace fullchipusc
