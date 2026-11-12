#pragma once

#include <stComm/stComm.h>

#include <cstddef>

namespace stPS::test_helpers {

// Shrinks stComm's point-to-point chunk size for the guard's lifetime, so the
// size_t-count collectives split even tiny test payloads into many chunks —
// the path a >1 GiB exchange takes in production. Every rank must hold the
// same guard: the chunk size is part of the wire protocol.
struct ChunkBytesGuard {
    explicit ChunkBytesGuard(std::size_t bytes) : saved(stComm::detail::p2pChunkBytes()) {
        stComm::detail::p2pChunkBytes() = bytes;
    }
    ~ChunkBytesGuard() { stComm::detail::p2pChunkBytes() = saved; }
    ChunkBytesGuard(const ChunkBytesGuard&)            = delete;
    ChunkBytesGuard& operator=(const ChunkBytesGuard&) = delete;

    std::size_t saved;
};

}  // namespace stPS::test_helpers
