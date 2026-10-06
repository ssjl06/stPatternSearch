#pragma once

#include <sys/types.h>

#include <cstddef>
#include <string>

namespace stComm { class Comm; }

namespace stPS::io {

// Collective: every rank pwrites its own `bytes` at `offset` of one shared
// file (created if missing); rank 0 additionally writes `head` (if any) at
// offset 0 — so a file header needn't be copied in front of rank 0's data.
// Rank 0 then truncates to `total_size` so a longer stale file can't leave
// garbage past the end. Used by the UPS stats file and the unique-hash file.
//
// Errors (open, pwrite, ftruncate, close — the last can be the first to
// report a deferred NFS/quota failure) are collected per rank and agreed on
// with an allreduce, so if any rank fails every rank throws
// std::runtime_error("<label>: <path>: ...") together instead of one rank
// stranding its peers in the next collective. Returns only once the file is
// complete on every rank.
//
// `path` must be one file for every rank (local disk on one node; a shared
// filesystem across nodes) — a node-local path in a multi-node run gives each
// node its own partial copy, undetected.
void write_shared_file(stComm::Comm& comm, const std::string& path, const char* label,
                       const void* data, std::size_t bytes, off_t offset,
                       off_t total_size,
                       const void* head = nullptr, std::size_t head_bytes = 0);

}  // namespace stPS::io
