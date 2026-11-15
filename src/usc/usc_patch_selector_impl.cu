#include "usc/usc_patch_selector_impl.hpp"

#include "core/collective.hpp"
#include "core/cuda_check.hpp"
#include "core/host_profiler.hpp"
#include "core/hash_id_map_device.hpp"
#include "core/inverted_index_device.hpp"

#include <stComm/stComm.h>

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <utility>
#include <string>
#include <cstdlib>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace stPS {

namespace {

// CUDA kernels and helpers used by solve() — kept in this TU's anonymous
// namespace so they don't pollute the public namespace.

constexpr int kScoreBlockThreads = 256;

// Resolve element `id`'s patch-list range [start, end) in a sorted-key inverted
// index (lower_bound over `inv_keys`). An element that does not occur on this
// rank / shard is not in the index and gets the empty range.
__device__ inline void inv_index_range(ElementId            id,
                                       const ElementId*     inv_keys,
                                       std::uint64_t        inv_keys_n,
                                       const std::uint64_t* inv_offsets,
                                       std::uint64_t&       start,
                                       std::uint64_t&       end) {
    std::uint64_t lo = 0, hi = inv_keys_n;
    while (lo < hi) {
        const std::uint64_t mid = lo + ((hi - lo) >> 1);
        if (inv_keys[mid] < id) lo = mid + 1;
        else                    hi = mid;
    }
    if (lo < inv_keys_n && inv_keys[lo] == id) {
        start = inv_offsets[lo];
        end   = inv_offsets[lo + 1];
    } else {
        start = 0;
        end   = 0;
    }
}

// Design doc §6.3 strategy A — inverted-index, affected-only score update:
// for each newly covered element, look it up in this rank's inverted index
// (binary search over the sorted keys) and decrement by one the score of every
// patch that contains it.
//
// Correctness vs the strategy-B full sweep: a patch's score equals its count of
// still-uncovered elements, so subtracting 1 for each newly covered element it
// contains is exactly popcount(patch ∩ newly_covered). Each element becomes
// "covered" exactly once over the whole solve, so every (element, patch)
// incidence is visited at most once across all iterations — total work is
// O(local inverted-index size). Elements not present on this rank fall out of
// the binary search and are skipped.

// ---------------------------------------------------------------------------
// Block maxima: argmax without scanning every score every step.
//
// The scores are cut into blocks of 2^shift patches, each with its maximum and
// the first patch holding it. A step changes few scores (the winner, plus the
// patches sharing its newly covered elements), so every writer marks the
// block it touched and the next step recomputes only those blocks; the winner
// is then the argmax over the block maxima — M / 2^shift values instead of M.
// Both stages keep the first position among equal scores, so the selection is
// the one a full scan gives.
// ---------------------------------------------------------------------------

struct MaxBlocks {
    WireScore*    block_val;   // per block: maximum score
    int*          block_arg;   // per block: first patch with that score
    unsigned int* dirty;       // per block: 1 while queued for a refresh
    unsigned int* list;        // queued blocks, list[0 .. *count)
    unsigned int* count;
    unsigned int  shift;
};

// Called by whoever changes scores[patch]. A block is queued once.
__device__ inline void mark_dirty(const MaxBlocks& t, std::uint64_t patch) {
    const unsigned int b = static_cast<unsigned int>(patch >> t.shift);
    if (t.dirty[b] == 0 && atomicExch(&t.dirty[b], 1u) == 0) {
        t.list[atomicAdd(t.count, 1u)] = b;
    }
}

constexpr int kRefreshThreads = 64;   // power of two (tree reduction)
constexpr WireScore kLowestScore = std::numeric_limits<WireScore>::min();
constexpr int       kNoPosition  = std::numeric_limits<int>::max();

// Recompute the queued blocks. One CUDA block per queued block, striding over
// the list; its threads scan the block's scores and reduce to (max, first
// position). The queue itself is emptied by the commit kernel that follows.
__global__ void block_max_refresh_kernel(const WireScore* scores, std::uint64_t M,
                                         MaxBlocks t) {
    __shared__ WireScore s_val[kRefreshThreads];
    __shared__ int       s_arg[kRefreshThreads];
    const unsigned int n   = *t.count;
    const unsigned int tid = threadIdx.x;
    for (unsigned int e = blockIdx.x; e < n; e += gridDim.x) {
        const unsigned int  b  = t.list[e];
        const std::uint64_t lo = static_cast<std::uint64_t>(b) << t.shift;
        std::uint64_t hi = lo + (std::uint64_t{1} << t.shift);
        if (hi > M) hi = M;
        WireScore best = kLowestScore;
        int       arg  = kNoPosition;
        // Ascending positions and a strict >: the first maximum is kept.
        for (std::uint64_t i = lo + tid; i < hi; i += kRefreshThreads) {
            const WireScore v = scores[i];
            if (v > best) { best = v; arg = static_cast<int>(i); }
        }
        s_val[tid] = best;
        s_arg[tid] = arg;
        __syncthreads();
        for (unsigned int half = kRefreshThreads / 2; half > 0; half >>= 1) {
            if (tid < half) {
                const WireScore ov = s_val[tid + half];
                const int       oa = s_arg[tid + half];
                if (ov > s_val[tid] || (ov == s_val[tid] && oa < s_arg[tid])) {
                    s_val[tid] = ov;
                    s_arg[tid] = oa;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            t.block_val[b] = s_val[0];
            t.block_arg[b] = s_arg[0];
            t.dirty[b]     = 0;
        }
        __syncthreads();  // results written before the next round reuses s_*
    }
}

constexpr int kTopThreads = 1024;   // one CUDA block; power of two

// Run by ONE CUDA block of kTopThreads threads: the maximum of
// values[lo, hi), first position among equals. On return thread 0 holds it in
// s_val[0] / s_arg[0] (kNoPosition for an empty range).
__device__ inline void reduce_range_max(const WireScore* values, std::uint64_t lo,
                                        std::uint64_t hi, WireScore* s_val, int* s_arg) {
    const unsigned int tid = threadIdx.x;
    WireScore best = kLowestScore;
    int       pos  = kNoPosition;
    for (std::uint64_t i = lo + tid; i < hi; i += kTopThreads) {
        const WireScore v = values[i];
        if (v > best) { best = v; pos = static_cast<int>(i); }
    }
    s_val[tid] = best;
    s_arg[tid] = pos;
    __syncthreads();
    for (unsigned int half = kTopThreads / 2; half > 0; half >>= 1) {
        if (tid < half) {
            const WireScore ov = s_val[tid + half];
            const int       oa = s_arg[tid + half];
            if (ov > s_val[tid] || (ov == s_val[tid] && oa < s_arg[tid])) {
                s_val[tid] = ov;
                s_arg[tid] = oa;
            }
        }
        __syncthreads();
    }
}

// Second argmax stage: the maximum over the block maxima; thread 0 ends up
// with the winning score and PATCH (via block_arg). The loops fuse this into
// the kernel that consumes the winner, saving a launch per step — the loops
// are bound by launches as much as by GPU work.
__device__ inline void reduce_block_maxima(const MaxBlocks& t, std::uint64_t num_blocks,
                                           WireScore* s_val, int* s_arg) {
    reduce_range_max(t.block_val, 0, num_blocks, s_val, s_arg);
    if (threadIdx.x == 0 && s_arg[0] != kNoPosition) s_arg[0] = t.block_arg[s_arg[0]];
}

// ---------------------------------------------------------------------------
// Device-resident loop kernels (Full-GPU iteration track), shared by both
// partition modes.
//
// The host enqueues whole batches of rounds blind, so every launch here uses a
// FIXED grid and reads counts from device memory — the host knows no outcome
// when it enqueues. All per-round control flow (which patches are selected,
// covered_count, termination) lives in each mode's walk kernel.
// ---------------------------------------------------------------------------

constexpr int kFixedGridBlocks   = 256;  // fixed grid for the grid-stride kernels
constexpr int kFixedBlockThreads = 128;

// Device-resident loop state — the walk kernel is its only writer; the host
// reads it once per batch (24 B D2H) to decide whether to enqueue more.
struct LoopState {
    std::uint64_t covered_count;
    std::uint64_t num_selected;   // == iterations executed
    std::uint32_t done;
    std::uint32_t last_score;     // most recent winner's score (batch sizing)
};

// Per-round scratch, written by the mode's walk / build kernels and read by
// the update kernels of the same (stream-ordered) round.
struct WinnerInfo {
    unsigned int newly_count;   // elements newly covered this round (this rank's view)
    unsigned int live;          // 1 while this round selected at least one patch
};

// One 32-bit word of the ByElement exchanges (a count, a flag or a patch slot
// — slots fit: M_global is within int range).
using DeltaWord = std::uint32_t;

// Strategy-A score update, fixed-grid: blocks stride over the newly covered
// elements. One block resolves an element's patch-list range (thread 0, shared
// to the block), then its threads decrement those patches' scores.
__global__ void fixed_score_update_kernel(const ElementId*      newly_ids,
                                          const WinnerInfo*     w,
                                          const ElementId*      inv_keys,
                                          std::uint64_t         inv_keys_n,
                                          const std::uint64_t*  inv_offsets,
                                          const PatchId*        inv_data,
                                          WireScore*            scores,
                                          MaxBlocks             blocks) {
    const unsigned int n = w->newly_count;
    __shared__ std::uint64_t s_start;
    __shared__ std::uint64_t s_end;
    for (std::uint64_t e = blockIdx.x; e < n; e += gridDim.x) {
        if (threadIdx.x == 0) {
            inv_index_range(newly_ids[e], inv_keys, inv_keys_n, inv_offsets, s_start, s_end);
        }
        __syncthreads();
        for (std::uint64_t i = s_start + threadIdx.x; i < s_end; i += blockDim.x) {
            // Decrements commute: the final scores are order-independent.
            const PatchId q = inv_data[i];
            atomicSub(reinterpret_cast<int*>(&scores[q]), 1);
            mark_dirty(blocks, q);
        }
        __syncthreads();  // all readers done before thread 0 overwrites s_start/s_end
    }
}

// ---------------------------------------------------------------------------
// ByElement delta exchange (multi-rank).
//
// Every rank keeps the FULL score of every patch. A selection changes them
// only through the newly covered elements: each (element, patch) incidence of
// such an element lowers that patch's score by one, and only the element's
// owner rank knows those incidences. So instead of re-forming all M_global
// scores every iteration, each rank ships the patch slots to decrement — its
// "delta stream" for the iteration: the inverted-index lists of its newly
// covered elements, concatenated.
//
// The stream travels through a fixed-size window so the host can enqueue
// blind: per step each rank fills its own segment of a P-segment buffer
// ([count, more-to-come flag, up to `window` slots]; the other segments stay
// zero) and one ncclAllReduce<SUM> gives every rank all segments. A stream
// longer than the window continues in the following steps, and no patch is
// selected until every rank has sent everything (see the commit kernel).
// Over a whole solve each incidence is sent once.
// ---------------------------------------------------------------------------

// Rank-local progress through the current iteration's delta stream.
struct DeltaCursor {
    std::uint64_t total;         // stream length
    std::uint64_t sent;          // entries already shipped
    std::uint64_t window_start;  // this step's window: [window_start, +n_send)
    std::uint32_t n_send;
    std::uint32_t n_newly;       // newly covered elements behind the stream
};

// deg[i] = inverted-index list length of newly covered element i, 0 for
// i in [newly_count, len]. Only on a step that selected a patch — while a
// stream is being drained the previous values must stay.
__global__ void elem_delta_degrees_kernel(const ElementId*     newly_ids,
                                          const WinnerInfo*    w,
                                          const ElementId*     inv_keys,
                                          std::uint64_t        inv_keys_n,
                                          const std::uint64_t* inv_offsets,
                                          std::uint64_t        len,
                                          std::uint64_t*       deg) {
    if (!w->live) return;
    const unsigned int n = w->newly_count;
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         i <= len; i += stride) {
        std::uint64_t d = 0;
        if (i < n) {
            std::uint64_t start, end;
            inv_index_range(newly_ids[i], inv_keys, inv_keys_n, inv_offsets, start, end);
            d = end - start;
        }
        deg[i] = d;
    }
}

// `prefix` is the exclusive prefix sum of deg over [0, len], so prefix[len]
// is the stream length.
__global__ void elem_delta_begin_kernel(const WinnerInfo* w, const std::uint64_t* prefix,
                                        std::uint64_t len, DeltaCursor* c) {
    if (!w->live) return;
    c->total   = prefix[len];
    c->sent    = 0;
    c->n_newly = w->newly_count;
}

// Take the next window off the stream and write this rank's segment header.
__global__ void elem_delta_plan_kernel(DeltaCursor* c, std::uint64_t window,
                                       DeltaWord* segment) {
    const std::uint64_t left = c->total - c->sent;
    const std::uint64_t n    = left < window ? left : window;
    c->window_start = c->sent;
    c->n_send       = static_cast<std::uint32_t>(n);
    c->sent        += n;
    segment[0] = static_cast<DeltaWord>(n);
    segment[1] = (left - n) ? 1u : 0u;
}

// Write the window's slots: element i owns stream positions
// [prefix[i], prefix[i + 1]).
__global__ void elem_delta_fill_kernel(const ElementId*     newly_ids,
                                       const DeltaCursor*   c,
                                       const std::uint64_t* prefix,
                                       const ElementId*     inv_keys,
                                       std::uint64_t        inv_keys_n,
                                       const std::uint64_t* inv_offsets,
                                       const PatchId*       inv_data,
                                       DeltaWord*           slots) {
    const std::uint64_t ws = c->window_start;
    const std::uint64_t we = ws + c->n_send;
    const std::uint64_t n  = c->n_newly;
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         i < n; i += stride) {
        const std::uint64_t lo = prefix[i];
        const std::uint64_t hi = prefix[i + 1];
        if (hi <= ws || lo >= we) continue;
        std::uint64_t start, end;
        inv_index_range(newly_ids[i], inv_keys, inv_keys_n, inv_offsets, start, end);
        const std::uint64_t from = lo > ws ? lo : ws;
        const std::uint64_t to   = hi < we ? hi : we;
        for (std::uint64_t pos = from; pos < to; ++pos) {
            slots[pos - ws] = static_cast<DeltaWord>(inv_data[start + (pos - lo)]);
        }
    }
}

// Apply every rank's window to the replicated scores. Decrements commute.
__global__ void elem_delta_apply_kernel(const DeltaWord* recv, std::uint64_t ranks,
                                        std::uint64_t window, WireScore* scores,
                                        MaxBlocks blocks) {
    const std::uint64_t total  = ranks * window;
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t idx = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         idx < total; idx += stride) {
        const DeltaWord* segment = recv + (idx / window) * (window + 2);
        const std::uint64_t j = idx % window;
        if (j < segment[0]) {
            const DeltaWord q = segment[2 + j];
            atomicSub(reinterpret_cast<int*>(&scores[q]), 1);
            mark_dirty(blocks, q);
        }
    }
}

// ---------------------------------------------------------------------------
// ByPatch device-resident loop: several selections per round.
//
// Sequential greedy picks the best patch, updates every score, and repeats. A
// round here takes the K best patches at once — ordered by (score descending,
// rank ascending, local patch ascending), the order greedy breaks ties in —
// and walks them in that order:
//
//   * A candidate sharing no still-uncovered element with the patches already
//     accepted this round keeps its score. Every other score can only have
//     dropped, so it is exactly the patch greedy would pick next: accept.
//   * A candidate that does share some loses at least that many points. It is
//     skipped; `skip_best` keeps the largest remaining score such a candidate
//     can have (an upper bound — later accepts may lower it further).
//   * Once skip_best reaches the next candidate's score, a skipped patch may
//     outrank everything left (it comes earlier on ties): the round ends, and
//     the next round sees the true scores.
//
// The accepted patches, in order, are therefore the selections sequential
// greedy makes. The first candidate is always accepted, so every round
// advances.
//
// A candidate's key is (score << 32) | (65535 - rank) << 16 | (65535 - j) with
// j its place among its rank's candidates: larger key = earlier in the order.
// 0 means "no candidate". Scores fit 32 bits (a patch holds at most INT_MAX
// elements, validated on entry).
// ---------------------------------------------------------------------------

constexpr std::uint64_t kKeyField = 0xFFFFull;

__device__ inline std::uint64_t pack_candidate_key(WireScore score, int rank, unsigned int j) {
    return (static_cast<std::uint64_t>(score) << 32)
         | ((kKeyField - static_cast<std::uint64_t>(rank)) << 16)
         | (kKeyField - j);
}

// The K best patches of one score array, best first (ByPatch: this rank's
// patches; ByElement: all patches, identical on every rank). One CUDA block: K times, take the
// maximum over the block maxima, set that patch aside (its score is parked in
// cand_score and the slot disabled) and recompute its block. The parked
// scores come back in restore_candidates_kernel unless the patch is selected.
__global__ void extract_candidates_kernel(WireScore* scores, std::uint64_t M,
                                                MaxBlocks blocks, std::uint64_t num_blocks,
                                                unsigned int K, int my_rank,
                                                int* cand_patch, WireScore* cand_score,
                                                std::uint64_t* key_segment) {
    __shared__ WireScore s_val[kTopThreads];
    __shared__ int       s_arg[kTopThreads];
    __shared__ int       s_patch;   // patch just set aside, -1 = no more candidates
    const unsigned int tid = threadIdx.x;
    if (tid == 0 && blocks.count) *blocks.count = 0;   // the refresh before this consumed the queue
    unsigned int j = 0;
    for (; j < K; ++j) {
        reduce_block_maxima(blocks, num_blocks, s_val, s_arg);
        if (tid == 0) {
            s_patch = -1;
            if (num_blocks && s_val[0] > 0) {
                const int p = s_arg[0];
                cand_patch[j]  = p;
                cand_score[j]  = s_val[0];
                key_segment[j] = pack_candidate_key(s_val[0], my_rank, j);
                scores[p]      = kDisabledWireScore;
                s_patch        = p;
            }
        }
        __syncthreads();
        if (s_patch < 0) break;
        const std::uint64_t b  = static_cast<std::uint64_t>(s_patch) >> blocks.shift;
        const std::uint64_t lo = b << blocks.shift;
        std::uint64_t hi = lo + (std::uint64_t{1} << blocks.shift);
        if (hi > M) hi = M;
        reduce_range_max(scores, lo, hi, s_val, s_arg);
        if (tid == 0) {
            blocks.block_val[b] = s_val[0];
            blocks.block_arg[b] = s_arg[0];
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (; j < K; ++j) key_segment[j] = 0;
    }
}

// Every rank holds all P segments of K keys (each sorted, best first, zeros
// at the tail). A key's place in the global order is the number of larger
// keys; the K best go to global_keys, and this rank notes where its own
// candidates landed (slot_of[j], -1 = not among the K best). The caller
// pre-fills global_keys with 0 and slot_of with -1.
__global__ void patch_merge_candidates_kernel(const std::uint64_t* keys, unsigned int ranks,
                                              unsigned int K, int my_rank,
                                              std::uint64_t* global_keys, int* slot_of) {
    const std::uint64_t total  = static_cast<std::uint64_t>(ranks) * K;
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         i < total; i += stride) {
        const std::uint64_t key = keys[i];
        if (key == 0) continue;
        std::uint64_t larger = 0;
        for (unsigned int r = 0; r < ranks; ++r) {
            const std::uint64_t* seg = keys + static_cast<std::uint64_t>(r) * K;
            unsigned int lo = 0, hi = K;   // first index with seg[index] <= key
            while (lo < hi) {
                const unsigned int mid = lo + (hi - lo) / 2;
                if (seg[mid] > key) lo = mid + 1; else hi = mid;
            }
            larger += lo;
        }
        if (larger < K) {
            global_keys[larger] = key;
            if (i / K == static_cast<std::uint64_t>(my_rank)) {
                slot_of[i % K] = static_cast<int>(larger);
            }
        }
    }
}

// Payload slot g (words): [count, global patch ID, count element IDs] — the
// still-uncovered elements of global candidate g, written by its owner rank.
// One CUDA block per local candidate, striding. The caller zeroes the buffer.
__global__ void patch_build_payload_kernel(const ElementId*     patch_data,
                                           const std::uint64_t* patch_offsets,
                                           const PatchId*       patch_global_ids,
                                           const std::uint64_t* covered,
                                           const int*           cand_patch,
                                           const int*           slot_of,
                                           unsigned int         K,
                                           std::uint64_t        slot_words,
                                           std::uint64_t*       payload) {
    __shared__ unsigned int s_count;
    for (unsigned int j = blockIdx.x; j < K; j += gridDim.x) {
        const int g = slot_of[j];
        if (g < 0) continue;   // same for every thread of the block
        if (threadIdx.x == 0) s_count = 0;
        __syncthreads();
        const int p = cand_patch[j];
        std::uint64_t* slot = payload + static_cast<std::uint64_t>(g) * slot_words;
        const std::uint64_t end = patch_offsets[p + 1];
        for (std::uint64_t i = patch_offsets[p] + threadIdx.x; i < end; i += blockDim.x) {
            const ElementId id = patch_data[i];
            if (((covered[id >> 6] >> (id & 63)) & 1ULL) == 0) {
                slot[2 + atomicAdd(&s_count, 1u)] = id;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            slot[0] = s_count;
            slot[1] = patch_global_ids[p];
        }
        __syncthreads();
    }
}

// The walk described above — identical on every rank, since it reads only
// reduced data and the replicated covered bitset. One CUDA block. An element
// "shared with an accepted patch" is one whose covered bit is already set:
// every listed element was uncovered when the round began.
// Accepting = setting those bits, recording the patch, and appending the
// elements to `newly` for the score update.
__global__ void patch_walk_kernel(const std::uint64_t* global_keys,
                                  const std::uint64_t* payload,
                                  unsigned int         K,
                                  std::uint64_t        slot_words,
                                  std::uint64_t        N,
                                  std::uint64_t*       covered,
                                  unsigned int*        accepted,
                                  ElementId*           newly,
                                  PatchId*             selected,
                                  LoopState*           st,
                                  WinnerInfo*          w) {
    __shared__ unsigned int s_overlap;
    __shared__ unsigned int s_offset;     // elements appended to `newly` so far
    __shared__ int          s_decision;   // 0 skip, 1 accept, 2 end of round
    const unsigned int tid = threadIdx.x;
    if (tid < K) accepted[tid] = 0;
    std::uint64_t skip_best = 0;   // thread 0 only
    if (tid == 0) {
        s_offset   = 0;
        s_decision = st->done ? 2 : 0;
    }
    __syncthreads();
    if (s_decision != 2) {
        for (unsigned int g = 0; g < K; ++g) {
            const std::uint64_t* slot  = payload + static_cast<std::uint64_t>(g) * slot_words;
            const std::uint64_t  key   = global_keys[g];
            const std::uint64_t  score = key >> 32;
            const std::uint64_t  count = slot[0];
            if (tid == 0) {
                s_overlap  = 0;
                s_decision = (key == 0 || skip_best >= score) ? 2 : 0;
            }
            __syncthreads();
            if (s_decision == 2) break;

            unsigned int mine = 0;
            for (std::uint64_t i = tid; i < count; i += kTopThreads) {
                const ElementId id = slot[2 + i];
                mine += static_cast<unsigned int>((covered[id >> 6] >> (id & 63)) & 1ULL);
            }
            if (mine) atomicAdd(&s_overlap, mine);
            __syncthreads();
            if (tid == 0) {
                if (s_overlap == 0) {
                    s_decision = 1;
                } else {
                    const std::uint64_t left = score - s_overlap;
                    if (left > skip_best) skip_best = left;
                    s_decision = 0;
                }
            }
            __syncthreads();
            if (s_decision == 1) {
                const unsigned int base = s_offset;
                for (std::uint64_t i = tid; i < count; i += kTopThreads) {
                    const ElementId id = slot[2 + i];
                    atomicOr(reinterpret_cast<unsigned long long*>(&covered[id >> 6]),
                             static_cast<unsigned long long>(1ULL << (id & 63)));
                    newly[base + i] = id;
                }
                __syncthreads();
                if (tid == 0) {
                    accepted[g] = 1;
                    s_offset   += static_cast<unsigned int>(count);
                    selected[st->num_selected] = slot[1];
                    st->num_selected  += 1;
                    st->last_score     = static_cast<std::uint32_t>(score);
                    st->covered_count += score;
                    if (st->covered_count >= N) { st->done = 1; s_decision = 2; }
                }
                __syncthreads();
                if (s_decision == 2) break;
            }
        }
    }
    if (tid == 0) {
        // No positive score anywhere: the solve is over.
        if (!st->done && global_keys[0] == 0) st->done = 1;
        w->newly_count = s_offset;
    }
}

// Give the parked scores back to every candidate that was not selected. Must
// run before the score update, which then lowers them like any other patch.
// A selected patch stays disabled.
__global__ void restore_candidates_kernel(const int* cand_patch, const WireScore* cand_score,
                                     const std::uint64_t* key_segment, const int* slot_of,
                                     const unsigned int* accepted, unsigned int K,
                                     WireScore* scores, MaxBlocks blocks) {
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= K || key_segment[j] == 0) return;
    const int p = cand_patch[j];
    const int g = slot_of ? slot_of[j] : static_cast<int>(j);   // null: local order IS the global one
    if (g < 0 || !accepted[g]) scores[p] = cand_score[j];
    mark_dirty(blocks, static_cast<std::uint64_t>(p));
}

// ---------------------------------------------------------------------------
// ByElement: several selections per round, same rule as ByPatch (see "ByPatch
// device-resident loop"). Every rank holds every score, so all ranks extract
// the same K candidates with no communication — but a candidate's elements
// are spread over the shards, so no rank can tell alone whether two
// candidates share a still-uncovered element. Each rank counts the shared
// elements inside its shard for every candidate pair; one reduction sums the
// K x K table; every rank then walks the candidates against the same table.
//
// The table gives pairwise counts, not the size of a union, so a skipped
// candidate's remaining score is bounded with the largest single overlap —
// never below the truth, which only ends a round earlier than necessary.
// ---------------------------------------------------------------------------

// overlap[i * K + j] (i < j) += still-uncovered elements of this shard that
// candidates i and j share. One CUDA block per candidate j, striding; each
// thread takes elements of j's sub-list and looks them up in every earlier
// candidate's (sorted) sub-list. The caller zeroes the table.
__global__ void elem_overlap_kernel(const ElementId*     patch_data,
                                    const std::uint64_t* sub_offsets,
                                    const std::uint64_t* covered,
                                    const int*           cand_patch,
                                    const std::uint64_t* keys,
                                    unsigned int         K,
                                    DeltaWord*           overlap) {
    for (unsigned int j = blockIdx.x; j < K; j += gridDim.x) {
        if (keys[j] == 0) continue;
        const int pj = cand_patch[j];
        const std::uint64_t end = sub_offsets[pj + 1];
        for (std::uint64_t e = sub_offsets[pj] + threadIdx.x; e < end; e += blockDim.x) {
            const ElementId id = patch_data[e];
            if ((covered[id >> 6] >> (id & 63)) & 1ULL) continue;
            for (unsigned int i = 0; i < j; ++i) {
                const int pi = cand_patch[i];
                std::uint64_t lo = sub_offsets[pi], hi = sub_offsets[pi + 1];
                const std::uint64_t stop = hi;
                while (lo < hi) {
                    const std::uint64_t mid = lo + (hi - lo) / 2;
                    if (patch_data[mid] < id) lo = mid + 1; else hi = mid;
                }
                if (lo < stop && patch_data[lo] == id) {
                    atomicAdd(&overlap[static_cast<std::uint64_t>(i) * K + j], 1u);
                }
            }
        }
    }
}

// The walk, identical on every rank (it reads only replicated or reduced
// data). One CUDA block. Nothing is selected while any rank still has score
// decrements to send (the delta exchange's flags): the scores the candidates
// were extracted from are not final then.
__global__ void elem_walk_kernel(const int*           cand_patch,
                                 const WireScore*     cand_score,
                                 const std::uint64_t* keys,
                                 const DeltaWord*     overlap,
                                 unsigned int         K,
                                 std::uint64_t        N,
                                 const DeltaWord*     delta_recv,
                                 int                  delta_ranks,
                                 std::uint64_t        delta_window,
                                 unsigned int*        accepted,
                                 PatchId*             selected_slots,
                                 LoopState*           st,
                                 WinnerInfo*          w) {
    __shared__ unsigned int s_max_overlap;
    __shared__ int          s_decision;   // 0 continue, 2 end of round
    const unsigned int tid = threadIdx.x;
    if (tid < K) accepted[tid] = 0;
    std::uint64_t skip_best = 0;      // thread 0 only
    bool          draining  = false;  // thread 0 only
    if (tid == 0) {
        w->newly_count = 0;
        w->live        = 0;
        for (int r = 0; r < delta_ranks; ++r) {
            if (delta_recv[static_cast<std::uint64_t>(r) * (delta_window + 2) + 1]) draining = true;
        }
        s_decision = (st->done || draining) ? 2 : 0;
    }
    __syncthreads();
    if (s_decision != 2) {
        for (unsigned int g = 0; g < K; ++g) {
            const std::uint64_t score = keys[g] ? static_cast<std::uint64_t>(cand_score[g]) : 0;
            if (tid == 0) {
                s_max_overlap = 0;
                s_decision    = (score == 0 || skip_best >= score) ? 2 : 0;
            }
            __syncthreads();
            if (s_decision == 2) break;
            if (tid < g && accepted[tid]) {
                const DeltaWord shared_count = overlap[static_cast<std::uint64_t>(tid) * K + g];
                if (shared_count) atomicMax(&s_max_overlap, shared_count);
            }
            __syncthreads();
            if (tid == 0) {
                if (s_max_overlap == 0) {
                    accepted[g] = 1;
                    selected_slots[st->num_selected] = static_cast<PatchId>(cand_patch[g]);
                    st->num_selected  += 1;
                    st->last_score     = static_cast<std::uint32_t>(score);
                    st->covered_count += score;
                    w->live = 1;
                    if (st->covered_count >= N) { st->done = 1; s_decision = 2; }
                } else {
                    const std::uint64_t left = score - s_max_overlap;
                    if (left > skip_best) skip_best = left;
                }
            }
            __syncthreads();
            if (s_decision == 2) break;
        }
    }
    // No positive score anywhere: the solve is over.
    if (tid == 0 && !st->done && !draining && keys[0] == 0) st->done = 1;
}

// This shard's part of the accepted patches: append their still-uncovered
// elements to `newly` and set their covered bits. Accepted patches share no
// uncovered element, so the blocks do not interfere.
__global__ void elem_build_accepted_kernel(const ElementId*     patch_data,
                                           const std::uint64_t* sub_offsets,
                                           std::uint64_t*       covered,
                                           const int*           cand_patch,
                                           const unsigned int*  accepted,
                                           unsigned int         K,
                                           ElementId*           newly,
                                           WinnerInfo*          w) {
    for (unsigned int j = blockIdx.x; j < K; j += gridDim.x) {
        if (!accepted[j]) continue;
        const int p = cand_patch[j];
        const std::uint64_t end = sub_offsets[p + 1];
        for (std::uint64_t e = sub_offsets[p] + threadIdx.x; e < end; e += blockDim.x) {
            const ElementId id = patch_data[e];
            if (((covered[id >> 6] >> (id & 63)) & 1ULL) == 0) {
                newly[atomicAdd(&w->newly_count, 1u)] = id;
                atomicOr(reinterpret_cast<unsigned long long*>(&covered[id >> 6]),
                         static_cast<unsigned long long>(1ULL << (id & 63)));
            }
        }
    }
}

// Host-side owner of the block maxima over one score array. Per step the
// loops enqueue refresh() and then their own kernel that runs
// reduce_block_maxima over num_blocks().
class BlockArgmax {
public:
    BlockArgmax(const WireScore* d_scores, std::uint64_t M) : d_scores_(d_scores), M_(M) {
        if (M_ == 0) return;
        // Block size near sqrt(M / 128): a step refreshes on the order of a
        // hundred blocks, so this balances the refresh against the scan of
        // the block maxima.
        shift_ = 4;
        while (shift_ < 12 && (std::uint64_t{1} << (2 * shift_)) * 128 < M_) ++shift_;
        num_blocks_ = (M_ + (std::uint64_t{1} << shift_) - 1) >> shift_;
        d_val_.resize(num_blocks_);
        d_arg_.resize(num_blocks_);
        d_dirty_.resize(num_blocks_);
        d_list_.resize(num_blocks_);
        d_count_.resize(1);
        // Everything queued: the first refresh builds all the maxima.
        std::vector<unsigned int> all(num_blocks_);
        for (std::uint64_t b = 0; b < num_blocks_; ++b) all[b] = static_cast<unsigned int>(b);
        d_list_.copy_from_host(all.data(), num_blocks_);
        std::fill(all.begin(), all.end(), 1u);
        d_dirty_.copy_from_host(all.data(), num_blocks_);
        const auto n = static_cast<unsigned int>(num_blocks_);
        d_count_.copy_from_host(&n, 1);
    }

    std::uint64_t num_blocks() const noexcept { return num_blocks_; }

    // All-null view when there are no patches: nothing ever marks or reads it.
    MaxBlocks view() {
        return {d_val_.data(), d_arg_.data(), d_dirty_.data(), d_list_.data(),
                d_count_.data(), shift_};
    }

    void refresh(cudaStream_t stream) {
        if (M_ == 0) return;
        block_max_refresh_kernel<<<kFixedGridBlocks, kRefreshThreads, 0, stream>>>(
            d_scores_, M_, view());
    }

private:
    const WireScore* d_scores_;
    std::uint64_t    M_;
    unsigned int     shift_      = 0;
    std::uint64_t    num_blocks_ = 0;
    DeviceBuffer<WireScore>    d_val_;
    DeviceBuffer<int>          d_arg_;
    DeviceBuffer<unsigned int> d_dirty_, d_list_, d_count_;
};

// -DUSC_PROFILE timing of the device-resident loops. Two views, picked with
// the STPS_LOOP_PROFILE environment variable:
//   (unset)  "batch"  — no extra synchronization: per batch, the time the host
//            spends enqueueing against the time it then waits for the stream
//            to drain. Enqueue ≈ total means the loop is bound by launching.
//   "stages" — the stream is drained after every stage and the wait is billed
//            to that stage. Stages then run one at a time, so the sum exceeds
//            the normal loop time; it shows where the device work is.
// Compiled out (every call a no-op) without USC_PROFILE.
class LoopProfiler {
public:
    LoopProfiler(cudaStream_t stream, std::vector<const char*> stages)
        : stream_(stream), names_(std::move(stages)), ns_(names_.size(), 0) {
        if (!detail::kProfileEnabled) return;
        const char* mode = std::getenv("STPS_LOOP_PROFILE");
        stages_mode_ = mode && std::string(mode) == "stages";
        last_ = Clock::now();
    }
    // End of stage `i` (stages view only).
    void mark(std::size_t i) {
        if (!detail::kProfileEnabled || !stages_mode_) return;
        cuda_check(cudaStreamSynchronize(stream_), "loop profiler sync");
        ns_[i] += lap();
    }
    void batch_enqueued() { if (detail::kProfileEnabled) enqueue_ns_ += lap(); }
    void batch_drained()  { if (detail::kProfileEnabled) drain_ns_ += lap(); }
    void step() { ++steps_; }

    void report(std::FILE* out, const char* title, int rank) const {
        if (!detail::kProfileEnabled) return;
        const double steps = static_cast<double>(steps_ ? steps_ : 1);
        std::fprintf(out, "%s rank=%d steps=%llu  (per-step us, total ms)\n", title, rank,
                     static_cast<unsigned long long>(steps_));
        const auto line = [&](const char* name, std::int64_t ns) {
            std::fprintf(out, "  %-22s %8.2f us  %10.1f ms\n", name,
                         static_cast<double>(ns) * 1e-3 / steps,
                         static_cast<double>(ns) * 1e-6);
        };
        if (stages_mode_) {
            for (std::size_t i = 0; i < names_.size(); ++i) line(names_[i], ns_[i]);
            line("(batch state read)", drain_ns_);
        } else {
            line("host enqueue", enqueue_ns_);
            line("wait for stream", drain_ns_);
        }
    }

private:
    using Clock = std::chrono::steady_clock;
    std::int64_t lap() {
        const auto now = Clock::now();
        const auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(now - last_).count();
        last_ = now;
        return d;
    }
    cudaStream_t              stream_;
    std::vector<const char*>  names_;
    std::vector<std::int64_t> ns_;
    std::int64_t              enqueue_ns_ = 0, drain_ns_ = 0;
    std::uint64_t             steps_ = 0;
    bool                      stages_mode_ = false;
    Clock::time_point         last_;
};

}  // namespace

UscPatchSelectorImpl::UscPatchSelectorImpl(stComm::Comm& comm)
    : comm_(comm) {
    if (!comm_.hasDevice()) {
        throw std::invalid_argument(
            "UscPatchSelectorImpl requires a device-enabled Comm (Comm::onDevice) — "
            "USC is GPU-only.");
    }
}

PatchSelection UscPatchSelectorImpl::patch_select(
        std::vector<std::vector<Hash>> raw_patches,
        std::vector<PatchId>           global_ids) {
    patch_global_ids_ = std::move(global_ids);
    using W = detail::PhaseTimer::Where;
    detail::PhaseTimer phases;

    // Shared preprocessing: distributed hash→id mapping + PatchCsr + InvertedIndex
    // + device mirrors. USC owns only the score/covered state initialized below.
    patch_set_ = std::make_unique<PatchSet>(comm_, std::move(raw_patches));
    phases.mark("PatchSet setup (detailed above)", W::Device);

    const std::uint64_t M_local = patch_set_->M_local();
    const std::uint64_t N       = patch_set_->N();

    // USC-specific device state: initial scores = patch sizes; covered empty.
    d_scores_.resize(M_local);
    d_covered_.resize((N + 63) / 64);
    std::uint64_t local_max = 0;
    if (M_local) {
        const PatchCsr& patches = patch_set_->patches();
        std::vector<WireScore> initial_scores(M_local);
        for (std::size_t p = 0; p < M_local; ++p) {
            initial_scores[p] = static_cast<WireScore>(patches.patch_size(p));
            local_max = std::max<std::uint64_t>(local_max, patches.patch_size(p));
        }
        d_scores_.copy_from_host(initial_scores.data(), M_local);
        d_global_ids_.resize(M_local);
        d_global_ids_.copy_from_host(patch_global_ids_.data(), M_local);
    }
    d_covered_.zero();
    // Loop sizing: the largest initial score bounds every winner's list, the
    // patch total bounds the number of selections.
    comm_.allreduce<stComm::Space::Host, std::uint64_t>(
        &local_max, &max_initial_score_, 1, stComm::ReduceOp::Max)->wait();
    comm_.allreduce<stComm::Space::Host, std::uint64_t>(
        &M_local, &M_global_, 1, stComm::ReduceOp::Sum)->wait();
    phases.mark("initial scores", W::Transfer);

    PatchSelection result = select();
    phases.mark("greedy loop", W::Device);
    phases.report(stderr, "[stps-profile usc by-patch]", comm_.getRank());
    return result;
}

// Patch-partitioned greedy loop — DEVICE-RESIDENT, several selections per
// round (see "ByPatch device-resident loop" above for the rule and why it
// reproduces sequential greedy exactly). The host enqueues a batch of rounds
// on the NCCL stream and syncs once per batch to read the 24 B loop state.
//
// Per round:
//   [device] refresh block maxima; extract this rank's K best patches
//        ncclAllReduce<SUM>(P segments of K keys)       — everyone's candidates
//   [device] merge → the K best overall; owners write their candidates'
//            still-uncovered elements into the payload slots
//        ncclAllReduce<SUM>(K slots of bound + 2 words) — only owners non-zero
//   [device] walk (accept / skip / end), restore the unselected candidates'
//            scores, strategy-A score update over all accepted elements
//
// Two collectives per round, as the one-selection loop had per selection.
// `bound` is the previous batch's last accepted score — no later score
// exceeds it — and K is as many candidates as fit the payload budget at that
// bound. size == 1 skips both collectives.
//
// -DUSC_PROFILE: see LoopProfiler.
PatchSelection UscPatchSelectorImpl::select() {
    PatchSelection result;
    result.covered_count = 0;
    result.iterations    = 0;
    const std::uint64_t N = patch_set_->N();
    if (M_global_ == 0 || N == 0) return result;   // agreed on every rank

    const int           size  = comm_.getSize();
    const int           rank  = comm_.getRank();
    const auto          usize = static_cast<std::uint64_t>(size);
    const std::uint64_t M_loc = patch_set_->M_local();
    if (usize > kKeyField) {   // same on every rank
        throw std::invalid_argument("ByPatch mode: more than 65535 ranks");
    }
    cudaStream_t stream = comm_.nccl().getStream();

    DeviceBuffer<LoopState>  d_state(1);
    DeviceBuffer<WinnerInfo> d_winner(1);
    d_state.zero();
    d_winner.zero();
    // Every selection covers at least one new element.
    DeviceBuffer<PatchId> d_selected(std::min(M_global_, N));

    // Round sizing. A payload slot is bound + 2 words; K candidates must fit
    // the budget, so K grows as the scores (and with them the lists) shrink.
    constexpr std::uint64_t kPayloadBudget = std::uint64_t{1} << 15;   // words
    const std::uint64_t hook_k = detail::usc_patch_test_hooks().max_candidates;
    const std::uint64_t k_cap  = hook_k ? hook_k : 256;                // <= kTopThreads
    const std::uint64_t max_bound = std::max<std::uint64_t>(max_initial_score_, 1);
    const auto candidates_for = [&](std::uint64_t bound) {
        return static_cast<unsigned int>(
            std::clamp<std::uint64_t>(kPayloadBudget / (bound + 2), 1, k_cap));
    };
    const std::uint64_t payload_words = std::max(kPayloadBudget, max_bound + 2);

    DeviceBuffer<int>           d_cand_patch(k_cap), d_slot_of(k_cap);
    DeviceBuffer<WireScore>     d_cand_score(k_cap);
    DeviceBuffer<unsigned int>  d_accepted(k_cap);
    DeviceBuffer<std::uint64_t> d_global_keys(k_cap);
    DeviceBuffer<std::uint64_t> d_keys_send(usize * k_cap);
    DeviceBuffer<std::uint64_t> d_keys_recv_buf(size > 1 ? usize * k_cap : 0);
    DeviceBuffer<std::uint64_t> d_pay_send(payload_words);
    DeviceBuffer<std::uint64_t> d_pay_recv_buf(size > 1 ? payload_words : 0);
    DeviceBuffer<ElementId>     d_newly(payload_words);
    d_keys_send.zero();
    // One rank: the "reduction" of a single contribution is itself.
    const std::uint64_t* d_keys_recv = size > 1 ? d_keys_recv_buf.data() : d_keys_send.data();
    const std::uint64_t* d_pay_recv  = size > 1 ? d_pay_recv_buf.data()  : d_pay_send.data();

    BlockArgmax block_argmax(d_scores_.data(), M_loc);

    enum { kPRefresh, kPExtract, kPKeyReduce, kPMerge, kPBuild, kPPayloadReduce, kPWalk,
           kPRestore, kPScoreUpdate };
    LoopProfiler prof(stream, {"refresh blocks", "extract candidates", "key reduce",
                               "merge (+clears)", "build payload", "payload reduce", "walk",
                               "restore", "score update"});

    constexpr std::uint64_t kMaxBatch = 16;   // rounds
    std::uint64_t bound = max_bound;
    unsigned int  K     = 0;
    LoopState h_state{};
    while (true) {
        const unsigned int k_now = candidates_for(bound);
        if (k_now != K) {
            // Segments move with K: the other ranks' segments of the send
            // buffer must read zero again.
            cuda_check(cudaMemsetAsync(d_keys_send.data(), 0, d_keys_send.bytes(), stream),
                       "key send clear");
            K = k_now;
        }
        const std::uint64_t slot_words = bound + 2;
        // A round covers at most K * bound elements, so at least this many
        // more rounds must run — enqueueing them never overshoots the end.
        const std::uint64_t remaining = N - h_state.covered_count;
        const std::uint64_t per_round = static_cast<std::uint64_t>(K) * bound;
        const std::uint64_t batch = std::clamp<std::uint64_t>(
            remaining / per_round + (remaining % per_round != 0), 1, kMaxBatch);
        std::uint64_t* d_key_segment = d_keys_send.data() + static_cast<std::uint64_t>(rank) * K;
        const unsigned merge_blocks = static_cast<unsigned>(
            (usize * K + kFixedBlockThreads - 1) / kFixedBlockThreads);

        for (std::uint64_t b = 0; b < batch; ++b) {
            block_argmax.refresh(stream);
            prof.mark(kPRefresh);
            extract_candidates_kernel<<<1, kTopThreads, 0, stream>>>(
                d_scores_.data(), M_loc, block_argmax.view(), block_argmax.num_blocks(),
                K, rank, d_cand_patch.data(), d_cand_score.data(), d_key_segment);
            prof.mark(kPExtract);
            if (size > 1) {
                comm_.allreduce<stComm::Space::Device, std::uint64_t>(
                    d_keys_send.data(), d_keys_recv_buf.data(), usize * K,
                    stComm::ReduceOp::Sum);
            }
            prof.mark(kPKeyReduce);
            cuda_check(cudaMemsetAsync(d_global_keys.data(), 0, K * sizeof(std::uint64_t), stream),
                       "global key clear");
            cuda_check(cudaMemsetAsync(d_slot_of.data(), 0xFF, K * sizeof(int), stream),
                       "slot clear");
            cuda_check(cudaMemsetAsync(d_pay_send.data(), 0,
                                       K * slot_words * sizeof(std::uint64_t), stream),
                       "payload clear");
            patch_merge_candidates_kernel<<<merge_blocks, kFixedBlockThreads, 0, stream>>>(
                d_keys_recv, static_cast<unsigned int>(size), K, rank,
                d_global_keys.data(), d_slot_of.data());
            prof.mark(kPMerge);
            patch_build_payload_kernel<<<kFixedGridBlocks, kFixedBlockThreads, 0, stream>>>(
                patch_set_->d_patch_data(), patch_set_->d_patch_offsets(),
                d_global_ids_.data(), d_covered_.data(),
                d_cand_patch.data(), d_slot_of.data(), K, slot_words, d_pay_send.data());
            prof.mark(kPBuild);
            if (size > 1) {
                comm_.allreduce<stComm::Space::Device, std::uint64_t>(
                    d_pay_send.data(), d_pay_recv_buf.data(), K * slot_words,
                    stComm::ReduceOp::Sum);
            }
            prof.mark(kPPayloadReduce);
            patch_walk_kernel<<<1, kTopThreads, 0, stream>>>(
                d_global_keys.data(), d_pay_recv, K, slot_words, N,
                d_covered_.data(), d_accepted.data(), d_newly.data(),
                d_selected.data(), d_state.data(), d_winner.data());
            prof.mark(kPWalk);
            restore_candidates_kernel<<<(K + kFixedBlockThreads - 1) / kFixedBlockThreads,
                                   kFixedBlockThreads, 0, stream>>>(
                d_cand_patch.data(), d_cand_score.data(), d_key_segment, d_slot_of.data(),
                d_accepted.data(), K, d_scores_.data(), block_argmax.view());
            prof.mark(kPRestore);
            fixed_score_update_kernel
                <<<kFixedGridBlocks, kScoreBlockThreads, 0, stream>>>(
                d_newly.data(), d_winner.data(),
                patch_set_->d_inv_keys(), patch_set_->inv_keys().size(),
                patch_set_->d_inv_offsets(), patch_set_->d_inv_data(),
                d_scores_.data(), block_argmax.view());
            prof.mark(kPScoreUpdate);
            prof.step();
        }
        cuda_launch_check("device-resident batch (patch)");
        prof.batch_enqueued();
        // Drain through stComm (polled: also watches for async NCCL failures),
        // then read the state — see the ByElement loop.
        comm_.nccl().barrier();
        cuda_check(cudaMemcpy(&h_state, d_state.data(), sizeof(h_state),
                              cudaMemcpyDeviceToHost),
                   "device-resident batch (patch) state copy");
        prof.batch_drained();
        if (h_state.done) break;
        bound = std::max<std::uint64_t>(h_state.last_score, 1);
    }

    prof.report(stderr, "[stps-profile usc by-patch loop]", rank);
    result.covered_count = h_state.covered_count;
    result.iterations    = h_state.num_selected;
    if (h_state.num_selected) {
        result.selected.resize(h_state.num_selected);
        cuda_check(cudaMemcpy(result.selected.data(), d_selected.data(),
                              h_state.num_selected * sizeof(PatchId), cudaMemcpyDeviceToHost),
                   "selected patches copy");
    }
    return result;
}

namespace detail {
UscPatchTestHooks& usc_patch_test_hooks() {
    static UscPatchTestHooks hooks;
    return hooks;
}
}  // namespace detail

void UscPatchSelectorImpl::print_selection(const PatchSelection& r) const {
    if (comm_.getRank() != 0) return;
    const std::uint64_t N = patch_set_->N();
    std::cout << "USC patch-select\n"
              << "  universe N=" << N
              << " ranks=" << comm_.getSize() << "\n"
              << "  result: selected=" << r.selected.size()
              << " covered=" << r.covered_count << "/" << N
              << " iterations=" << r.iterations << "\n";
}

std::uint64_t        UscPatchSelectorImpl::N() const            { return patch_set_->N(); }
std::uint64_t        UscPatchSelectorImpl::M_local() const      { return patch_set_->M_local(); }
const PatchCsr&      UscPatchSelectorImpl::patches() const      { return patch_set_->patches(); }

// ============================================================================
// UscElemPatchSelectorImpl — PartitionMode::ByElement (design doc §7.3)
// ============================================================================

UscElemPatchSelectorImpl::UscElemPatchSelectorImpl(stComm::Comm& comm)
    : comm_(comm) {
    if (!comm_.hasDevice()) {
        throw std::invalid_argument(
            "UscElemPatchSelectorImpl requires a device-enabled Comm (Comm::onDevice) — "
            "USC is GPU-only.");
    }
}

PatchSelection UscElemPatchSelectorImpl::patch_select(
        std::vector<std::vector<Hash>> raw_patches,
        std::vector<PatchId>           global_ids) {
    const int size = comm_.getSize();
    const int rank = comm_.getRank();
    using W = detail::PhaseTimer::Where;
    detail::PhaseTimer phases;

    // Shared §5.1 preprocessing: hash → dense ElementId translation.
    // The mapping runs on the device; the translated patches come back as a
    // host CSR for the redistribution below (still host-side).
    PatchCsr id_csr;
    {
        DeviceIdMappedPatches mapped =
            map_hashes_to_element_ids_device(comm_, std::move(raw_patches));
        N_ = mapped.N;
        phases.mark("hash->id map (detailed above)", W::Device);
        id_csr.offsets.resize(mapped.M_local + 1);
        mapped.offsets.copy_to_host(id_csr.offsets.data(), id_csr.offsets.size());
        id_csr.data.resize(mapped.data.size());
        if (!id_csr.data.empty()) {
            mapped.data.copy_to_host(id_csr.data.data(), id_csr.data.size());
        }
        phases.mark("id patches download", W::Transfer);
    }

    // ---- Global slot assignment (rank-major input order) --------------------
    // Slot of local patch p on rank r = (Σ M_local below r) + p. With
    // slice_patches_by_rank's contiguous IDs this equals ascending global-ID
    // order, so the smallest-slot tie-break in select() matches
    // brute_force_select's smallest-PatchId rule (and ByPatch's
    // smallest-rank-then-smallest-local-index MAXLOC rule).
    const std::uint64_t M_local = id_csr.M();
    std::uint64_t slot_start = 0;
    comm_.exscan<stComm::Space::Host>(M_local, &slot_start, stComm::ReduceOp::Sum)->wait();
    std::uint64_t M_global = (rank == size - 1) ? slot_start + M_local : 0;
    comm_.bcast<stComm::Space::Host, std::uint64_t>(&M_global, 1, size - 1)->wait();
    M_global_ = M_global;
    if (M_global_ > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "ByElement mode: M_global exceeds CUB ArgMax's int index range");
    }

    // Replicated slot → caller PatchId map (used only to report the result).
    std::vector<std::size_t> mcounts(static_cast<std::size_t>(size), 0);
    {
        std::vector<int> ones(static_cast<std::size_t>(size), 1);
        const std::size_t m_local = M_local;
        comm_.allgatherv<stComm::Space::Host, std::size_t>(&m_local, 1,
                                                           mcounts.data(), ones.data())->wait();
    }
    slot_to_gid_.resize(M_global_);
    comm_.allgatherv<stComm::Space::Host, PatchId>(global_ids.data(), mcounts[static_cast<std::size_t>(rank)],
                                                   slot_to_gid_.data(), mcounts.data())->wait();

    // ---- Shard geometry ------------------------------------------------------
    // Static contiguous ranges: rank r owns [r·⌈N/P⌉, (r+1)·⌈N/P⌉) ∩ [0, N).
    // Static (not hash-based) is enough: the §5.1 sample sort already assigns
    // dense IDs with balanced per-rank density.
    const std::uint64_t P          = static_cast<std::uint64_t>(size);
    const std::uint64_t shard_size = (N_ + P - 1) / P;
    shard_base_ = std::min<std::uint64_t>(static_cast<std::uint64_t>(rank) * shard_size, N_);
    const std::uint64_t shard_end =
        std::min<std::uint64_t>((static_cast<std::uint64_t>(rank) + 1) * shard_size, N_);
    shard_extent_ = shard_end - shard_base_;

    phases.mark("slot assignment", W::HostComm);

    // ---- Incidence redistribution --------------------------------------------
    // Every (slot, element) incidence flies to the element's owner shard. Each
    // rank then holds, for ALL M_global patches, the sub-list inside its shard —
    // every incidence stored on exactly one rank (M·K/P per rank), tighter than
    // §7.5's conservative full patch replication.
    const auto usize = static_cast<std::size_t>(size);
    const auto owner_of = [shard_size](ElementId e) {
        return shard_size ? static_cast<std::size_t>(e / shard_size) : std::size_t{0};
    };
    // Counted in 64 bits and shipped with stComm's size_t-count exchange: a
    // rank's incidences pass INT_MAX on a full-chip input, and nothing here
    // caps them.
    std::vector<std::size_t> sendcounts(usize, 0);
    for (ElementId e : id_csr.data) ++sendcounts[owner_of(e)];

    // Exchange the counts so every rank learns its recv counts.
    const ExchangeCounts counts = exchange_counts(comm_, std::move(sendcounts));

    std::vector<std::size_t> cursor(usize, 0);  // per-target write head
    for (std::size_t r = 1; r < usize; ++r) cursor[r] = cursor[r - 1] + counts.send[r - 1];
    const std::size_t total_send =
        std::accumulate(counts.send.begin(), counts.send.end(), std::size_t{0});
    std::vector<PatchId>   send_slots(total_send);
    std::vector<ElementId> send_elems(total_send);
    for (std::uint64_t p = 0; p < M_local; ++p) {
        const PatchId slot = slot_start + p;
        for (ElementId e : id_csr.patch(p)) {
            const std::size_t at = cursor[owner_of(e)]++;
            send_slots[at] = slot;
            send_elems[at] = e;
        }
    }
    id_csr = PatchCsr{};

    // Ship the incidences as two parallel alltoallv's sharing one count layout.
    const std::size_t total_recv =
        std::accumulate(counts.recv.begin(), counts.recv.end(), std::size_t{0});

    std::vector<PatchId>   recv_slots(total_recv);
    std::vector<ElementId> recv_elems(total_recv);
    comm_.alltoallv<stComm::Space::Host, PatchId>(send_slots.data(), counts.send.data(),
                                                  recv_slots.data(), counts.recv.data())->wait();
    comm_.alltoallv<stComm::Space::Host, ElementId>(send_elems.data(), counts.send.data(),
                                                    recv_elems.data(), counts.recv.data())->wait();
    send_slots.clear(); send_slots.shrink_to_fit();
    send_elems.clear(); send_elems.shrink_to_fit();

    phases.mark("incidence redistribution", W::HostComm);

    // ---- Shard-local sub-CSR over all M_global patches -----------------------
    // Store SHARD-LOCAL element IDs (global − shard_base_): the covered bitset
    // and every kernel then index the local shard exactly like ByPatch indexes
    // the full universe. Incidences are globally unique, so no dedupe needed.
    struct Incidence { PatchId slot; ElementId elem; };
    std::vector<Incidence> inc(total_recv);
    for (std::size_t i = 0; i < total_recv; ++i)
        inc[i] = { recv_slots[i], recv_elems[i] - shard_base_ };
    recv_slots.clear(); recv_slots.shrink_to_fit();
    recv_elems.clear(); recv_elems.shrink_to_fit();
    std::sort(inc.begin(), inc.end(), [](const Incidence& a, const Incidence& b) {
        return a.slot != b.slot ? a.slot < b.slot : a.elem < b.elem;
    });

    PatchCsr sub;
    sub.offsets.assign(M_global_ + 1, 0);
    sub.data.resize(total_recv);
    for (const Incidence& x : inc) ++sub.offsets[x.slot + 1];
    for (std::uint64_t s = 0; s < M_global_; ++s) sub.offsets[s + 1] += sub.offsets[s];
    for (std::size_t i = 0; i < inc.size(); ++i) sub.data[i] = inc[i].elem;
    inc.clear(); inc.shrink_to_fit();

    phases.mark("sub-CSR build", W::Host);

    // ---- Device mirrors + algorithm state -------------------------------------
    // Shard-local inverted index (keys are shard-local IDs, values are global
    // slots) — drives the strategy-A partial-score update. Built on the device.
    {
        DeviceInvertedIndex inv =
            build_inverted_index_device(sub, shard_extent_ ? shard_extent_ : 1);
        num_inv_keys_  = inv.num_keys;
        d_inv_keys_    = std::move(inv.keys);
        d_inv_offsets_ = std::move(inv.offsets);
        d_inv_data_    = std::move(inv.data);
    }
    phases.mark("inverted index build", W::Device);
    if (!sub.data.empty()) {
        d_patch_data_.resize(sub.data.size());
        d_patch_data_.copy_from_host(sub.data.data(), sub.data.size());
    }

    d_scores_.resize(M_global_);
    max_sub_len_ = 0;
    if (M_global_) {
        // This shard's slice of each patch; the full initial scores are their
        // sum over the ranks (below). Also record the widest sub-patch — it
        // bounds the newly-covered scratch for the whole solve.
        std::vector<WireScore> init(M_global_);
        for (std::uint64_t s = 0; s < M_global_; ++s) {
            const std::uint64_t len = sub.offsets[s + 1] - sub.offsets[s];
            init[s] = static_cast<WireScore>(len);
            max_sub_len_ = std::max(max_sub_len_, len);
        }
        // Offsets mirror — the commit kernel resolves winner ranges on device,
        // so the host never touches the sub-CSR during the loop.
        d_sub_offsets_.resize(sub.offsets.size());
        d_sub_offsets_.copy_from_host(sub.offsets.data(), sub.offsets.size());

        // Full initial scores, replicated on every rank: one host allreduce of
        // the per-shard slices, in int64 so the int32 fit can be proven
        // exactly. Initial scores (= deduped patch sizes) are the solve's
        // maxima — scores only decrease — and the disabled floor is
        // −1 − patch_size, so if every initial score fits with a margin, every
        // later value fits.
        std::vector<std::int64_t> init64(M_global_), sum64(M_global_);
        for (std::uint64_t s = 0; s < M_global_; ++s) init64[s] = init[s];
        comm_.allreduce<stComm::Space::Host, std::int64_t>(
            init64.data(), sum64.data(), M_global_, stComm::ReduceOp::Sum)->wait();
        const std::int64_t wire_max =
            static_cast<std::int64_t>(std::numeric_limits<WireScore>::max()) - size - 1;
        // The largest initial score bounds every later winner (scores only
        // decrease) — select() sizes its first batch from it.
        max_initial_score_ = static_cast<std::uint64_t>(
            std::max<std::int64_t>(*std::max_element(sum64.begin(), sum64.end()), 0));
        std::uint64_t incidences = 0;
        for (std::uint64_t s = 0; s < M_global_; ++s) {
            if (sum64[s] > wire_max) {
                throw std::invalid_argument(
                    "ByElement mode: patch " + std::to_string(slot_to_gid_[s]) +
                    " has " + std::to_string(sum64[s]) +
                    " unique elements — exceeds the int32 score wire format");
            }
            init[s] = static_cast<WireScore>(sum64[s]);
            incidences += static_cast<std::uint64_t>(sum64[s]);
        }
        d_scores_.copy_from_host(init.data(), M_global_);
        // Average patches per element, rounded up — sizes the delta window.
        avg_degree_ = N_ ? (incidences + N_ - 1) / N_ : 1;
    }

    d_covered_.resize((shard_extent_ + 63) / 64);
    d_covered_.zero();
    phases.mark("upload + initial scores", W::Transfer);

    PatchSelection result = select();
    phases.mark("greedy loop", W::Device);
    phases.report(stderr, "[stps-profile usc by-element]", rank);
    return result;
}

// Element-partitioned greedy loop (§7.3) — DEVICE-RESIDENT, several
// selections per round.
//
// Every rank holds the full, identical score of every patch, so all ranks
// extract the same K candidates on their own. What travels is (1) which
// candidates share still-uncovered elements — a K x K table of counts, each
// rank contributing its shard — and (2) what the selections change: the patch
// slots whose score drops (see "ByElement delta exchange"). Per step, all
// enqueued on the NCCL stream with no host knowledge of the outcome:
//
//   [device] plan + fill this rank's window of pending decrements
//        ncclAllReduce<SUM>(P segments of window + 2 words)
//   [device] apply every rank's window to the scores
//   [device] refresh block maxima, extract the K best patches (scores parked)
//   [device] count shared uncovered elements per candidate pair, this shard
//        ncclAllReduce<SUM>(K x K words)
//   [device] walk: accept / skip / end — nothing while any rank has more
//            decrements to send; this shard's elements of the accepted
//            patches → newly covered + covered bits; restore parked scores
//   [device] new delta stream: list lengths → prefix sum → cursor
//
// Two collectives per step. The host enqueues a batch of steps and syncs
// once to read the 24 B loop state; steps enqueued past termination no-op,
// and every branch derives from reduced or replicated values, so all ranks
// stay in lockstep.
//
// K is bounded so that the pair counting (K^2 lookups per element of a
// sub-patch) stays cheap; the delta window is re-sized per batch from values
// every rank agrees on.
//
// size == 1 has no one to exchange with: no collective at all, and the
// decrements go straight into the scores (strategy-A update).
//
// -DUSC_PROFILE: see LoopProfiler.
PatchSelection UscElemPatchSelectorImpl::select() {
    PatchSelection result;
    result.covered_count = 0;
    result.iterations    = 0;
    if (M_global_ == 0 || N_ == 0) return result;   // agreed on every rank

    const int  size  = comm_.getSize();
    const auto usize = static_cast<std::uint64_t>(size);
    const auto urank = static_cast<std::uint64_t>(comm_.getRank());
    const detail::UscElemTestHooks& hooks = detail::usc_elem_test_hooks();
    const bool exchange = size > 1 || hooks.force_delta_exchange;
    // Kernels ride the same stream as the collectives, so ordering is carried
    // by the stream alone — no events, no per-step syncs.
    cudaStream_t stream = comm_.nccl().getStream();

    DeviceBuffer<LoopState>  d_state(1);
    DeviceBuffer<WinnerInfo> d_winner(1);
    d_state.zero();
    d_winner.zero();
    DeviceBuffer<PatchId> d_selected(std::min(M_global_, N_));

    // Candidates per round: the pair counting does K^2 lookups per element of
    // a sub-patch, so K shrinks as the widest sub-patch grows.
    const std::uint64_t max_sub = std::max<std::uint64_t>(max_sub_len_, 1);
    std::uint64_t k_fit = 1;
    while (k_fit < 256 && (2 * k_fit) * (2 * k_fit) * max_sub <= (std::uint64_t{1} << 26)) k_fit *= 2;
    const auto K = static_cast<unsigned int>(
        hooks.max_candidates ? std::min<std::uint64_t>(hooks.max_candidates, kTopThreads) : k_fit);
    const std::uint64_t newly_cap = static_cast<std::uint64_t>(K) * max_sub;

    DeviceBuffer<int>           d_cand_patch(K);
    DeviceBuffer<WireScore>     d_cand_score(K);
    DeviceBuffer<std::uint64_t> d_keys(K);
    DeviceBuffer<unsigned int>  d_accepted(K);
    DeviceBuffer<DeltaWord>     d_overlap_send(static_cast<std::uint64_t>(K) * K);
    DeviceBuffer<DeltaWord>     d_overlap_recv_buf(size > 1 ? static_cast<std::uint64_t>(K) * K : 0);
    const DeltaWord* d_overlap = size > 1 ? d_overlap_recv_buf.data() : d_overlap_send.data();
    DeviceBuffer<ElementId>     d_newly_ids(newly_cap);

    BlockArgmax block_argmax(d_scores_.data(), M_global_);

    std::uint64_t score_bound = std::max<std::uint64_t>(max_initial_score_, 1);

    // Delta exchange state. The window only shrinks (score_bound does), so the
    // first one sizes the buffers. About four times the decrements one rank
    // expects from a full round.
    const auto window_for = [&](std::uint64_t bound) -> std::uint64_t {
        if (hooks.delta_window) return hooks.delta_window;
        const std::uint64_t expected = K * bound * avg_degree_ / usize;
        return std::clamp<std::uint64_t>(4 * expected + 16, 64, std::uint64_t{1} << 18);
    };
    DeviceBuffer<DeltaCursor>   d_cursor;
    DeviceBuffer<std::uint64_t> d_deg, d_prefix;
    DeviceBuffer<DeltaWord>     d_send, d_recv_buf;
    DeviceBuffer<std::uint8_t>  d_scan_temp;
    std::uint64_t window = 0;
    if (exchange) {
        d_cursor.resize(1);
        d_cursor.zero();
        d_deg.resize(newly_cap + 1);
        d_prefix.resize(newly_cap + 1);
        d_deg.zero();
        d_prefix.zero();
        const std::uint64_t words = usize * (window_for(score_bound) + 2);
        d_send.resize(words);
        if (size > 1) d_recv_buf.resize(words);
        std::size_t temp_bytes = 0;
        cuda_check(cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_deg.data(),
                                                 d_prefix.data(), newly_cap + 1),
                   "delta prefix-sum sizing");
        d_scan_temp.resize(temp_bytes);
    }
    // One rank (test hook): the "reduction" of a single segment is itself.
    const DeltaWord* d_recv = size > 1 ? d_recv_buf.data() : d_send.data();

    constexpr std::uint64_t kMaxBatch = 16;   // steps
    enum { kEPlan, kEFill, kEReduce, kEApply, kERefresh, kEExtract, kEOverlap, kEOverlapReduce,
           kEWalk, kEBuild, kERestore, kEDegrees, kEScan, kEBegin, kEScoreUpdate };
    LoopProfiler prof(stream, {"delta plan", "delta fill", "delta reduce", "delta apply",
                               "refresh blocks", "extract candidates", "pair overlaps",
                               "overlap reduce", "walk", "build accepted", "restore",
                               "delta degrees", "delta prefix scan", "delta begin",
                               "score update (1 rank)"});
    LoopState h_state{};
    while (true) {
        // A step covers at most K * score_bound elements, so at least this
        // many more steps must run — enqueueing them never overshoots.
        const std::uint64_t remaining = N_ - h_state.covered_count;
        const std::uint64_t per_step  = static_cast<std::uint64_t>(K) * score_bound;
        const std::uint64_t batch = std::clamp<std::uint64_t>(
            remaining / per_step + (remaining % per_step != 0), 1, kMaxBatch);
        // A step's newly-covered list never outgrows K winners' scores or K
        // sub-patches.
        const std::uint64_t len = std::min(per_step, newly_cap);
        if (exchange) {
            const std::uint64_t w = window_for(score_bound);
            if (w != window) {
                // Segments move with the window: the other ranks' segments of
                // the send buffer must read zero again.
                cuda_check(cudaMemsetAsync(d_send.data(), 0, d_send.bytes(), stream),
                           "delta send clear");
                window = w;
            }
        }
        DeltaWord* d_segment = exchange ? d_send.data() + urank * (window + 2) : nullptr;
        const unsigned apply_blocks = static_cast<unsigned>(
            (usize * window + kFixedBlockThreads - 1) / kFixedBlockThreads);

        for (std::uint64_t b = 0; b < batch; ++b) {
            if (exchange) {
                elem_delta_plan_kernel<<<1, 1, 0, stream>>>(d_cursor.data(), window, d_segment);
                prof.mark(kEPlan);
                elem_delta_fill_kernel<<<kFixedGridBlocks, kFixedBlockThreads, 0, stream>>>(
                    d_newly_ids.data(), d_cursor.data(), d_prefix.data(),
                    d_inv_keys_.data(), num_inv_keys_, d_inv_offsets_.data(),
                    d_inv_data_.data(), d_segment + 2);
                prof.mark(kEFill);
                if (size > 1) {
                    // Enqueue-only — no wait.
                    comm_.allreduce<stComm::Space::Device, DeltaWord>(
                        d_send.data(), d_recv_buf.data(), usize * (window + 2),
                        stComm::ReduceOp::Sum);
                }
                prof.mark(kEReduce);
                elem_delta_apply_kernel<<<apply_blocks, kFixedBlockThreads, 0, stream>>>(
                    d_recv, usize, window, d_scores_.data(), block_argmax.view());
                prof.mark(kEApply);
            }
            block_argmax.refresh(stream);
            prof.mark(kERefresh);
            extract_candidates_kernel<<<1, kTopThreads, 0, stream>>>(
                d_scores_.data(), M_global_, block_argmax.view(), block_argmax.num_blocks(),
                K, 0, d_cand_patch.data(), d_cand_score.data(), d_keys.data());
            prof.mark(kEExtract);
            cuda_check(cudaMemsetAsync(d_overlap_send.data(), 0, d_overlap_send.bytes(), stream),
                       "overlap clear");
            elem_overlap_kernel<<<kFixedGridBlocks, kFixedBlockThreads, 0, stream>>>(
                d_patch_data_.data(), d_sub_offsets_.data(), d_covered_.data(),
                d_cand_patch.data(), d_keys.data(), K, d_overlap_send.data());
            prof.mark(kEOverlap);
            if (size > 1) {
                comm_.allreduce<stComm::Space::Device, DeltaWord>(
                    d_overlap_send.data(), d_overlap_recv_buf.data(),
                    static_cast<std::uint64_t>(K) * K, stComm::ReduceOp::Sum);
            }
            prof.mark(kEOverlapReduce);
            elem_walk_kernel<<<1, kTopThreads, 0, stream>>>(
                d_cand_patch.data(), d_cand_score.data(), d_keys.data(), d_overlap, K, N_,
                d_recv, exchange ? size : 0, window,
                d_accepted.data(), d_selected.data(), d_state.data(), d_winner.data());
            prof.mark(kEWalk);
            elem_build_accepted_kernel<<<kFixedGridBlocks, kFixedBlockThreads, 0, stream>>>(
                d_patch_data_.data(), d_sub_offsets_.data(), d_covered_.data(),
                d_cand_patch.data(), d_accepted.data(), K,
                d_newly_ids.data(), d_winner.data());
            prof.mark(kEBuild);
            restore_candidates_kernel<<<(K + kFixedBlockThreads - 1) / kFixedBlockThreads,
                                        kFixedBlockThreads, 0, stream>>>(
                d_cand_patch.data(), d_cand_score.data(), d_keys.data(), nullptr,
                d_accepted.data(), K, d_scores_.data(), block_argmax.view());
            prof.mark(kERestore);
            if (exchange) {
                elem_delta_degrees_kernel<<<kFixedGridBlocks, kFixedBlockThreads, 0, stream>>>(
                    d_newly_ids.data(), d_winner.data(),
                    d_inv_keys_.data(), num_inv_keys_, d_inv_offsets_.data(),
                    len, d_deg.data());
                prof.mark(kEDegrees);
                std::size_t scan_bytes = d_scan_temp.bytes();
                cub::DeviceScan::ExclusiveSum(d_scan_temp.data(), scan_bytes, d_deg.data(),
                                              d_prefix.data(), len + 1, stream);
                prof.mark(kEScan);
                elem_delta_begin_kernel<<<1, 1, 0, stream>>>(
                    d_winner.data(), d_prefix.data(), len, d_cursor.data());
                prof.mark(kEBegin);
            } else {
                fixed_score_update_kernel<<<kFixedGridBlocks, kScoreBlockThreads, 0, stream>>>(
                    d_newly_ids.data(), d_winner.data(),
                    d_inv_keys_.data(), num_inv_keys_,
                    d_inv_offsets_.data(), d_inv_data_.data(),
                    d_scores_.data(), block_argmax.view());
                prof.mark(kEScoreUpdate);
            }
            prof.step();
        }
        cuda_launch_check("device-resident batch (elem)");
        prof.batch_enqueued();
        // Drain the stream through stComm, not cudaStreamSynchronize: the
        // batch has allreduces in flight, and stComm's polled sync also
        // watches the communicator for async failures (dead peer, network
        // fault) — a plain synchronize would hang on those until walltime.
        // Throws on a CUDA or NCCL error. Only then read the state: a copy
        // into pageable host memory enqueued behind the batch would itself
        // block until the batch finished, before the polled drain ever ran.
        comm_.nccl().barrier();
        cuda_check(cudaMemcpy(&h_state, d_state.data(), sizeof(h_state),
                              cudaMemcpyDeviceToHost),
                   "device-resident batch (elem) state copy");
        prof.batch_drained();
        if (h_state.done) break;
        if (h_state.num_selected) {
            score_bound = std::max<std::uint64_t>(h_state.last_score, 1);
        }
    }

    prof.report(stderr, "[stps-profile usc by-element loop]", comm_.getRank());
    // Drain the device-side result: selected slots → caller PatchIds.
    result.covered_count = h_state.covered_count;
    result.iterations    = h_state.num_selected;
    if (h_state.num_selected) {
        std::vector<PatchId> slots(h_state.num_selected);
        cuda_check(cudaMemcpy(slots.data(), d_selected.data(),
                              h_state.num_selected * sizeof(PatchId), cudaMemcpyDeviceToHost),
                   "selected slots copy");
        result.selected.reserve(slots.size());
        for (PatchId s : slots) result.selected.push_back(slot_to_gid_[s]);
    }
    return result;
}

namespace detail {
UscElemTestHooks& usc_elem_test_hooks() {
    static UscElemTestHooks hooks;
    return hooks;
}
}  // namespace detail

// ============================================================================
// Public facade: stPS::UscPatchSelector — pImpl routing to the ByPatch or
// ByElement implementation (hides all internals).
// ============================================================================

struct UscPatchSelector::Impl {
    // Exactly one is engaged, per the ctor's PartitionMode.
    std::unique_ptr<UscPatchSelectorImpl>     by_patch;
    std::unique_ptr<UscElemPatchSelectorImpl> by_element;
    stComm::Comm& comm;
    Impl(stComm::Comm& c, PartitionMode mode) : comm(c) {
        if (mode == PartitionMode::ByElement) {
            by_element = std::make_unique<UscElemPatchSelectorImpl>(comm);
        } else {
            by_patch = std::make_unique<UscPatchSelectorImpl>(comm);
        }
    }
};

UscPatchSelector::UscPatchSelector(stComm::Comm& comm, PartitionMode mode)
    : impl_(std::make_unique<Impl>(comm, mode)) {}
UscPatchSelector::~UscPatchSelector() = default;
UscPatchSelector::UscPatchSelector(UscPatchSelector&&) noexcept = default;
UscPatchSelector& UscPatchSelector::operator=(UscPatchSelector&&) noexcept = default;

PatchSelection UscPatchSelector::patch_select(std::vector<std::vector<Hash>> patches,
                                              std::vector<PatchId>           global_ids) {
    // Validate this rank's slice, then agree on the verdict: a rank throwing
    // on its own would leave its peers waiting in the setup collectives.
    const auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    std::string err;
    if (global_ids.size() != patches.size()) {
        // Both modes index global_ids by local patch index (ByElement also
        // allgathers it as M_local entries) — a short vector reads out of bounds.
        err = "patch_select: global_ids has " + std::to_string(global_ids.size()) +
              " entries for " + std::to_string(patches.size()) + " patches";
    } else if (patches.size() > int_max) {
        // CUB ArgMax takes an int item count; a truncated one would scan a
        // prefix and silently pick the wrong winner.
        err = "patch_select: " + std::to_string(patches.size()) +
              " patches on one rank exceeds the int index range — use more ranks";
    } else {
        for (std::size_t p = 0; p < patches.size(); ++p) {
            if (patches[p].size() > int_max) {  // kernel grid sizes derive from it
                err = "patch_select: patch " + std::to_string(global_ids[p]) +
                      " has more than INT_MAX elements";
                break;
            }
        }
    }
    if (!all_ranks_ok(impl_->comm, err.empty())) {
        throw std::invalid_argument(
            err.empty() ? std::string("patch_select: invalid input on another rank") : err);
    }
    return impl_->by_element
        ? impl_->by_element->patch_select(std::move(patches), std::move(global_ids))
        : impl_->by_patch->patch_select(std::move(patches), std::move(global_ids));
}

}  // namespace stPS
