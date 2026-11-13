#include "usc/usc_patch_selector_impl.hpp"

#include "core/collective.hpp"
#include "core/cuda_check.hpp"
#include "core/host_profiler.hpp"
#include "core/hash_id_map_device.hpp"
#include "core/inverted_index_device.hpp"

#include <stComm/stComm.h>

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/util_type.cuh>
#include <cuda_runtime.h>

#include <algorithm>
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
// Device-resident loop kernels (Full-GPU iteration track), shared by both
// partition modes.
//
// The host enqueues whole batches of iterations blind, so every launch here
// uses a FIXED grid and reads the winner's range / newly-covered count from
// device memory — the host knows neither when it enqueues. All per-iteration
// control flow (winner resolution, covered_count, termination, winner
// disable) lives in each mode's single-thread commit kernel.
// ---------------------------------------------------------------------------

constexpr int kFixedGridBlocks   = 256;  // fixed grid for the grid-stride kernels
constexpr int kFixedBlockThreads = 128;

// Device-resident loop state — the commit kernel is its only writer; the host
// reads it once per batch (24 B D2H) to decide whether to enqueue more.
struct LoopState {
    std::uint64_t covered_count;
    std::uint64_t num_selected;   // == iterations executed
    std::uint32_t done;
    std::uint32_t last_score;     // most recent winner's score (batch sizing)
};

// Per-iteration winner scratch: written by the commit kernel, consumed by the
// three update kernels of the same (stream-ordered) iteration.
struct WinnerInfo {
    std::uint64_t start;          // winner's shard-local sub-CSR range
    std::uint64_t end;
    unsigned int  newly_count;    // atomic output of the build kernel
    unsigned int  live;           // ByPatch: 1 while this iteration selects a patch
};

// One 32-bit word of the delta exchange (a count, a flag or a patch slot —
// slots fit: M_global is within int range).
using DeltaWord = std::uint32_t;

// Single-thread commit: consume the argmax result entirely on device — the
// decision the host used to make each iteration. Appends the winner slot,
// advances covered_count by the (globally identical) winner score, resolves
// the winner's sub-CSR range for the update kernels, fuses the winner disable,
// and raises `done` on termination. Once done, every later enqueued iteration
// sees start == end and newly_count == 0 and no-ops — a batch may overshoot
// the end of the solve but never changes the result, and `done` derives only
// from allreduced values, so it flips on every rank in the same iteration.
__global__ void elem_commit_winner_kernel(const cub::KeyValuePair<int, WireScore>* argmax,
                                          const std::uint64_t* sub_offsets,
                                          std::uint64_t        N,
                                          WireScore*           scores,
                                          PatchId*             selected_slots,
                                          const DeltaWord*     delta_recv,
                                          int                  delta_ranks,
                                          std::uint64_t        delta_window,
                                          LoopState*           st,
                                          WinnerInfo*          w) {
    w->newly_count = 0;
    w->start = 0;
    w->end   = 0;
    w->live  = 0;
    if (st->done) return;
    // Delta exchange: while any rank still has score decrements to send, the
    // replicated scores are not final — select nothing this step. The flags
    // come out of the reduction, so every rank takes the same branch.
    for (int r = 0; r < delta_ranks; ++r) {
        if (delta_recv[static_cast<std::uint64_t>(r) * (delta_window + 2) + 1]) return;
    }
    const WireScore score = argmax->value;
    if (score <= 0) { st->done = 1; return; }
    const int slot = argmax->key;
    selected_slots[st->num_selected] = static_cast<PatchId>(slot);
    st->num_selected  += 1;
    st->last_score     = static_cast<std::uint32_t>(score);
    st->covered_count += static_cast<std::uint64_t>(score);
    if (st->covered_count >= N) st->done = 1;
    w->live  = 1;
    w->start = sub_offsets[slot];
    w->end   = sub_offsets[slot + 1];
    // Winner disable, fused. This iteration's decrements lower it further — it
    // stays permanently negative either way.
    scores[slot] = kDisabledWireScore;
}

// build_newly_covered, fixed-grid grid-stride: the range comes from w; the
// compact output count is w->newly_count (same non-deterministic order /
// deterministic set as build_newly_covered_kernel).
__global__ void fixed_build_newly_kernel(const ElementId*     patch_data,
                                        const std::uint64_t* covered,
                                        ElementId*           out_ids,
                                        WinnerInfo*      w) {
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t i = w->start
             + blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         i < w->end; i += stride) {
        const ElementId id = patch_data[i];
        if (((covered[id >> 6] >> (id & 63)) & 1ULL) == 0) {
            const unsigned int slot = atomicAdd(&w->newly_count, 1u);
            out_ids[slot] = id;
        }
    }
}

// set_bits, fixed-grid grid-stride over w->newly_count (final by stream order:
// the build kernel completed before this launch starts).
__global__ void fixed_set_bits_kernel(std::uint64_t*        bitset,
                                     const ElementId*      ids,
                                     const WinnerInfo* w) {
    const unsigned int n = w->newly_count;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t i =
             blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         i < n; i += stride) {
        const ElementId id = ids[i];
        atomicOr(reinterpret_cast<unsigned long long*>(&bitset[id >> 6]),
                 static_cast<unsigned long long>(1ULL << (id & 63)));
    }
}

// Strategy-A score update, fixed-grid: blocks stride over the newly covered
// elements. One block resolves an element's patch-list range (thread 0, shared
// to the block), then its threads decrement those patches' scores.
__global__ void fixed_score_update_kernel(const ElementId*      newly_ids,
                                          const WinnerInfo*     w,
                                          const ElementId*      inv_keys,
                                          std::uint64_t         inv_keys_n,
                                          const std::uint64_t*  inv_offsets,
                                          const PatchId*        inv_data,
                                          WireScore*            scores) {
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
            atomicSub(reinterpret_cast<int*>(&scores[inv_data[i]]), 1);
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
                                        std::uint64_t window, WireScore* scores) {
    const std::uint64_t total  = ranks * window;
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t idx = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         idx < total; idx += stride) {
        const DeltaWord* segment = recv + (idx / window) * (window + 2);
        const std::uint64_t j = idx % window;
        if (j < segment[0]) atomicSub(reinterpret_cast<int*>(&scores[segment[2 + j]]), 1);
    }
}

// ---------------------------------------------------------------------------
// ByPatch device-resident loop: the three single-thread kernels around the
// shared fixed-grid ones.
//
// A rank's candidate travels as one 64-bit key, (score << 32) | (2^32-1 - rank):
// the maximum over all ranks is the best score, ties going to the lowest rank
// — and CUB ArgMax already returns the earliest local patch among equal
// scores. 0 means "no candidate". Scores fit 32 bits: a patch holds at most
// INT_MAX elements (validated on entry).
// ---------------------------------------------------------------------------

constexpr std::uint64_t kKeyRankMask = 0xFFFFFFFFull;

__global__ void patch_pack_candidate_kernel(const cub::KeyValuePair<int, WireScore>* argmax,
                                            int has_patches, int my_rank,
                                            std::uint64_t* key) {
    std::uint64_t k = 0;
    if (has_patches && argmax->value > 0) {
        k = (static_cast<std::uint64_t>(argmax->value) << 32)
          | (kKeyRankMask - static_cast<std::uint64_t>(my_rank));
    }
    *key = k;
}

// Consume the global key on every rank: advance the loop state (identical on
// all ranks — it derives only from the reduced key) and, on the winner rank,
// resolve the patch range, disable the patch and put its global ID in the
// payload's last slot. Once done, later iterations of the batch see an empty
// range and a zero count and no-op.
__global__ void patch_commit_winner_kernel(const std::uint64_t* global_key,
                                           const cub::KeyValuePair<int, WireScore>* argmax,
                                           int                  my_rank,
                                           std::uint64_t        N,
                                           const std::uint64_t* patch_offsets,
                                           const PatchId*       patch_global_ids,
                                           WireScore*           scores,
                                           std::uint64_t*       payload,
                                           std::uint64_t        gid_slot,
                                           LoopState*           st,
                                           WinnerInfo*          w) {
    w->newly_count = 0;
    w->start = 0;
    w->end   = 0;
    w->live  = 0;
    if (st->done) return;
    const std::uint64_t key   = *global_key;
    const std::uint64_t score = key >> 32;
    if (score == 0) { st->done = 1; return; }
    st->num_selected  += 1;
    st->last_score     = static_cast<std::uint32_t>(score);
    st->covered_count += score;
    if (st->covered_count >= N) st->done = 1;
    w->live = 1;
    if (kKeyRankMask - (key & kKeyRankMask) == static_cast<std::uint64_t>(my_rank)) {
        const int p = argmax->key;
        w->start = patch_offsets[p];
        w->end   = patch_offsets[p + 1];
        // Fused winner disable; this iteration's score update decrements it
        // further, so it stays negative.
        scores[p] = kDisabledWireScore;
        payload[gid_slot] = patch_global_ids[p];
    }
}

// After the payload reached every rank: record the selected patch and publish
// the newly-covered count (the winner score) for the update kernels.
__global__ void patch_record_winner_kernel(const std::uint64_t* global_key,
                                           const std::uint64_t* payload,
                                           std::uint64_t        gid_slot,
                                           PatchId*             selected,
                                           const LoopState*     st,
                                           WinnerInfo*          w) {
    if (!w->live) { w->newly_count = 0; return; }
    selected[st->num_selected - 1] = payload[gid_slot];
    w->newly_count = static_cast<unsigned int>(*global_key >> 32);
}

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

// Patch-partitioned greedy loop — DEVICE-RESIDENT, like the ByElement one: the
// host enqueues a batch of iterations on the NCCL stream and syncs once per
// batch to read the 24 B loop state. No per-iteration D2H/H2D, no host MPI.
//
// Per iteration:
//   [device] CUB ArgMax over the local scores → pack as a 64-bit key
//        ncclAllReduce<MAX>(key, 1)             — winner score + rank
//   [device] commit: loop state on every rank; the winner resolves its patch
//   [device] winner compacts its still-uncovered elements into the payload
//        ncclAllReduce<SUM>(payload, bound + 1) — only the winner's is non-zero
//   [device] record the selected patch, set_bits, strategy-A score update
//
// The winner's list replaces the old root-and-count broadcast (whose root and
// count had to be host scalars) with a fixed-length reduction: `bound` is the
// previous batch's last winner score, and winner scores never increase, so
// every list of the batch fits. size == 1 skips both collectives.
//
// -DUSC_PROFILE stage timers are absent here for the same reason as in
// ByElement: there are no host-visible stage boundaries inside a batch.
PatchSelection UscPatchSelectorImpl::select() {
    PatchSelection result;
    result.covered_count = 0;
    result.iterations    = 0;
    const std::uint64_t N = patch_set_->N();
    if (M_global_ == 0 || N == 0) return result;   // agreed on every rank

    const int           size  = comm_.getSize();
    const int           rank  = comm_.getRank();
    const std::uint64_t M_loc = patch_set_->M_local();
    cudaStream_t stream = comm_.nccl().getStream();

    DeviceBuffer<LoopState>  d_state(1);
    DeviceBuffer<WinnerInfo> d_winner(1);
    d_state.zero();
    d_winner.zero();
    // Every selection covers at least one new element.
    DeviceBuffer<PatchId> d_selected(std::min(M_global_, N));

    // Payload: [0, bound) newly covered IDs, [bound] the winner's global ID.
    const std::uint64_t max_bound = std::max<std::uint64_t>(max_initial_score_, 1);
    DeviceBuffer<std::uint64_t> d_send(max_bound + 1);
    DeviceBuffer<std::uint64_t> d_recv(size > 1 ? max_bound + 1 : 0);
    std::uint64_t* d_payload = size > 1 ? d_recv.data() : d_send.data();

    DeviceBuffer<std::uint64_t> d_key_local(1), d_key_global(size > 1 ? 1 : 0);
    const std::uint64_t* d_key = size > 1 ? d_key_global.data() : d_key_local.data();

    using ArgmaxPair = cub::KeyValuePair<int, WireScore>;
    DeviceBuffer<ArgmaxPair>   d_argmax_result(1);
    DeviceBuffer<std::uint8_t> d_argmax_temp;
    d_argmax_result.zero();
    if (M_loc > 0) {
        std::size_t temp_bytes = 0;
        cub::DeviceReduce::ArgMax(nullptr, temp_bytes, d_scores_.data(),
                                  d_argmax_result.data(), static_cast<int>(M_loc));
        d_argmax_temp.resize(temp_bytes);
    }

    // Batch sizing as in ByElement: at least ceil(remaining / bound) more
    // iterations must run, so that many never overshoot the end of the solve.
    constexpr std::uint64_t kMaxBatch = 128;
    std::uint64_t bound = max_bound;
    LoopState h_state{};
    while (true) {
        const std::uint64_t remaining = N - h_state.covered_count;
        const std::uint64_t batch = std::clamp<std::uint64_t>(
            remaining / bound + (remaining % bound != 0), 1, kMaxBatch);
        for (std::uint64_t b = 0; b < batch; ++b) {
            if (M_loc > 0) {
                std::size_t temp_bytes = d_argmax_temp.bytes();
                cub::DeviceReduce::ArgMax(d_argmax_temp.data(), temp_bytes,
                                          d_scores_.data(), d_argmax_result.data(),
                                          static_cast<int>(M_loc), stream);
            }
            patch_pack_candidate_kernel<<<1, 1, 0, stream>>>(
                d_argmax_result.data(), M_loc > 0 ? 1 : 0, rank, d_key_local.data());
            if (size > 1) {
                comm_.allreduce<stComm::Space::Device, std::uint64_t>(
                    d_key_local.data(), d_key_global.data(), 1, stComm::ReduceOp::Max);
                cuda_check(cudaMemsetAsync(d_send.data(), 0,
                                           (bound + 1) * sizeof(std::uint64_t), stream),
                           "payload clear");
            }
            patch_commit_winner_kernel<<<1, 1, 0, stream>>>(
                d_key, d_argmax_result.data(), rank, N,
                patch_set_->d_patch_offsets(), d_global_ids_.data(),
                d_scores_.data(), d_send.data(), bound,
                d_state.data(), d_winner.data());
            fixed_build_newly_kernel<<<kFixedGridBlocks, kFixedBlockThreads, 0, stream>>>(
                patch_set_->d_patch_data(), d_covered_.data(),
                d_send.data(), d_winner.data());
            if (size > 1) {
                comm_.allreduce<stComm::Space::Device, std::uint64_t>(
                    d_send.data(), d_recv.data(), bound + 1, stComm::ReduceOp::Sum);
            }
            patch_record_winner_kernel<<<1, 1, 0, stream>>>(
                d_key, d_payload, bound, d_selected.data(),
                d_state.data(), d_winner.data());
            fixed_set_bits_kernel<<<kFixedGridBlocks, kFixedBlockThreads, 0, stream>>>(
                d_covered_.data(), d_payload, d_winner.data());
            fixed_score_update_kernel
                <<<kFixedGridBlocks, kScoreBlockThreads, 0, stream>>>(
                d_payload, d_winner.data(),
                patch_set_->d_inv_keys(), patch_set_->inv_keys().size(),
                patch_set_->d_inv_offsets(), patch_set_->d_inv_data(),
                d_scores_.data());
        }
        cuda_launch_check("device-resident batch (patch)");
        // Drain through stComm (polled: also watches for async NCCL failures),
        // then read the state — see the ByElement loop.
        comm_.nccl().barrier();
        cuda_check(cudaMemcpy(&h_state, d_state.data(), sizeof(h_state),
                              cudaMemcpyDeviceToHost),
                   "device-resident batch (patch) state copy");
        if (h_state.done) break;
        bound = std::max<std::uint64_t>(h_state.last_score, 1);
    }

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

// Element-partitioned greedy loop (§7.3) — DEVICE-RESIDENT.
//
// Every rank holds the full, identical score of every patch, so each rank
// computes the same argmax on its own — no MAXLOC, no winner broadcast. What
// travels is only what a selection changes: the patch slots whose score drops
// (see "ByElement delta exchange" above). Per step, all enqueued on the NCCL
// stream with no host knowledge of the outcome:
//
//   [device] plan + fill this rank's window of pending decrements
//        ncclAllReduce<SUM>(P segments of window + 2 words)   ← only collective
//   [device] apply every rank's window to the scores
//   [device] CUB ArgMax over the scores
//   [device] commit: nothing while any rank has more to send; otherwise
//            append slot, covered_count += score, winner range, disable
//   [device] build newly covered (this shard) + set_bits
//   [device] new delta stream: list lengths → prefix sum → cursor
//
// A step whose windows carried everything selects in the same step, so the
// common case is one small collective per selection. The host enqueues a
// batch of steps and syncs once to read the 24 B loop state; steps enqueued
// past termination no-op, and every branch above derives from reduced
// values, so all ranks stay in lockstep.
//
// The window is re-sized per batch from values every rank agrees on (the
// winner-score bound and the average element degree): about four times the
// expected decrements per rank and selection.
//
// size == 1 has no one to exchange with: the decrements go straight into the
// scores (strategy-A update), no collective at all.
//
// -DUSC_PROFILE stage timers are deliberately absent here: with no
// per-step syncs there are no host-visible stage boundaries to time.
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
    DeviceBuffer<PatchId>   d_selected(std::min(M_global_, N_));
    const std::uint64_t max_newly = std::max<std::uint64_t>(max_sub_len_, 1);
    DeviceBuffer<ElementId> d_newly_ids(max_newly);

    using ArgmaxPair = cub::KeyValuePair<int, WireScore>;
    DeviceBuffer<ArgmaxPair>   d_argmax_result(1);
    DeviceBuffer<std::uint8_t> d_argmax_temp;
    {
        std::size_t temp_bytes = 0;
        cub::DeviceReduce::ArgMax(nullptr, temp_bytes, d_scores_.data(),
                                  d_argmax_result.data(),
                                  static_cast<int>(M_global_));
        d_argmax_temp.resize(temp_bytes);
    }

    std::uint64_t score_bound = std::max<std::uint64_t>(max_initial_score_, 1);

    // Delta exchange state. The window only shrinks (score_bound does), so the
    // first one sizes the buffers.
    const auto window_for = [&](std::uint64_t bound) -> std::uint64_t {
        if (hooks.delta_window) return hooks.delta_window;
        const std::uint64_t expected = bound * avg_degree_ / usize;
        return std::clamp<std::uint64_t>(4 * expected + 16, 64, std::uint64_t{1} << 16);
    };
    DeviceBuffer<DeltaCursor>   d_cursor;
    DeviceBuffer<std::uint64_t> d_deg, d_prefix;
    DeviceBuffer<DeltaWord>     d_send, d_recv_buf;
    DeviceBuffer<std::uint8_t>  d_scan_temp;
    std::uint64_t window = 0;
    if (exchange) {
        d_cursor.resize(1);
        d_cursor.zero();
        d_deg.resize(max_newly + 1);
        d_prefix.resize(max_newly + 1);
        d_deg.zero();
        d_prefix.zero();
        const std::uint64_t words = usize * (window_for(score_bound) + 2);
        d_send.resize(words);
        if (size > 1) d_recv_buf.resize(words);
        std::size_t temp_bytes = 0;
        cuda_check(cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_deg.data(),
                                                 d_prefix.data(), max_newly + 1),
                   "delta prefix-sum sizing");
        d_scan_temp.resize(temp_bytes);
    }
    // One rank (test hook): the "reduction" of a single segment is itself.
    const DeltaWord* d_recv = size > 1 ? d_recv_buf.data() : d_send.data();

    // Batch sizing. Winner scores never increase over a greedy solve (covering
    // elements only lowers every patch's score), so each remaining selection
    // covers at most `score_bound` elements and at least
    // ceil(remaining / score_bound) more steps must run. Enqueueing exactly
    // that many — capped at kMaxBatch — never overshoots the end of the solve.
    // Both inputs come from reduced state, so every rank picks the same batch.
    constexpr std::uint64_t kMaxBatch = 128;
    LoopState h_state{};
    while (true) {
        const std::uint64_t remaining = N_ - h_state.covered_count;
        const std::uint64_t batch = std::clamp<std::uint64_t>(
            remaining / score_bound + (remaining % score_bound != 0), 1, kMaxBatch);
        // A newly-covered list never outgrows the winner score or the widest
        // sub-patch.
        const std::uint64_t len = std::min(score_bound, max_newly);
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
                elem_delta_fill_kernel<<<kFixedGridBlocks, kFixedBlockThreads, 0, stream>>>(
                    d_newly_ids.data(), d_cursor.data(), d_prefix.data(),
                    d_inv_keys_.data(), num_inv_keys_, d_inv_offsets_.data(),
                    d_inv_data_.data(), d_segment + 2);
                if (size > 1) {
                    // Enqueue-only — no wait.
                    comm_.allreduce<stComm::Space::Device, DeltaWord>(
                        d_send.data(), d_recv_buf.data(), usize * (window + 2),
                        stComm::ReduceOp::Sum);
                }
                elem_delta_apply_kernel<<<apply_blocks, kFixedBlockThreads, 0, stream>>>(
                    d_recv, usize, window, d_scores_.data());
            }
            std::size_t temp_bytes = d_argmax_temp.bytes();
            cub::DeviceReduce::ArgMax(d_argmax_temp.data(), temp_bytes,
                                      d_scores_.data(), d_argmax_result.data(),
                                      static_cast<int>(M_global_), stream);
            elem_commit_winner_kernel<<<1, 1, 0, stream>>>(
                d_argmax_result.data(), d_sub_offsets_.data(), N_,
                d_scores_.data(), d_selected.data(),
                d_recv, exchange ? size : 0, window,
                d_state.data(), d_winner.data());
            fixed_build_newly_kernel<<<kFixedGridBlocks, kFixedBlockThreads, 0, stream>>>(
                d_patch_data_.data(), d_covered_.data(),
                d_newly_ids.data(), d_winner.data());
            fixed_set_bits_kernel<<<kFixedGridBlocks, kFixedBlockThreads, 0, stream>>>(
                d_covered_.data(), d_newly_ids.data(), d_winner.data());
            if (exchange) {
                elem_delta_degrees_kernel<<<kFixedGridBlocks, kFixedBlockThreads, 0, stream>>>(
                    d_newly_ids.data(), d_winner.data(),
                    d_inv_keys_.data(), num_inv_keys_, d_inv_offsets_.data(),
                    len, d_deg.data());
                std::size_t scan_bytes = d_scan_temp.bytes();
                cub::DeviceScan::ExclusiveSum(d_scan_temp.data(), scan_bytes, d_deg.data(),
                                              d_prefix.data(), len + 1, stream);
                elem_delta_begin_kernel<<<1, 1, 0, stream>>>(
                    d_winner.data(), d_prefix.data(), len, d_cursor.data());
            } else {
                fixed_score_update_kernel<<<kFixedGridBlocks, kScoreBlockThreads, 0, stream>>>(
                    d_newly_ids.data(), d_winner.data(),
                    d_inv_keys_.data(), num_inv_keys_,
                    d_inv_offsets_.data(), d_inv_data_.data(),
                    d_scores_.data());
            }
        }
        cuda_launch_check("device-resident batch (elem)");
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
        if (h_state.done) break;
        if (h_state.num_selected) {
            score_bound = std::max<std::uint64_t>(h_state.last_score, 1);
        }
    }

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
