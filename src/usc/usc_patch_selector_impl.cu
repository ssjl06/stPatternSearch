#include "usc/usc_patch_selector_impl.hpp"

#include "core/gpu_profiler.hpp"
#include "core/host_profiler.hpp"

#include <stComm/stComm.h>

#include <cub/device/device_reduce.cuh>
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

// Set the bits at the given element IDs in a dense uint64 bitset.
// One thread per ID; atomicOr keeps concurrent writers to the same word safe.
__global__ void set_bits_kernel(std::uint64_t* bitset,
                                const ElementId* ids,
                                std::size_t n) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const ElementId id = ids[i];
    const std::uint64_t word_idx = id >> 6;
    const std::uint64_t bit_mask = 1ULL << (id & 63);
    atomicOr(reinterpret_cast<unsigned long long*>(&bitset[word_idx]),
             static_cast<unsigned long long>(bit_mask));
}

// Winner-only: scan the chosen patch's element IDs against d_covered_ and emit
// the IDs that are NOT yet covered into out_ids. atomicAdd on out_count
// allocates output slots; ordering of out_ids is therefore non-deterministic,
// but the resulting *set* matches the host build. Downstream consumers
// (bcast + set_bits_kernel into d_covered_, strategy-A score update) are
// order-insensitive (atomicOr on a bitset / commutative atomic decrements).
__global__ void build_newly_covered_kernel(const ElementId*     patch_data,
                                           std::uint64_t        start,
                                           std::uint64_t        end,
                                           const std::uint64_t* covered,
                                           ElementId*           out_ids,
                                           unsigned int*        out_count) {
    const std::uint64_t i =
        start + blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
    if (i >= end) return;
    const ElementId id = patch_data[i];
    const std::uint64_t bit = (covered[id >> 6] >> (id & 63)) & 1ULL;
    if (bit == 0) {
        const unsigned int slot = atomicAdd(out_count, 1u);
        out_ids[slot] = id;
    }
}

// Design doc §6.3 strategy A — inverted-index, affected-only score update.
//
// One block per *newly covered* element (grid = winner_score). The block looks
// up that element in this rank's inverted index (binary search over the sorted
// `inv_keys`) and decrements by one the score of every patch that contains it.
//
// Correctness vs the strategy-B full sweep: a patch's score equals its count of
// still-uncovered elements, so subtracting 1 for each newly covered element it
// contains is exactly popcount(patch ∩ newly_covered). Because each element
// transitions to "covered" exactly once over the whole solve, every
// (element, patch) incidence is visited at most once across all iterations —
// total work is O(local inverted-index size), not O(iters × M·K) like the full
// sweep. A patch whose score has reached 0 has all its elements covered, so none
// of them can reappear in a future newly_covered list and its score is never
// touched again — matching strategy B's `score <= 0` early-out bit-for-bit.
// Elements not present on this rank fall out of the binary search and are
// skipped (the index stores only locally-occurring elements).
__global__ void score_update_invidx_kernel(const ElementId*     newly_ids,
                                           std::uint64_t        n,
                                           const ElementId*     inv_keys,
                                           std::uint64_t        inv_keys_n,
                                           const std::uint64_t* inv_offsets,
                                           const PatchId*       inv_data,
                                           Score*               scores) {
    const std::uint64_t e = blockIdx.x;
    if (e >= n) return;

    // Resolve the element's patch-list range once (thread 0), share to the block.
    __shared__ std::uint64_t s_start;
    __shared__ std::uint64_t s_end;
    if (threadIdx.x == 0) {
        const ElementId id = newly_ids[e];
        std::uint64_t lo = 0, hi = inv_keys_n;
        while (lo < hi) {                       // lower_bound over sorted keys
            const std::uint64_t mid = lo + ((hi - lo) >> 1);
            if (inv_keys[mid] < id) lo = mid + 1;
            else                    hi = mid;
        }
        if (lo < inv_keys_n && inv_keys[lo] == id) {
            s_start = inv_offsets[lo];
            s_end   = inv_offsets[lo + 1];
        } else {
            s_start = 0;        // element not on this rank → empty range
            s_end   = 0;
        }
    }
    __syncthreads();

    // Score is int64; CUDA has no 64-bit atomicSub, so add the two's-complement
    // of 1 via the unsigned 64-bit atomicAdd. Subtractions commute, so the final
    // per-patch score is order-independent (deterministic).
    for (std::uint64_t i = s_start + threadIdx.x; i < s_end; i += blockDim.x) {
        const PatchId q = inv_data[i];
        atomicAdd(reinterpret_cast<unsigned long long*>(&scores[q]),
                  static_cast<unsigned long long>(-1LL));
    }
}

// ---------------------------------------------------------------------------
// Device-resident ByElement loop kernels (Full-GPU iteration track).
//
// The host enqueues whole batches of iterations blind, so every launch here
// uses a FIXED grid and reads the winner's range / newly-covered count from
// device memory — the host knows neither when it enqueues. All control flow
// that the ByPatch loop keeps on the host (winner resolution, covered_count,
// termination, winner disable) lives in the single-thread commit kernel.
// ---------------------------------------------------------------------------

constexpr int kElemGridBlocks   = 256;  // fixed grid for the grid-stride kernels
constexpr int kElemBlockThreads = 128;

// Device-resident loop state — the commit kernel is its only writer; the host
// reads it once per batch (24 B D2H) to decide whether to enqueue more.
struct ElemLoopState {
    std::uint64_t covered_count;
    std::uint64_t num_selected;   // == iterations executed
    std::uint32_t done;
    std::uint32_t pad_;
};

// Per-iteration winner scratch: written by the commit kernel, consumed by the
// three update kernels of the same (stream-ordered) iteration.
struct ElemWinnerInfo {
    std::uint64_t start;          // winner's shard-local sub-CSR range
    std::uint64_t end;
    unsigned int  newly_count;    // atomic output of the build kernel
    unsigned int  pad_;
};

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
                                          WireScore*           scores_partial,
                                          PatchId*             selected_slots,
                                          ElemLoopState*       st,
                                          ElemWinnerInfo*      w) {
    w->newly_count = 0;
    w->start = 0;
    w->end   = 0;
    if (st->done) return;
    const WireScore score = argmax->value;
    if (score <= 0) { st->done = 1; return; }
    const int slot = argmax->key;
    selected_slots[st->num_selected] = static_cast<PatchId>(slot);
    st->num_selected  += 1;
    st->covered_count += static_cast<std::uint64_t>(score);
    if (st->covered_count >= N) st->done = 1;
    w->start = sub_offsets[slot];
    w->end   = sub_offsets[slot + 1];
    // Winner disable, fused. The strategy-A update of this same iteration will
    // further decrement it by this shard's newly-covered count — it stays
    // permanently negative either way, so the selection sequence is identical
    // to the host loop's update-then-disable order.
    scores_partial[slot] = kDisabledWireScore;
}

// build_newly_covered, fixed-grid grid-stride: the range comes from w; the
// compact output count is w->newly_count (same non-deterministic order /
// deterministic set as build_newly_covered_kernel).
__global__ void elem_build_newly_kernel(const ElementId*     patch_data,
                                        const std::uint64_t* covered,
                                        ElementId*           out_ids,
                                        ElemWinnerInfo*      w) {
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
__global__ void elem_set_bits_kernel(std::uint64_t*        bitset,
                                     const ElementId*      ids,
                                     const ElemWinnerInfo* w) {
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
// elements instead of grid == count. Per-element logic matches
// score_update_invidx_kernel (shared-memory range resolve + per-patch −1).
__global__ void elem_score_update_kernel(const ElementId*      newly_ids,
                                         const ElemWinnerInfo* w,
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
            const ElementId id = newly_ids[e];
            std::uint64_t lo = 0, hi = inv_keys_n;
            while (lo < hi) {                       // lower_bound over sorted keys
                const std::uint64_t mid = lo + ((hi - lo) >> 1);
                if (inv_keys[mid] < id) lo = mid + 1;
                else                    hi = mid;
            }
            if (lo < inv_keys_n && inv_keys[lo] == id) {
                s_start = inv_offsets[lo];
                s_end   = inv_offsets[lo + 1];
            } else {
                s_start = 0;        // element not on this shard → empty range
                s_end   = 0;
            }
        }
        __syncthreads();
        for (std::uint64_t i = s_start + threadIdx.x; i < s_end; i += blockDim.x) {
            // WireScore is int32 — native atomicSub, no two's-complement dance.
            atomicSub(reinterpret_cast<int*>(&scores[inv_data[i]]), 1);
        }
        __syncthreads();  // all readers done before thread 0 overwrites s_start/s_end
    }
}

// Small helper: launch + sync-check. Aborts on any kernel error so a bug
// surfaces at the responsible kernel rather than later as an opaque memcpy fail.
inline void cuda_launch_check(const char* what) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA launch error in ") + what +
                                 ": " + cudaGetErrorString(err));
    }
}

inline void cuda_sync_check(const char* what) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA sync error in ") + what +
                                 ": " + cudaGetErrorString(err));
    }
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

    // Shared preprocessing: distributed hash→id mapping + PatchCsr + InvertedIndex
    // + device mirrors. USC owns only the score/covered state initialized below.
    patch_set_ = std::make_unique<PatchSet>(comm_, std::move(raw_patches));

    const std::uint64_t M_local = patch_set_->M_local();
    const std::uint64_t N       = patch_set_->N();

    // USC-specific device state: initial scores = patch sizes; covered empty.
    d_scores_.resize(M_local);
    d_covered_.resize((N + 63) / 64);
    if (M_local) {
        const PatchCsr& patches = patch_set_->patches();
        std::vector<Score> initial_scores(M_local);
        for (std::size_t p = 0; p < M_local; ++p) {
            initial_scores[p] = static_cast<Score>(patches.patch_size(p));
        }
        d_scores_.copy_from_host(initial_scores.data(), M_local);
    }
    d_covered_.zero();

    return select();
}

// Multi-rank greedy loop. Hot data (PatchCsr, scores, covered, newly_covered)
// lives on the device after patch_select() builds the PatchSet and stays there.
// Each iteration touches host memory only for the small metadata that NCCL's
// API requires as host scalars:
//
//   [device] CUB ArgMax → (Score, idx) D2H (16B)
//        [host] MPI_Allreduce<MAXLOC>  — NCCL has no MAXLOC
//        [host] MPI_Bcast winner_global PatchId (8B)
//   [device] build_newly_covered_kernel writes sparse list  (winner only)
//        NCCL_Bcast<ElementId>(d_newly_covered_ids, winner_score, winner_rank)
//          — device-direct, no D2H/H2D for the payload
//   [device] set_bits (d_covered_) + score_update_invidx_kernel (strategy A)
//        H2D 1 Score = -1 (winner only)
PatchSelection UscPatchSelectorImpl::select() {
    // Views into the shared PatchSet (built in patch_select). USC keeps only
    // d_scores_ / d_covered_ of its own; everything else is read from here.
    const PatchCsr&      patches       = patch_set_->patches();
    const InvertedIndex& inv           = patch_set_->inverted_index();
    const std::uint64_t  N             = patch_set_->N();
    const ElementId*     d_patch_data  = patch_set_->d_patch_data();
    const ElementId*     d_inv_keys    = patch_set_->d_inv_keys();
    const std::uint64_t* d_inv_offsets = patch_set_->d_inv_offsets();
    const PatchId*       d_inv_data    = patch_set_->d_inv_data();

    const std::uint64_t M_loc = patches.M();
    PatchSelection result;
    result.covered_count = 0;
    result.iterations    = 0;

    // Sparse newly_covered list — entirely device-resident; NCCL broadcasts
    // it in place. The cumulative covered state also lives on device
    // (d_covered_); no host bitset, no host staging buffer.
    DeviceBuffer<ElementId>    d_newly_covered_ids;
    DeviceBuffer<unsigned int> d_newly_count(1);   // atomicAdd counter for build kernel

    // CUB DeviceReduce::ArgMax scratch + result. Query the byte requirements
    // once for the known M_loc, then reuse the buffers across all iterations.
    using ArgmaxPair = cub::KeyValuePair<int, Score>;
    DeviceBuffer<std::uint8_t> d_argmax_temp;
    DeviceBuffer<ArgmaxPair>   d_argmax_result(M_loc > 0 ? 1 : 0);
    if (M_loc > 0) {
        std::size_t temp_bytes = 0;
        cub::DeviceReduce::ArgMax(nullptr, temp_bytes,
                                  d_scores_.data(),
                                  d_argmax_result.data(),
                                  static_cast<int>(M_loc));
        d_argmax_temp.resize(temp_bytes);
    }

    // Per-iteration stage timing, split by where the cost lands. Pure
    // default-stream GPU compute is measured with CUDA events (GpuProfiler);
    // host-blocking collectives — the MPI host-space ops and the NCCL bcast,
    // whose device work runs on a stComm-internal stream that default-stream
    // events can't see — are measured with host wall-clock around wait()
    // (HostProfiler). Compile-time gated by -DUSC_PROFILE: when off, every
    // begin/end is a no-op, so the production hot path keeps no synchronization
    // and no environment lookup.
#ifdef USC_PROFILE
    constexpr bool kProfile = true;
#else
    constexpr bool kProfile = false;
#endif
    enum GpuStage  { kArgmax = 0, kBuildKernel, kSetBits, kScoreUpdate, kWinnerDisable };
    enum HostStage { kMaxloc = 0, kPatchIdBcast, kNcclBcast };
    detail::GpuProfiler gpu_prof(
        {"argmax+D2H", "build_kernel", "set_bits", "score_update", "winner_disable"},
        kProfile);
    detail::HostProfiler host_prof(
        {"maxloc", "patchid_bcast", "nccl_bcast"}, kProfile);

    while (result.covered_count < N) {
        gpu_prof.begin(kArgmax);
        Score   local_best_score = kDisabledScore;
        PatchId local_best_patch = 0;
        if (M_loc > 0) {
            std::size_t temp_bytes = d_argmax_temp.bytes();
            cub::DeviceReduce::ArgMax(d_argmax_temp.data(), temp_bytes,
                                      d_scores_.data(),
                                      d_argmax_result.data(),
                                      static_cast<int>(M_loc));
            cuda_launch_check("cub::DeviceReduce::ArgMax");
            ArgmaxPair host_result{};
            cudaMemcpy(&host_result, d_argmax_result.data(),
                       sizeof(host_result), cudaMemcpyDeviceToHost);
            local_best_score = host_result.value;
            local_best_patch = static_cast<PatchId>(host_result.key);
        }
        gpu_prof.end(kArgmax);

        host_prof.begin(kMaxloc);
        // allreduceMaxloc is async on the unified Comm; wait() for the pair.
        std::pair<std::int64_t, int> maxloc;
        comm_.allreduceMaxloc<stComm::Space::Host>(local_best_score, &maxloc)->wait();
        const std::int64_t winner_score = maxloc.first;
        const int          winner_rank  = maxloc.second;
        host_prof.end(kMaxloc);
        if (winner_score <= 0) break;

        host_prof.begin(kPatchIdBcast);
        PatchId winner_global = 0;
        if (comm_.getRank() == winner_rank) {
            winner_global = patch_global_ids_[local_best_patch];
        }
        comm_.bcast<stComm::Space::Host, PatchId>(&winner_global, 1, winner_rank)->wait();
        host_prof.end(kPatchIdBcast);

        // ------------------------------------------------------------------
        // Build the sparse newly_covered list on device (winner only), then
        // broadcast it device-to-device via NCCL. d_newly_covered_ids never
        // visits host memory — the winner kernel writes it, NCCL bcast
        // distributes it, and set_bits_kernel consumes it.
        // ------------------------------------------------------------------
        gpu_prof.begin(kBuildKernel);
        if (d_newly_covered_ids.size() < static_cast<std::size_t>(winner_score)) {
            d_newly_covered_ids.resize(static_cast<std::size_t>(winner_score));
        }
        if (comm_.getRank() == winner_rank) {
            const std::uint64_t start = patches.offsets[local_best_patch];
            const std::uint64_t end   = patches.offsets[local_best_patch + 1];
            const std::uint64_t patch_len = end - start;
            d_newly_count.zero();
            if (patch_len > 0) {
                const int block = 128;
                const int grid  = static_cast<int>((patch_len + block - 1) / block);
                build_newly_covered_kernel<<<grid, block>>>(
                    d_patch_data, start, end,
                    d_covered_.data(),
                    d_newly_covered_ids.data(),
                    d_newly_count.data());
                cuda_launch_check("build_newly_covered_kernel");
            }
        }
        gpu_prof.end(kBuildKernel);

        // Device-direct broadcast. winner_score and winner_rank (16B) are the
        // only host-visible metadata per iteration — that's a NCCL API
        // constraint (root/count must be host scalars), not a payload cost.
        host_prof.begin(kNcclBcast);
        comm_.bcast<stComm::Space::Device, ElementId>(d_newly_covered_ids.data(),
                                              static_cast<std::size_t>(winner_score),
                                              winner_rank)->wait();
        host_prof.end(kNcclBcast);

        result.covered_count += static_cast<std::uint64_t>(winner_score);
        result.selected.push_back(winner_global);

        // ------------------------------------------------------------------
        // All ranks fold the sparse newly_covered list into the cumulative
        // d_covered_ bitset (consumed by next iteration's build kernel and the
        // termination check). Strategy A reads the sparse list + inverted index
        // directly, so the per-iteration newly_covered *bitset* is no longer
        // needed — one set_bits launch instead of two.
        // ------------------------------------------------------------------
        gpu_prof.begin(kSetBits);
        if (winner_score > 0) {
            const int block = 128;
            const int grid  = static_cast<int>(
                (static_cast<std::size_t>(winner_score) + block - 1) / block);
            set_bits_kernel<<<grid, block>>>(d_covered_.data(),
                                              d_newly_covered_ids.data(),
                                              static_cast<std::size_t>(winner_score));
            cuda_launch_check("set_bits_kernel (covered)");
        }
        gpu_prof.end(kSetBits);

        // Score update — design doc §6.3 strategy A (inverted-index, affected
        // only). One block per newly covered element; each decrements the score
        // of every local patch containing it. Bit-identical to the strategy-B
        // full sweep (see kernel comment), but visits each incidence once over
        // the whole solve instead of re-sweeping all patches every iteration.
        gpu_prof.begin(kScoreUpdate);
        if (winner_score > 0) {
            score_update_invidx_kernel<<<static_cast<unsigned>(winner_score),
                                         kScoreBlockThreads>>>(
                d_newly_covered_ids.data(),
                static_cast<std::uint64_t>(winner_score),
                d_inv_keys,
                inv.num_keys(),
                d_inv_offsets,
                d_inv_data,
                d_scores_.data());
            cuda_launch_check("score_update_invidx_kernel");
        }
        gpu_prof.end(kScoreUpdate);

        gpu_prof.begin(kWinnerDisable);
        if (comm_.getRank() == winner_rank) {
            // Disable winner on device so next iteration's kernel + ArgMax skip
            // it (kernel early-outs on score <= 0; ArgMax picks the next-best).
            const Score neg = kDisabledScore;
            cudaMemcpy(&d_scores_.data()[local_best_patch], &neg, sizeof(Score),
                       cudaMemcpyHostToDevice);
        }
        gpu_prof.end(kWinnerDisable);

        ++result.iterations;
    }

    if (comm_.getRank() == 0) {
        gpu_prof.report(stderr, "[usc-profile gpu]", result.iterations);
        host_prof.report(stderr, "[usc-profile host]", result.iterations);
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
const InvertedIndex& UscPatchSelectorImpl::inverted_index() const { return patch_set_->inverted_index(); }

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

    // Shared §5.1 preprocessing: hash → dense ElementId translation.
    IdMappedPatches mapped = map_hashes_to_element_ids(comm_, std::move(raw_patches));
    N_ = mapped.N;

    // ---- Global slot assignment (rank-major input order) --------------------
    // Slot of local patch p on rank r = (Σ M_local below r) + p. With
    // slice_patches_by_rank's contiguous IDs this equals ascending global-ID
    // order, so the smallest-slot tie-break in select() matches
    // brute_force_select's smallest-PatchId rule (and ByPatch's
    // smallest-rank-then-smallest-local-index MAXLOC rule).
    const std::uint64_t M_local = mapped.id_patches.size();
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
    std::vector<int> mcounts(static_cast<std::size_t>(size), 0);
    {
        std::vector<int> ones(static_cast<std::size_t>(size), 1);
        const int m_local_int = static_cast<int>(M_local);
        comm_.allgatherv<stComm::Space::Host, int>(&m_local_int, 1,
                                                   mcounts.data(), ones.data())->wait();
    }
    slot_to_gid_.resize(M_global_);
    comm_.allgatherv<stComm::Space::Host, PatchId>(global_ids.data(),
                                                   static_cast<int>(M_local),
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

    // ---- Incidence redistribution --------------------------------------------
    // Every (slot, element) incidence flies to the element's owner shard. Each
    // rank then holds, for ALL M_global patches, the sub-list inside its shard —
    // every incidence stored on exactly one rank (M·K/P per rank), tighter than
    // §7.5's conservative full patch replication.
    std::vector<int> sendcounts(static_cast<std::size_t>(size), 0);
    for (const auto& p : mapped.id_patches)
        for (ElementId e : p)
            ++sendcounts[shard_size ? static_cast<int>(e / shard_size) : 0];

    std::vector<int> send_offsets(static_cast<std::size_t>(size), 0);
    for (int r = 1; r < size; ++r) send_offsets[r] = send_offsets[r - 1] + sendcounts[r - 1];

    const std::size_t total_send =
        std::accumulate(sendcounts.begin(), sendcounts.end(), std::size_t{0});
    std::vector<PatchId>   send_slots(total_send);
    std::vector<ElementId> send_elems(total_send);
    {
        std::vector<int> cursor = send_offsets;  // per-target write head
        for (std::uint64_t p = 0; p < M_local; ++p) {
            const PatchId slot = slot_start + p;
            for (ElementId e : mapped.id_patches[p]) {
                const int target = shard_size ? static_cast<int>(e / shard_size) : 0;
                send_slots[cursor[target]] = slot;
                send_elems[cursor[target]] = e;
                ++cursor[target];
            }
        }
    }
    mapped.id_patches.clear();
    mapped.id_patches.shrink_to_fit();

    // Exchange counts (same allgatherv trick as §5.1 step 3), then ship the
    // incidences as two parallel alltoallv's sharing one count layout.
    std::vector<int> all_sendcounts(static_cast<std::size_t>(size) * size);
    {
        std::vector<int> uniform(static_cast<std::size_t>(size), size);
        comm_.allgatherv<stComm::Space::Host, int>(sendcounts.data(), size,
                                                   all_sendcounts.data(), uniform.data())->wait();
    }
    std::vector<int> recvcounts(static_cast<std::size_t>(size), 0);
    for (int s = 0; s < size; ++s) recvcounts[s] = all_sendcounts[s * size + rank];
    const std::size_t total_recv =
        std::accumulate(recvcounts.begin(), recvcounts.end(), std::size_t{0});

    std::vector<PatchId>   recv_slots(total_recv);
    std::vector<ElementId> recv_elems(total_recv);
    comm_.alltoallv<stComm::Space::Host, PatchId>(send_slots.data(), sendcounts.data(),
                                                  recv_slots.data(), recvcounts.data())->wait();
    comm_.alltoallv<stComm::Space::Host, ElementId>(send_elems.data(), sendcounts.data(),
                                                    recv_elems.data(), recvcounts.data())->wait();
    send_slots.clear(); send_slots.shrink_to_fit();
    send_elems.clear(); send_elems.shrink_to_fit();

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

    // Shard-local inverted index (keys are shard-local IDs, values are global
    // slots) — drives the strategy-A partial-score update.
    inv_ = build_inverted_index(sub, shard_extent_ ? shard_extent_ : 1);

    // ---- Device mirrors + algorithm state -------------------------------------
    if (!sub.data.empty()) {
        d_patch_data_.resize(sub.data.size());
        d_patch_data_.copy_from_host(sub.data.data(), sub.data.size());
    }
    if (!inv_.keys.empty()) {
        d_inv_keys_.resize(inv_.keys.size());
        d_inv_keys_.copy_from_host(inv_.keys.data(), inv_.keys.size());
    }
    if (!inv_.offsets.empty()) {
        d_inv_offsets_.resize(inv_.offsets.size());
        d_inv_offsets_.copy_from_host(inv_.offsets.data(), inv_.offsets.size());
    }
    if (!inv_.data.empty()) {
        d_inv_data_.resize(inv_.data.size());
        d_inv_data_.copy_from_host(inv_.data.data(), inv_.data.size());
    }

    d_scores_partial_.resize(M_global_);
    d_scores_sum_.resize(M_global_);
    max_sub_len_ = 0;
    if (M_global_) {
        // Initial partial score = this shard's slice of each patch (full score
        // arrives via the per-iteration AllReduce<SUM>). Also record the widest
        // sub-patch — it bounds the newly-covered scratch for the whole solve.
        std::vector<WireScore> init(M_global_);
        for (std::uint64_t s = 0; s < M_global_; ++s) {
            const std::uint64_t len = sub.offsets[s + 1] - sub.offsets[s];
            init[s] = static_cast<WireScore>(len);
            max_sub_len_ = std::max(max_sub_len_, len);
        }
        d_scores_partial_.copy_from_host(init.data(), M_global_);
        // Offsets mirror — the commit kernel resolves winner ranges on device,
        // so the host never touches the sub-CSR during the loop.
        d_sub_offsets_.resize(sub.offsets.size());
        d_sub_offsets_.copy_from_host(sub.offsets.data(), sub.offsets.size());

        // Prove the int32 wire fits, once. Initial global scores (= deduped
        // patch sizes) are the solve's maxima — scores only decrease — so if
        // every initial sum fits with a P-sized margin (the disabled floor is
        // −P − patch_size), every value the wire ever carries fits. A one-time
        // host allreduce of the int64 partials makes the check exact.
        std::vector<std::int64_t> init64(M_global_), sum64(M_global_);
        for (std::uint64_t s = 0; s < M_global_; ++s) init64[s] = init[s];
        comm_.allreduce<stComm::Space::Host, std::int64_t>(
            init64.data(), sum64.data(), M_global_, stComm::ReduceOp::Sum)->wait();
        const std::int64_t wire_max =
            static_cast<std::int64_t>(std::numeric_limits<WireScore>::max()) - size - 1;
        for (std::uint64_t s = 0; s < M_global_; ++s) {
            if (sum64[s] > wire_max) {
                throw std::invalid_argument(
                    "ByElement mode: patch " + std::to_string(slot_to_gid_[s]) +
                    " has " + std::to_string(sum64[s]) +
                    " unique elements — exceeds the int32 score wire format");
            }
        }
    }

    d_covered_.resize((shard_extent_ + 63) / 64);
    d_covered_.zero();

    return select();
}

// Element-partitioned greedy loop (§7.3) — DEVICE-RESIDENT (Full-GPU track).
//
// Per iteration, all enqueued on the NCCL stream with no host knowledge of
// the outcome:
//
//        ncclAllReduce<SUM>(d_scores_partial_ → d_scores_sum_, M)   ← only collective
//   [device] CUB ArgMax over the (identical) full scores
//   [device] elem_commit_winner: append slot, covered_count += score,
//            winner range resolve, winner disable, done flag   (1 thread)
//   [device] elem_build_newly / elem_set_bits / elem_score_update
//            (fixed grids; range/count read from device memory)
//
// The host enqueues kBatch iterations at a time and then syncs ONCE to read
// the 24 B loop state — there are no per-iteration D2H/H2D round-trips. This
// is only possible in ByElement: its collective has a fixed count and no
// root, so it needs no host scalars (the exact blockers that keep the
// ByPatch multi-rank loop host-driven; see ROADMAP "Full-GPU iteration
// track"). Iterations enqueued past termination no-op on device, and `done`
// derives only from allreduced values, so every rank leaves after the same
// batch — collectives stay matched.
//
// size == 1 additionally skips the allreduce (partials ARE the full scores):
// the pure kernel chain is the single-rank device-resident loop the ROADMAP
// prototype validated at 2.7×.
//
// -DUSC_PROFILE stage timers are deliberately absent here: with no
// per-iteration syncs there are no host-visible stage boundaries to time.
// Profile this loop with Nsight Systems instead.
PatchSelection UscElemPatchSelectorImpl::select() {
    PatchSelection result;
    result.covered_count = 0;
    result.iterations    = 0;
    if (M_global_ == 0 || N_ == 0) return result;   // agreed on every rank

    const int size = comm_.getSize();
    // Kernels ride the same stream as the collectives, so batch ordering is
    // carried by the stream alone — no events, no per-iteration syncs.
    cudaStream_t stream = comm_.nccl().getStream();

    DeviceBuffer<ElemLoopState>  d_state(1);
    DeviceBuffer<ElemWinnerInfo> d_winner(1);
    d_state.zero();
    d_winner.zero();
    DeviceBuffer<PatchId>   d_selected(M_global_);
    DeviceBuffer<ElementId> d_newly_ids(std::max<std::uint64_t>(max_sub_len_, 1));

    // size == 1: the partial scores ARE the full scores — no allreduce at all.
    const WireScore* d_full_scores = (size == 1) ? d_scores_partial_.data()
                                                 : d_scores_sum_.data();

    using ArgmaxPair = cub::KeyValuePair<int, WireScore>;
    DeviceBuffer<ArgmaxPair>   d_argmax_result(1);
    DeviceBuffer<std::uint8_t> d_argmax_temp;
    {
        std::size_t temp_bytes = 0;
        cub::DeviceReduce::ArgMax(nullptr, temp_bytes, d_full_scores,
                                  d_argmax_result.data(),
                                  static_cast<int>(M_global_));
        d_argmax_temp.resize(temp_bytes);
    }

    constexpr int kBatch = 128;
    ElemLoopState h_state{};
    while (true) {
        for (int b = 0; b < kBatch; ++b) {
            if (size > 1) {
                // Enqueue-only — no wait. Stream order makes iteration k's
                // updated partials the input of iteration k+1's allreduce.
                // int32 wire: half the payload of the naive int64 scores —
                // this allreduce is the mode's dominant cost (see STATUS).
                comm_.allreduce<stComm::Space::Device, WireScore>(
                    d_scores_partial_.data(), d_scores_sum_.data(), M_global_,
                    stComm::ReduceOp::Sum);
            }
            std::size_t temp_bytes = d_argmax_temp.bytes();
            cub::DeviceReduce::ArgMax(d_argmax_temp.data(), temp_bytes,
                                      d_full_scores,
                                      d_argmax_result.data(),
                                      static_cast<int>(M_global_), stream);
            elem_commit_winner_kernel<<<1, 1, 0, stream>>>(
                d_argmax_result.data(), d_sub_offsets_.data(), N_,
                d_scores_partial_.data(), d_selected.data(),
                d_state.data(), d_winner.data());
            elem_build_newly_kernel<<<kElemGridBlocks, kElemBlockThreads, 0, stream>>>(
                d_patch_data_.data(), d_covered_.data(),
                d_newly_ids.data(), d_winner.data());
            elem_set_bits_kernel<<<kElemGridBlocks, kElemBlockThreads, 0, stream>>>(
                d_covered_.data(), d_newly_ids.data(), d_winner.data());
            elem_score_update_kernel<<<kElemGridBlocks, kScoreBlockThreads, 0, stream>>>(
                d_newly_ids.data(), d_winner.data(),
                d_inv_keys_.data(), inv_.num_keys(),
                d_inv_offsets_.data(), d_inv_data_.data(),
                d_scores_partial_.data());
        }
        cuda_launch_check("device-resident batch (elem)");
        cudaMemcpyAsync(&h_state, d_state.data(), sizeof(h_state),
                        cudaMemcpyDeviceToHost, stream);
        const cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("CUDA error in device-resident batch (elem): ") +
                cudaGetErrorString(err));
        }
        if (h_state.done) break;
    }

    // Drain the device-side result: selected slots → caller PatchIds.
    result.covered_count = h_state.covered_count;
    result.iterations    = h_state.num_selected;
    if (h_state.num_selected) {
        std::vector<PatchId> slots(h_state.num_selected);
        cudaMemcpy(slots.data(), d_selected.data(),
                   h_state.num_selected * sizeof(PatchId), cudaMemcpyDeviceToHost);
        result.selected.reserve(slots.size());
        for (PatchId s : slots) result.selected.push_back(slot_to_gid_[s]);
    }
    return result;
}

// ============================================================================
// Public facade: stPS::UscPatchSelector — pImpl routing to the ByPatch or
// ByElement implementation (hides all internals).
// ============================================================================

struct UscPatchSelector::Impl {
    // Exactly one is engaged, per the ctor's PartitionMode.
    std::unique_ptr<UscPatchSelectorImpl>     by_patch;
    std::unique_ptr<UscElemPatchSelectorImpl> by_element;
    Impl(stComm::Comm& comm, PartitionMode mode) {
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
    return impl_->by_element
        ? impl_->by_element->patch_select(std::move(patches), std::move(global_ids))
        : impl_->by_patch->patch_select(std::move(patches), std::move(global_ids));
}

}  // namespace stPS
