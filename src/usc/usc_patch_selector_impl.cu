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
#include <memory>
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
// Public facade: stPS::UscPatchSelector — pImpl over UscPatchSelectorImpl
// (hides all internals).
// ============================================================================

struct UscPatchSelector::Impl {
    UscPatchSelectorImpl usc;
    explicit Impl(stComm::Comm& comm) : usc(comm) {}
};

UscPatchSelector::UscPatchSelector(stComm::Comm& comm) : impl_(std::make_unique<Impl>(comm)) {}
UscPatchSelector::~UscPatchSelector() = default;
UscPatchSelector::UscPatchSelector(UscPatchSelector&&) noexcept = default;
UscPatchSelector& UscPatchSelector::operator=(UscPatchSelector&&) noexcept = default;

PatchSelection UscPatchSelector::patch_select(std::vector<std::vector<Hash>> patches,
                                              std::vector<PatchId>           global_ids) {
    return impl_->usc.patch_select(std::move(patches), std::move(global_ids));
}

}  // namespace stPS
