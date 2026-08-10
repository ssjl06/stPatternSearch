#include <stPS/ups_pattern_stats.hpp>

#include "core/device_buffer.hpp"
#include "core/patch_set.hpp"

#include <stComm/stComm.h>

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace stPS {

namespace {

// Ordering for statistics output: most patches first, then smaller hash —
// deterministic across ranks so every rank derives the identical top-k list.
bool stat_before(const PatternStat& a, const PatternStat& b) {
    return a.count != b.count ? a.count > b.count : a.hash < b.hash;
}

// Point's host operator< isn't __device__; the reduction uses this functor.
// Same lexicographic rule: smaller x first, then smaller y.
struct LexMinPoint {
    __device__ Point operator()(const Point& a, const Point& b) const {
        if (a.x != b.x) return a.x < b.x ? a : b;
        return a.y <= b.y ? a : b;
    }
};

struct SumU64 {
    __device__ std::uint64_t operator()(std::uint64_t a, std::uint64_t b) const {
        return a + b;
    }
};

__global__ void degrees_kernel(const std::uint64_t* offsets, std::uint64_t n,
                               std::uint64_t* deg) {
    const std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
    if (i < n) deg[i] = offsets[i + 1] - offsets[i];
}

__global__ void iota_kernel(std::uint32_t* idx, std::uint64_t n) {
    const std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
    if (i < n) idx[i] = static_cast<std::uint32_t>(i);
}

__global__ void gather_u64_kernel(const std::uint64_t* src, const std::uint32_t* perm,
                                  std::uint64_t n, std::uint64_t* dst) {
    const std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
    if (i < n) dst[i] = src[perm[i]];
}

__global__ void gather_pt_kernel(const Point* src, const std::uint32_t* perm,
                                 std::uint64_t n, Point* dst) {
    const std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
    if (i < n) dst[i] = src[perm[i]];
}

// Scatter the reduced (id, count, minloc) runs into the dense shard arrays.
__global__ void scatter_shard_kernel(const std::uint64_t* ids,
                                     const std::uint64_t* cnt, const Point* pts,
                                     std::uint64_t n_runs, std::uint64_t shard_start,
                                     std::uint64_t* out_cnt, Point* out_pts) {
    const std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
    if (i < n_runs) {
        const std::uint64_t slot = ids[i] - shard_start;
        out_cnt[slot] = cnt[i];
        out_pts[slot] = pts[i];
    }
}

// Sort key for the shard top-k: ascending ~count == descending count. The
// radix sort is stable and the payload slots enter in ascending order (== hash
// order), so ties come out hash-ascending for free.
__global__ void invert_count_kernel(const std::uint64_t* cnt, std::uint64_t n,
                                    std::uint64_t* key) {
    const std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
    if (i < n) key[i] = ~cnt[i];
}

// Compact the top-k winners' counts/locations by their sorted slot order.
__global__ void take_top_kernel(const std::uint32_t* sorted_slots, std::uint64_t k,
                                const std::uint64_t* cnt, const Point* pts,
                                std::uint64_t* out_cnt, Point* out_pts) {
    const std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
    if (i < k) {
        const std::uint32_t slot = sorted_slots[i];
        out_cnt[i] = cnt[slot];
        out_pts[i] = pts[slot];
    }
}

inline unsigned blocks_for(std::uint64_t n, unsigned threads = 256) {
    return static_cast<unsigned>((n + threads - 1) / threads);
}

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

// Two-phase CUB call: query temp bytes, (re)alloc, run.
template <typename F>
void cub_run(DeviceBuffer<unsigned char>& temp, const char* what, F&& f) {
    std::size_t bytes = 0;
    f(nullptr, bytes);
    if (bytes > temp.bytes()) temp.resize(bytes);
    f(temp.data(), bytes);
    cuda_launch_check(what);
}

}  // namespace

struct UpsPatternStats::Impl {
    stComm::Comm& comm;

    explicit Impl(stComm::Comm& c) : comm(c) {
        if (!comm.hasDevice()) {
            throw std::invalid_argument(
                "UpsPatternStats requires a device-enabled Comm (Comm::onDevice) — "
                "UPS is GPU-only.");
        }
    }

    std::vector<PatternStat> run(std::vector<std::vector<Hash>>  patches,
                                 std::vector<std::vector<Point>> coords,
                                 std::uint64_t k);
};

std::vector<PatternStat> UpsPatternStats::Impl::run(
    std::vector<std::vector<Hash>>  patches,
    std::vector<std::vector<Point>> coords,
    std::uint64_t k) {
    const int size = comm.getSize();
    const int rank = comm.getRank();
    DeviceBuffer<unsigned char> d_temp;

    // ---- A. Local occurrence reduction (device): sort this rank's raw
    // (hash, point) occurrences by hash, then reduce-by-key with lexicographic
    // min — one representative candidate per local unique hash, hash-sorted.
    std::size_t total_occ = 0;
    for (const auto& p : patches) total_occ += p.size();

    std::vector<Hash>  h_occ_hash;  h_occ_hash.reserve(total_occ);
    std::vector<Point> h_occ_pts;   h_occ_pts.reserve(total_occ);
    for (std::size_t p = 0; p < patches.size(); ++p) {
        h_occ_hash.insert(h_occ_hash.end(), patches[p].begin(), patches[p].end());
        h_occ_pts.insert(h_occ_pts.end(), coords[p].begin(), coords[p].end());
    }
    coords.clear(); coords.shrink_to_fit();

    DeviceBuffer<Hash>  d_uniq_hash;   // ReduceByKey needs the buffer; only the
    DeviceBuffer<Point> d_min_pts;     // run count feeds the alignment check.
    std::uint64_t n_uniq = 0;
    if (total_occ > 0) {
        DeviceBuffer<Hash>  d_occ_hash(total_occ), d_sorted_hash(total_occ);
        DeviceBuffer<Point> d_occ_pts(total_occ),  d_sorted_pts(total_occ);
        d_occ_hash.copy_from_host(h_occ_hash.data(), total_occ);
        d_occ_pts.copy_from_host(h_occ_pts.data(), total_occ);
        h_occ_hash.clear(); h_occ_hash.shrink_to_fit();
        h_occ_pts.clear();  h_occ_pts.shrink_to_fit();

        cub_run(d_temp, "radix-sort occurrences", [&](void* t, std::size_t& b) {
            cub::DeviceRadixSort::SortPairs(t, b,
                d_occ_hash.data(), d_sorted_hash.data(),
                d_occ_pts.data(),  d_sorted_pts.data(),
                static_cast<int>(total_occ));
        });

        d_uniq_hash.resize(total_occ);
        d_min_pts.resize(total_occ);
        DeviceBuffer<int> d_num_runs(1);
        cub_run(d_temp, "reduce-by-key min-location", [&](void* t, std::size_t& b) {
            cub::DeviceReduce::ReduceByKey(t, b,
                d_sorted_hash.data(), d_uniq_hash.data(),
                d_sorted_pts.data(),  d_min_pts.data(),
                d_num_runs.data(), LexMinPoint{}, static_cast<int>(total_occ));
        });
        int n_runs = 0;
        d_num_runs.copy_to_host(&n_runs, 1);
        n_uniq = static_cast<std::uint64_t>(n_runs);
    }

    // ---- Shared distributed setup. The §5.1 mapping assigns IDs in global
    // hash order, so the hash-sorted reduction above pairs positionally with
    // the PatchSet's sorted inverted-index keys.
    PatchSet ps(comm, std::move(patches));
    const InvertedIndex& inv = ps.inverted_index();
    const std::uint64_t n_local = inv.keys.size();
    if (n_uniq != n_local) {
        throw std::runtime_error(
            "pattern_stats: occurrence reduction / inverted-index disagree (" +
            std::to_string(n_uniq) + " vs " + std::to_string(n_local) + ")");
    }

    // ---- B. Local degrees (device): adjacent difference over the PatchSet's
    // inverted-index offsets mirror.
    DeviceBuffer<std::uint64_t> d_deg(n_local);
    if (n_local > 0) {
        degrees_kernel<<<blocks_for(n_local), 256>>>(ps.d_inv_offsets(), n_local,
                                                     d_deg.data());
        cuda_launch_check("degrees_kernel");
    }

    // ---- C. Route (id, degree, minloc) to each element's owner rank —
    // device-direct NCCL alltoallv for the big arrays, host MPI for the tiny
    // count metadata (USC M5 precedent).
    std::vector<std::uint64_t> starts(static_cast<std::size_t>(size) + 1);
    {
        const std::uint64_t my_start = ps.shard_start();
        std::vector<int> ones(static_cast<std::size_t>(size), 1);
        comm.allgatherv<stComm::Space::Host, std::uint64_t>(&my_start, 1,
                                                starts.data(), ones.data())->wait();
        starts[static_cast<std::size_t>(size)] = ps.N();
    }

    // inv.keys is sorted → each owner's entries form one contiguous run; the
    // host copy of the keys yields sendcounts without a D2H transfer.
    std::vector<int> sendcounts(static_cast<std::size_t>(size), 0);
    for (std::size_t i = 0; i < n_local; ++i) {
        const int owner = static_cast<int>(
            std::upper_bound(starts.begin() + 1, starts.begin() + size, inv.keys[i]) -
            (starts.begin() + 1));
        ++sendcounts[owner];
    }
    std::vector<int> recvcounts(static_cast<std::size_t>(size), 0);
    {
        std::vector<int> all(static_cast<std::size_t>(size) * size);
        std::vector<int> uniform(static_cast<std::size_t>(size), size);
        comm.allgatherv<stComm::Space::Host, int>(sendcounts.data(), size,
                                       all.data(), uniform.data())->wait();
        for (int s = 0; s < size; ++s) recvcounts[s] = all[s * size + rank];
    }
    const std::size_t total_recv = static_cast<std::size_t>(
        std::accumulate(recvcounts.begin(), recvcounts.end(), 0));

    // Points travel as interleaved doubles (Point is two packed doubles) with
    // doubled counts — no split kernel, pairing preserved.
    static_assert(sizeof(Point) == 2 * sizeof(double),
                  "Point must be two packed doubles for the coordinate exchange");
    std::vector<int> sendcounts2(sendcounts), recvcounts2(recvcounts);
    for (int r = 0; r < size; ++r) { sendcounts2[r] *= 2; recvcounts2[r] *= 2; }

    DeviceBuffer<std::uint64_t> d_r_ids(total_recv), d_r_deg(total_recv);
    DeviceBuffer<Point>         d_r_pts(total_recv);
    cuda_sync_check("pre-alltoallv");  // kernel outputs visible before NCCL reads
    comm.alltoallv<stComm::Space::Device, std::uint64_t>(
        ps.d_inv_keys(), sendcounts.data(), d_r_ids.data(), recvcounts.data())->wait();
    comm.alltoallv<stComm::Space::Device, std::uint64_t>(
        d_deg.data(), sendcounts.data(), d_r_deg.data(), recvcounts.data())->wait();
    comm.alltoallv<stComm::Space::Device, double>(
        reinterpret_cast<const double*>(d_min_pts.data()), sendcounts2.data(),
        reinterpret_cast<double*>(d_r_pts.data()), recvcounts2.data())->wait();

    // ---- D. Owner-side reduction (device): sort arrivals by id, sum the
    // degrees, keep the lexicographic-min location, scatter into the dense
    // shard arrays. Integer sums + comparison mins → order-independent, so
    // results stay bit-identical across rank counts.
    const std::vector<Hash>& shard = ps.shard_hashes();
    const std::uint64_t shard_n     = shard.size();
    const std::uint64_t shard_start = ps.shard_start();

    DeviceBuffer<std::uint64_t> d_shard_cnt(shard_n);
    DeviceBuffer<Point>         d_shard_pts(shard_n);
    if (shard_n > 0) { d_shard_cnt.zero(); d_shard_pts.zero(); }

    if (total_recv > 0) {
        DeviceBuffer<std::uint32_t> d_idx(total_recv), d_perm(total_recv);
        iota_kernel<<<blocks_for(total_recv), 256>>>(d_idx.data(), total_recv);
        cuda_launch_check("iota_kernel");

        DeviceBuffer<std::uint64_t> d_s_ids(total_recv);
        cub_run(d_temp, "radix-sort arrivals", [&](void* t, std::size_t& b) {
            cub::DeviceRadixSort::SortPairs(t, b,
                d_r_ids.data(), d_s_ids.data(),
                d_idx.data(),   d_perm.data(),
                static_cast<int>(total_recv));
        });

        DeviceBuffer<std::uint64_t> d_g_deg(total_recv);
        DeviceBuffer<Point>         d_g_pts(total_recv);
        gather_u64_kernel<<<blocks_for(total_recv), 256>>>(
            d_r_deg.data(), d_perm.data(), total_recv, d_g_deg.data());
        gather_pt_kernel<<<blocks_for(total_recv), 256>>>(
            d_r_pts.data(), d_perm.data(), total_recv, d_g_pts.data());
        cuda_launch_check("gather kernels");

        DeviceBuffer<std::uint64_t> d_o_ids(total_recv), d_o_cnt(total_recv);
        DeviceBuffer<Point>         d_o_pts(total_recv);
        DeviceBuffer<std::uint64_t> d_o_ids2(total_recv);
        DeviceBuffer<int>           d_num_runs(1);
        cub_run(d_temp, "reduce-by-key count-sum", [&](void* t, std::size_t& b) {
            cub::DeviceReduce::ReduceByKey(t, b,
                d_s_ids.data(), d_o_ids.data(),
                d_g_deg.data(), d_o_cnt.data(),
                d_num_runs.data(), SumU64{}, static_cast<int>(total_recv));
        });
        cub_run(d_temp, "reduce-by-key location-min", [&](void* t, std::size_t& b) {
            cub::DeviceReduce::ReduceByKey(t, b,
                d_s_ids.data(), d_o_ids2.data(),
                d_g_pts.data(), d_o_pts.data(),
                d_num_runs.data(), LexMinPoint{}, static_cast<int>(total_recv));
        });
        int n_runs = 0;
        d_num_runs.copy_to_host(&n_runs, 1);

        scatter_shard_kernel<<<blocks_for(static_cast<std::uint64_t>(n_runs)), 256>>>(
            d_o_ids.data(), d_o_cnt.data(), d_o_pts.data(),
            static_cast<std::uint64_t>(n_runs), shard_start,
            d_shard_cnt.data(), d_shard_pts.data());
        cuda_launch_check("scatter_shard_kernel");
    }

    // ---- E. Local top-k of this shard (device): one stable radix sort on
    // ~count (slots enter hash-ascending, stability keeps ties hash-asc).
    const std::size_t k_local = std::min<std::size_t>(k, shard_n);
    std::vector<Hash>          c_hash(k_local);
    std::vector<std::uint64_t> c_cnt(k_local);
    std::vector<double>        c_x(k_local), c_y(k_local);
    if (k_local > 0) {
        DeviceBuffer<std::uint64_t> d_key(shard_n), d_key_out(shard_n);
        DeviceBuffer<std::uint32_t> d_slots(shard_n), d_sorted_slots(shard_n);
        invert_count_kernel<<<blocks_for(shard_n), 256>>>(d_shard_cnt.data(), shard_n,
                                                          d_key.data());
        iota_kernel<<<blocks_for(shard_n), 256>>>(d_slots.data(), shard_n);
        cuda_launch_check("top-k key kernels");

        cub_run(d_temp, "radix-sort top-k", [&](void* t, std::size_t& b) {
            cub::DeviceRadixSort::SortPairs(t, b,
                d_key.data(),   d_key_out.data(),
                d_slots.data(), d_sorted_slots.data(),
                static_cast<int>(shard_n));
        });

        DeviceBuffer<std::uint64_t> d_top_cnt(k_local);
        DeviceBuffer<Point>         d_top_pts(k_local);
        take_top_kernel<<<blocks_for(k_local), 256>>>(
            d_sorted_slots.data(), k_local,
            d_shard_cnt.data(), d_shard_pts.data(),
            d_top_cnt.data(), d_top_pts.data());
        cuda_launch_check("take_top_kernel");

        std::vector<std::uint32_t> top_slots(k_local);
        std::vector<Point>         top_pts(k_local);
        d_sorted_slots.copy_to_host(top_slots.data(), k_local);
        d_top_cnt.copy_to_host(c_cnt.data(), k_local);
        d_top_pts.copy_to_host(top_pts.data(), k_local);
        for (std::size_t i = 0; i < k_local; ++i) {
            c_hash[i] = shard[top_slots[i]];
            c_x[i]    = top_pts[i].x;
            c_y[i]    = top_pts[i].y;
        }
    }

    // ---- F. Global merge (host): gather every rank's candidates (variable
    // counts, tiny) and reduce identically everywhere. Any global top-k entry
    // is in its owner's local top-k, so this is exact. All ranks hold size×k
    // candidates — fine for a stats tool; revisit if k ever gets huge.
    const int n_cand = static_cast<int>(k_local);
    std::vector<int> cand_counts(static_cast<std::size_t>(size));
    {
        std::vector<int> ones(static_cast<std::size_t>(size), 1);
        comm.allgatherv<stComm::Space::Host, int>(&n_cand, 1,
                                       cand_counts.data(), ones.data())->wait();
    }
    const std::size_t total_cand = static_cast<std::size_t>(
        std::accumulate(cand_counts.begin(), cand_counts.end(), 0));

    std::vector<Hash>          all_hash(total_cand);
    std::vector<std::uint64_t> all_cnt(total_cand);
    std::vector<double>        all_x(total_cand), all_y(total_cand);
    comm.allgatherv<stComm::Space::Host, Hash>(c_hash.data(), n_cand,
                                   all_hash.data(), cand_counts.data())->wait();
    comm.allgatherv<stComm::Space::Host, std::uint64_t>(c_cnt.data(), n_cand,
                                   all_cnt.data(), cand_counts.data())->wait();
    comm.allgatherv<stComm::Space::Host, double>(c_x.data(), n_cand,
                                   all_x.data(), cand_counts.data())->wait();
    comm.allgatherv<stComm::Space::Host, double>(c_y.data(), n_cand,
                                   all_y.data(), cand_counts.data())->wait();

    std::vector<PatternStat> merged(total_cand);
    for (std::size_t i = 0; i < total_cand; ++i) {
        merged[i] = PatternStat{all_hash[i], all_cnt[i], Point{all_x[i], all_y[i]}};
    }
    const std::size_t k_global = std::min<std::size_t>(k, merged.size());
    std::partial_sort(merged.begin(), merged.begin() + k_global, merged.end(), stat_before);
    merged.resize(k_global);
    return merged;
}

UpsPatternStats::UpsPatternStats(stComm::Comm& comm)
    : impl_(std::make_unique<Impl>(comm)) {}

UpsPatternStats::~UpsPatternStats()                                      = default;
UpsPatternStats::UpsPatternStats(UpsPatternStats&&) noexcept             = default;
UpsPatternStats& UpsPatternStats::operator=(UpsPatternStats&&) noexcept = default;

std::vector<PatternStat> UpsPatternStats::pattern_stats(
    std::vector<std::vector<Hash>>  patches,
    std::vector<std::vector<Point>> coords,
    std::uint64_t k) {
    if (coords.size() != patches.size()) {
        throw std::invalid_argument("pattern_stats: coords/patches shape mismatch");
    }
    for (std::size_t p = 0; p < patches.size(); ++p) {
        if (coords[p].size() != patches[p].size()) {
            throw std::invalid_argument(
                "pattern_stats: coords/patches shape mismatch at patch " +
                std::to_string(p));
        }
    }
    return impl_->run(std::move(patches), std::move(coords), k);
}

}  // namespace stPS
