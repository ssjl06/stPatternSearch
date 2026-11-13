#include "core/device_primitives.hpp"

#include "core/cub_run.hpp"
#include "core/cuda_check.hpp"

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <bit>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace stPS::device {

namespace {

using u64 = std::uint64_t;

__global__ void mark_heads_kernel(const u64* sorted, u64 n, u64* heads) {
    const u64 i = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    heads[i] = (i == 0 || sorted[i] != sorted[i - 1]) ? 1 : 0;
}

// `rank` is the inclusive prefix sum of the head flags: a head at i is the
// (rank[i] - 1)-th distinct value.
__global__ void scatter_heads_kernel(const u64* sorted, const u64* rank, u64 n, u64* out) {
    const u64 i = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i == 0 || sorted[i] != sorted[i - 1]) out[rank[i] - 1] = sorted[i];
}

__global__ void lower_bound_kernel(const u64* haystack, u64 n_haystack, const u64* needles,
                                   u64 n_needles, u64* out, u64 add) {
    const u64 i = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_needles) return;
    const u64 x = needles[i];
    u64 lo = 0, hi = n_haystack;  // first index with haystack[index] >= x
    while (lo < hi) {
        const u64 mid = lo + (hi - lo) / 2;
        if (haystack[mid] < x) lo = mid + 1; else hi = mid;
    }
    out[i] = add + lo;
}

// The last p with offsets[p] <= i: empty rows share an offset with their
// successor and are skipped by taking the last.
__global__ void expand_owners_kernel(const u64* offsets, u64 M, u64 total, u64* owners) {
    const u64 i = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    u64 lo = 0, hi = M;  // answer in [lo, hi)
    while (hi - lo > 1) {
        const u64 mid = lo + (hi - lo) / 2;
        if (offsets[mid] <= i) lo = mid; else hi = mid;
    }
    owners[i] = lo;
}

}  // namespace

int bits_for(u64 bound) {
    return bound > 1 ? static_cast<int>(std::bit_width(bound - 1)) : 1;
}

void copy(u64* dst, const u64* src, u64 n) {
    if (n == 0) return;
    cuda_check(cudaMemcpy(dst, src, n * sizeof(u64), cudaMemcpyDeviceToDevice),
               "device::copy");
}

u64 read_at(const u64* d_values, u64 index) {
    u64 v = 0;
    cuda_check(cudaMemcpy(&v, d_values + index, sizeof(v), cudaMemcpyDeviceToHost),
               "device::read_at");
    return v;
}

void sort_keys(DeviceBuffer<u64>& keys, int end_bit) {
    const u64 n = keys.size();
    if (n < 2) return;
    DeviceBuffer<u64> alt(n);
    cub::DoubleBuffer<u64> k(keys.data(), alt.data());
    cub_run("radix-sort keys", [&](void* t, std::size_t& b) {
        return cub::DeviceRadixSort::SortKeys(t, b, k, n, 0, end_bit);
    });
    if (k.Current() == alt.data()) std::swap(keys, alt);
}

void sort_pairs(DeviceBuffer<u64>& keys, DeviceBuffer<u64>& vals, int end_bit) {
    const u64 n = keys.size();
    if (vals.size() != n) throw std::invalid_argument("device::sort_pairs: size mismatch");
    if (n < 2) return;
    DeviceBuffer<u64> keys_alt(n), vals_alt(n);
    cub::DoubleBuffer<u64> k(keys.data(), keys_alt.data());
    cub::DoubleBuffer<u64> v(vals.data(), vals_alt.data());
    cub_run("radix-sort pairs", [&](void* t, std::size_t& b) {
        return cub::DeviceRadixSort::SortPairs(t, b, k, v, n, 0, end_bit);
    });
    if (k.Current() == keys_alt.data()) std::swap(keys, keys_alt);
    if (v.Current() == vals_alt.data()) std::swap(vals, vals_alt);
}

DeviceBuffer<u64> unique_sorted(const u64* sorted, u64 n) {
    if (n == 0) return {};
    DeviceBuffer<u64> rank(n);
    mark_heads_kernel<<<device_blocks_for(n), kDeviceBlock>>>(sorted, n, rank.data());
    cuda_launch_check("mark_heads_kernel");
    cub_run("prefix-sum heads", [&](void* t, std::size_t& b) {
        return cub::DeviceScan::InclusiveSum(t, b, rank.data(), rank.data(), n);
    });
    DeviceBuffer<u64> out(read_at(rank.data(), n - 1));
    scatter_heads_kernel<<<device_blocks_for(n), kDeviceBlock>>>(sorted, rank.data(), n,
                                                                out.data());
    cuda_launch_check("scatter_heads_kernel");
    cuda_sync_check("scatter_heads_kernel");  // `rank` is freed on return
    return out;
}

void lower_bound(const u64* haystack, u64 n_haystack, const u64* needles, u64 n_needles,
                 u64* out, u64 add) {
    if (n_needles == 0) return;
    lower_bound_kernel<<<device_blocks_for(n_needles), kDeviceBlock>>>(
        haystack, n_haystack, needles, n_needles, out, add);
    cuda_launch_check("lower_bound_kernel");
}

void expand_owners(const u64* offsets, u64 M, u64 total, u64* owners) {
    if (total == 0) return;
    expand_owners_kernel<<<device_blocks_for(total), kDeviceBlock>>>(offsets, M, total, owners);
    cuda_launch_check("expand_owners_kernel");
}

}  // namespace stPS::device
