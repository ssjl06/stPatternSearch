#include "core/inverted_index_device.hpp"

#include "core/cuda_check.hpp"

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <bit>
#include <cstddef>
#include <utility>

namespace stPS {

namespace {

constexpr unsigned kBlock = 256;

unsigned blocks_for(std::uint64_t n) {
    return static_cast<unsigned>((n + kBlock - 1) / kBlock);
}

// owners[i] = the patch whose CSR range holds incidence i: the last p with
// offsets[p] <= i (empty patches share an offset with their successor and are
// skipped by taking the last).
__global__ void expand_owners_kernel(const std::uint64_t* offsets, std::uint64_t M,
                                     std::uint64_t total, PatchId* owners) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    std::uint64_t lo = 0, hi = M;  // answer in [lo, hi)
    while (hi - lo > 1) {
        const std::uint64_t mid = lo + (hi - lo) / 2;
        if (offsets[mid] <= i) lo = mid; else hi = mid;
    }
    owners[i] = lo;
}

// heads[i] = 1 where a new key starts in the sorted key array.
__global__ void mark_heads_kernel(const ElementId* keys, std::uint64_t total,
                                  std::uint64_t* heads) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    heads[i] = (i == 0 || keys[i] != keys[i - 1]) ? 1 : 0;
}

// `rank` is the inclusive prefix sum of the head flags, so a head at i is the
// (rank[i] - 1)-th distinct key and its group starts at i.
__global__ void scatter_groups_kernel(const ElementId* keys, const std::uint64_t* rank,
                                      std::uint64_t total, ElementId* out_keys,
                                      std::uint64_t* out_offsets) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    if (i == 0 || keys[i] != keys[i - 1]) {
        out_keys[rank[i] - 1]    = keys[i];
        out_offsets[rank[i] - 1] = i;
    }
    if (i == total - 1) out_offsets[rank[i]] = total;
}

// Two-phase CUB call: size the temporary storage, then run.
template <typename Call>
void cub_run(const char* what, Call&& call) {
    std::size_t bytes = 0;
    cuda_check(call(nullptr, bytes), what);
    DeviceBuffer<unsigned char> temp(bytes);
    cuda_check(call(temp.data(), bytes), what);
}

}  // namespace

DeviceInvertedIndex build_inverted_index_device(const PatchCsr& patches, std::uint64_t N) {
    DeviceInvertedIndex out;
    const std::uint64_t total = patches.data.size();
    if (total == 0) {
        const std::uint64_t zero = 0;
        out.offsets.resize(1);
        out.offsets.copy_from_host(&zero, 1);
        return out;
    }
    const std::uint64_t M = patches.M();

    // Pairs in patch order: keys = element IDs (the CSR data as is), values =
    // owning patch, expanded from the offsets on the device.
    DeviceBuffer<ElementId> d_keys(total), d_keys_alt(total);
    DeviceBuffer<PatchId>   d_vals(total), d_vals_alt(total);
    d_keys.copy_from_host(patches.data.data(), total);
    {
        DeviceBuffer<std::uint64_t> d_offsets(M + 1);
        d_offsets.copy_from_host(patches.offsets.data(), M + 1);
        expand_owners_kernel<<<blocks_for(total), kBlock>>>(d_offsets.data(), M, total,
                                                           d_vals.data());
        cuda_launch_check("expand_owners_kernel");
        cuda_sync_check("expand_owners_kernel");  // d_offsets is freed at scope exit
    }

    // Stable sort by element ID, looking only at the bits an ID below N can set.
    cub::DoubleBuffer<ElementId> keys(d_keys.data(), d_keys_alt.data());
    cub::DoubleBuffer<PatchId>   vals(d_vals.data(), d_vals_alt.data());
    const int end_bit = N > 1 ? static_cast<int>(std::bit_width(N - 1)) : 1;
    cub_run("radix-sort incidences", [&](void* t, std::size_t& b) {
        return cub::DeviceRadixSort::SortPairs(t, b, keys, vals, total, 0, end_bit);
    });
    const bool keys_in_alt = keys.Current() == d_keys_alt.data();
    const bool vals_in_alt = vals.Current() == d_vals_alt.data();
    DeviceBuffer<ElementId>& sorted_keys = keys_in_alt ? d_keys_alt : d_keys;
    DeviceBuffer<ElementId>& spare       = keys_in_alt ? d_keys : d_keys_alt;

    // Group boundaries: head flags → inclusive prefix sum, in the buffer the
    // sort no longer needs.
    static_assert(sizeof(ElementId) == sizeof(std::uint64_t));
    std::uint64_t* d_rank = spare.data();
    mark_heads_kernel<<<blocks_for(total), kBlock>>>(sorted_keys.data(), total, d_rank);
    cuda_launch_check("mark_heads_kernel");
    cub_run("prefix-sum group heads", [&](void* t, std::size_t& b) {
        return cub::DeviceScan::InclusiveSum(t, b, d_rank, d_rank, total);
    });
    cuda_check(cudaMemcpy(&out.num_keys, d_rank + (total - 1), sizeof(out.num_keys),
                          cudaMemcpyDeviceToHost),
               "inverted index key count");

    out.keys.resize(out.num_keys);
    out.offsets.resize(out.num_keys + 1);
    scatter_groups_kernel<<<blocks_for(total), kBlock>>>(
        sorted_keys.data(), d_rank, total, out.keys.data(), out.offsets.data());
    cuda_launch_check("scatter_groups_kernel");
    cuda_sync_check("scatter_groups_kernel");  // the sort buffers are freed on return

    out.data = std::move(vals_in_alt ? d_vals_alt : d_vals);
    return out;
}

}  // namespace stPS
