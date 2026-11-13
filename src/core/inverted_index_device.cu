#include "core/inverted_index_device.hpp"

#include "core/cub_run.hpp"
#include "core/cuda_check.hpp"
#include "core/device_primitives.hpp"

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <cstddef>
#include <utility>

namespace stPS {

namespace {

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

}  // namespace

DeviceInvertedIndex build_inverted_index_device(const PatchCsr& patches, std::uint64_t N) {
    const std::uint64_t total = patches.data.size();
    const std::uint64_t M     = patches.M();
    DeviceBuffer<ElementId>     d_data(total);
    DeviceBuffer<std::uint64_t> d_offsets(M + 1);
    if (total) d_data.copy_from_host(patches.data.data(), total);
    if (patches.offsets.empty()) d_offsets.zero();
    else d_offsets.copy_from_host(patches.offsets.data(), M + 1);
    return build_inverted_index_device(d_data.data(), d_offsets.data(), M, total, N);
}

DeviceInvertedIndex build_inverted_index_device(const ElementId* d_data,
                                                const std::uint64_t* d_offsets,
                                                std::uint64_t M, std::uint64_t total,
                                                std::uint64_t N) {
    DeviceInvertedIndex out;
    if (total == 0) {
        const std::uint64_t zero = 0;
        out.offsets.resize(1);
        out.offsets.copy_from_host(&zero, 1);
        return out;
    }

    // Pairs in patch order: keys = element IDs (the CSR data as is), values =
    // owning patch, expanded from the offsets.
    DeviceBuffer<ElementId> d_keys(total), d_keys_alt(total);
    DeviceBuffer<PatchId>   d_vals(total), d_vals_alt(total);
    device::copy(d_keys.data(), d_data, total);
    device::expand_owners(d_offsets, M, total, d_vals.data());

    // Stable sort by element ID, looking only at the bits an ID below N can set.
    cub::DoubleBuffer<ElementId> keys(d_keys.data(), d_keys_alt.data());
    cub::DoubleBuffer<PatchId>   vals(d_vals.data(), d_vals_alt.data());
    const int end_bit = device::bits_for(N);
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
    mark_heads_kernel<<<device_blocks_for(total), kDeviceBlock>>>(sorted_keys.data(), total, d_rank);
    cuda_launch_check("mark_heads_kernel");
    cub_run("prefix-sum group heads", [&](void* t, std::size_t& b) {
        return cub::DeviceScan::InclusiveSum(t, b, d_rank, d_rank, total);
    });
    cuda_check(cudaMemcpy(&out.num_keys, d_rank + (total - 1), sizeof(out.num_keys),
                          cudaMemcpyDeviceToHost),
               "inverted index key count");

    out.keys.resize(out.num_keys);
    out.offsets.resize(out.num_keys + 1);
    scatter_groups_kernel<<<device_blocks_for(total), kDeviceBlock>>>(
        sorted_keys.data(), d_rank, total, out.keys.data(), out.offsets.data());
    cuda_launch_check("scatter_groups_kernel");
    cuda_sync_check("scatter_groups_kernel");  // the sort buffers are freed on return

    out.data = std::move(vals_in_alt ? d_vals_alt : d_vals);
    return out;
}

}  // namespace stPS
