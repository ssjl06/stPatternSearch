#include "core/hash_id_map_device.hpp"

#include "core/collective.hpp"
#include "core/cub_run.hpp"
#include "core/cuda_check.hpp"
#include "core/device_primitives.hpp"
#include "core/host_profiler.hpp"
#include "core/patch_set.hpp"

#include <stComm/stComm.h>

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

#include <cstddef>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace stPS {

namespace {

using u64 = std::uint64_t;

// values[i] (a hash) becomes the ID paired with it: ids[j] where uniq[j] is
// that hash. Every hash is present in `uniq` by construction.
__global__ void translate_kernel(const u64* uniq, const u64* ids, u64 n_uniq,
                                 u64* values, u64 total) {
    const u64 i = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const u64 x = values[i];
    u64 lo = 0, hi = n_uniq;
    while (lo < hi) {
        const u64 mid = lo + (hi - lo) / 2;
        if (uniq[mid] < x) lo = mid + 1; else hi = mid;
    }
    values[i] = ids[lo];
}

// Pairs are sorted by (owner, id): keep[i] = 1 for the first of each run of
// equal pairs.
__global__ void mark_distinct_pairs_kernel(const u64* owners, const u64* ids, u64 total,
                                           u64* keep) {
    const u64 i = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    keep[i] = (i == 0 || owners[i] != owners[i - 1] || ids[i] != ids[i - 1]) ? 1 : 0;
}

// `rank` is the inclusive prefix sum of keep.
__global__ void compact_ids_kernel(const u64* owners, const u64* ids, const u64* rank,
                                   u64 total, u64* out) {
    const u64 i = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    if (i == 0 || owners[i] != owners[i - 1] || ids[i] != ids[i - 1]) out[rank[i] - 1] = ids[i];
}

// offsets[p] = kept pairs whose owner is below p, for p in [0, M].
__global__ void csr_offsets_kernel(const u64* owners, const u64* rank, u64 total, u64 M,
                                   u64* offsets) {
    const u64 p = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p > M) return;
    u64 lo = 0, hi = total;  // first pair with owner >= p
    while (lo < hi) {
        const u64 mid = lo + (hi - lo) / 2;
        if (owners[mid] < p) lo = mid + 1; else hi = mid;
    }
    offsets[p] = lo == 0 ? 0 : rank[lo - 1];
}

}  // namespace

DeviceIdMappedPatches map_hashes_to_element_ids_device(
        stComm::Comm& comm, std::vector<std::vector<Hash>> raw_patches) {
    if (!comm.hasDevice()) {
        throw std::invalid_argument(
            "map_hashes_to_element_ids_device requires a device-enabled Comm");
    }
    const int  size  = comm.getSize();
    const int  rank  = comm.getRank();
    const auto usize = static_cast<std::size_t>(size);
    using W = detail::PhaseTimer::Where;
    detail::PhaseTimer phases;

    DeviceIdMappedPatches out;
    const u64 M = raw_patches.size();
    out.M_local = M;

    // ---- Upload: one flat hash array + the patch offsets ---------------------
    std::vector<u64> offsets(M + 1, 0);
    for (u64 p = 0; p < M; ++p) offsets[p + 1] = offsets[p] + raw_patches[p].size();
    const u64 total = offsets[M];

    DeviceBuffer<u64> d_values(total);  // hashes now, element IDs after translation
    {
        std::vector<Hash> flat;
        flat.reserve(total);
        for (auto& p : raw_patches) {
            flat.insert(flat.end(), p.begin(), p.end());
            std::vector<Hash>().swap(p);  // release the hashes as we go
        }
        if (total) d_values.copy_from_host(flat.data(), total);
    }
    DeviceBuffer<u64> d_offsets(M + 1);
    d_offsets.copy_from_host(offsets.data(), M + 1);
    std::vector<u64>().swap(offsets);
    phases.mark("flatten + upload", W::Transfer);

    // ---- This rank's distinct hashes, sorted ---------------------------------
    DeviceBuffer<u64> d_uniq;
    {
        DeviceBuffer<u64> sorted(total);
        device::copy(sorted.data(), d_values.data(), total);
        device::sort_keys(sorted);
        d_uniq = device::unique_sorted(sorted.data(), total);
    }
    const u64 n_uniq = d_uniq.size();
    phases.mark("sort + unique", W::Device);

    // ---- Splitters: a few samples to the host, the rest as on the host path ---
    std::vector<Hash> sample;
    for (std::size_t idx : detail::splitter_sample_indices(n_uniq)) {
        sample.push_back(device::read_at(d_uniq.data(), idx));
    }
    const std::vector<Hash> splitters = detail::splitters_from_sample(comm, sample);

    // d_uniq is sorted, so it already is the send buffer in target-rank order;
    // the hashes for rank r end where the first hash >= splitters[r] sits.
    std::vector<std::size_t> sendcounts(usize, 0);
    if (size > 1) {
        DeviceBuffer<u64> d_split(usize - 1), d_pos(usize - 1);
        d_split.copy_from_host(splitters.data(), usize - 1);
        device::lower_bound(d_uniq.data(), n_uniq, d_split.data(), usize - 1, d_pos.data());
        std::vector<u64> pos(usize - 1);
        d_pos.copy_to_host(pos.data(), usize - 1);
        u64 prev = 0;
        for (std::size_t r = 0; r + 1 < usize; ++r) {
            sendcounts[r] = pos[r] - prev;
            prev = pos[r];
        }
        sendcounts[usize - 1] = n_uniq - prev;
    } else {
        sendcounts[0] = n_uniq;
    }
    const ExchangeCounts counts = exchange_counts(comm, std::move(sendcounts));
    const std::size_t total_recv =
        std::accumulate(counts.recv.begin(), counts.recv.end(), std::size_t{0});

    // ---- Every hash flies to its owner rank, device to device -----------------
    DeviceBuffer<u64> d_recv(total_recv);
    cuda_sync_check("pre hash exchange");  // kernel outputs visible before NCCL reads
    comm.alltoallv<stComm::Space::Device, Hash>(d_uniq.data(), counts.send.data(),
                                                d_recv.data(), counts.recv.data())->wait();
    phases.mark("splitters + hash exchange", W::Mixed);

    // ---- Owner side: this rank's shard, and an ID for every arrival slot ------
    DeviceBuffer<u64> d_bucket;
    {
        DeviceBuffer<u64> sorted(total_recv);
        device::copy(sorted.data(), d_recv.data(), total_recv);
        device::sort_keys(sorted);
        d_bucket = device::unique_sorted(sorted.data(), total_recv);
    }
    const u64 shard_size = d_bucket.size();
    u64 id_start = 0;
    comm.exscan<stComm::Space::Host>(shard_size, &id_start, stComm::ReduceOp::Sum)->wait();
    u64 N_total = (rank == size - 1) ? (id_start + shard_size) : 0;
    comm.bcast<stComm::Space::Host, u64>(&N_total, 1, size - 1)->wait();

    DeviceBuffer<u64> d_response(total_recv);
    device::lower_bound(d_bucket.data(), shard_size, d_recv.data(), total_recv,
                        d_response.data(), id_start);
    phases.mark("shard sort + id assignment", W::Device);

    // ---- IDs travel back: d_ids[i] is the ID of d_uniq[i] ---------------------
    DeviceBuffer<u64> d_ids(n_uniq);
    cuda_sync_check("pre id exchange");
    d_recv = DeviceBuffer<u64>();
    comm.alltoallv<stComm::Space::Device, ElementId>(d_response.data(), counts.recv.data(),
                                                     d_ids.data(), counts.send.data())->wait();
    d_response = DeviceBuffer<u64>();
    phases.mark("id return exchange", W::Device);

    out.N           = N_total;
    out.shard_start = id_start;
    out.shard_hashes.resize(shard_size);
    if (shard_size) d_bucket.copy_to_host(out.shard_hashes.data(), shard_size);
    d_bucket = DeviceBuffer<u64>();
    phases.mark("shard hashes to host", W::Transfer);

    // ---- Translate every occurrence, then sort + de-duplicate each patch ------
    DeviceBuffer<u64> d_csr_offsets(M + 1);
    if (total) {
        translate_kernel<<<device_blocks_for(total), kDeviceBlock>>>(
            d_uniq.data(), d_ids.data(), n_uniq, d_values.data(), total);
        cuda_launch_check("translate_kernel");
        cuda_sync_check("translate_kernel");  // the lookup tables are freed next
    }
    d_uniq = DeviceBuffer<u64>();
    d_ids  = DeviceBuffer<u64>();

    DeviceBuffer<u64> d_owners(total);
    device::expand_owners(d_offsets.data(), M, total, d_owners.data());
    // Two stable passes: by ID, then by owning patch — pairs end up ordered by
    // (patch, ID).
    device::sort_pairs(d_values, d_owners, device::bits_for(N_total));
    device::sort_pairs(d_owners, d_values, device::bits_for(M));

    u64 kept = 0;
    DeviceBuffer<u64> d_rank(total);
    if (total) {
        mark_distinct_pairs_kernel<<<device_blocks_for(total), kDeviceBlock>>>(
            d_owners.data(), d_values.data(), total, d_rank.data());
        cuda_launch_check("mark_distinct_pairs_kernel");
        cub_run("prefix-sum distinct pairs", [&](void* t, std::size_t& b) {
            return cub::DeviceScan::InclusiveSum(t, b, d_rank.data(), d_rank.data(), total);
        });
        kept = device::read_at(d_rank.data(), total - 1);
    }
    out.data.resize(kept);
    if (total) {
        compact_ids_kernel<<<device_blocks_for(total), kDeviceBlock>>>(
            d_owners.data(), d_values.data(), d_rank.data(), total, out.data.data());
        cuda_launch_check("compact_ids_kernel");
    }
    csr_offsets_kernel<<<device_blocks_for(M + 1), kDeviceBlock>>>(
        d_owners.data(), d_rank.data(), total, M, d_csr_offsets.data());
    cuda_launch_check("csr_offsets_kernel");
    cuda_sync_check("csr_offsets_kernel");  // the pair buffers are freed on return
    out.offsets = std::move(d_csr_offsets);

    phases.mark("translate + order patches", W::Device);
    phases.report(stderr, "[stps-profile hash->id map (device)]", rank);
    return out;
}

}  // namespace stPS
