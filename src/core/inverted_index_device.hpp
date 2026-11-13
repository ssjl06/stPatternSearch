#pragma once

#include "core/csr.hpp"
#include "core/device_buffer.hpp"
#include <stPS/types.hpp>

#include <cstdint>

namespace stPS {

// An InvertedIndex (see core/inverted_index.hpp) living in device memory:
// the same three arrays, with the same contents build_inverted_index gives.
struct DeviceInvertedIndex {
    DeviceBuffer<ElementId>     keys;
    DeviceBuffer<std::uint64_t> offsets;  // num_keys + 1 entries
    DeviceBuffer<PatchId>       data;
    std::uint64_t               num_keys = 0;
};

// Transpose a PatchCsr on the GPU. The host arrays are uploaded once; the
// (element, patch) pairs are expanded, sorted and grouped on the device.
//
// Result is bit-identical to build_inverted_index(patches, N): keys ascending,
// and each key's patch IDs ascending — the pairs start in patch order and the
// radix sort is stable.
//
// `N` bounds the element IDs (every ID < N); it only limits how many key bits
// the sort has to look at. Peak device memory is 32 B per incidence while the
// sort runs (two key and two value buffers), 8 B of which stay as `data`.
DeviceInvertedIndex build_inverted_index_device(const PatchCsr& patches, std::uint64_t N);

}  // namespace stPS
