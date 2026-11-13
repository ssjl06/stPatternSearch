#pragma once

#include "core/device_buffer.hpp"

#include <cstdint>

// Device-side building blocks shared by the setup stages (hash → ID mapping,
// inverted index). Everything works on 64-bit values and takes 64-bit counts:
// only CUB entry points without an int item-count limit are used underneath.
// Declarations only — includable from host translation units.
namespace stPS::device {

// Bits needed to represent every value below `bound` (at least 1) — the
// `end_bit` of a radix sort whose keys are all < bound.
int bits_for(std::uint64_t bound);

// dst[0, n) = src[0, n), device to device.
void copy(std::uint64_t* dst, const std::uint64_t* src, std::uint64_t n);

// One element read back to the host.
std::uint64_t read_at(const std::uint64_t* d_values, std::uint64_t index);

// Sort all of `keys` ascending, looking at bits [0, end_bit). The buffer may
// be swapped for another of the same size. Peak: one extra buffer.
void sort_keys(DeviceBuffer<std::uint64_t>& keys, int end_bit = 64);

// Stable sort of (key, value) pairs by key bits [0, end_bit). Both buffers
// hold the same number of entries and may be swapped. Peak: two extra buffers.
void sort_pairs(DeviceBuffer<std::uint64_t>& keys, DeviceBuffer<std::uint64_t>& vals,
                int end_bit = 64);

// The distinct values of sorted[0, n), ascending, in a buffer of exactly that
// many entries.
DeviceBuffer<std::uint64_t> unique_sorted(const std::uint64_t* sorted, std::uint64_t n);

// out[i] = add + (number of entries of the sorted haystack below needles[i]).
void lower_bound(const std::uint64_t* haystack, std::uint64_t n_haystack,
                 const std::uint64_t* needles, std::uint64_t n_needles,
                 std::uint64_t* out, std::uint64_t add = 0);

// owners[i] = the row whose CSR range [offsets[p], offsets[p+1]) holds i, for
// i in [0, total) with total == offsets[M].
void expand_owners(const std::uint64_t* offsets, std::uint64_t M, std::uint64_t total,
                   std::uint64_t* owners);

}  // namespace stPS::device
