#pragma once

// CUB's two-phase calling convention in one place. Include only from files
// compiled with the CUDA toolkit on the include path.

#include "core/cuda_check.hpp"
#include "core/device_buffer.hpp"

#include <cstddef>

namespace stPS {

inline constexpr unsigned kDeviceBlock = 256;

inline unsigned device_blocks_for(std::uint64_t n) {
    return static_cast<unsigned>((n + kDeviceBlock - 1) / kDeviceBlock);
}

// `call(temp_storage, temp_bytes)` is invoked twice: once to size the
// temporary storage, once to run.
template <typename Call>
void cub_run(const char* what, Call&& call) {
    std::size_t bytes = 0;
    cuda_check(call(nullptr, bytes), what);
    DeviceBuffer<unsigned char> temp(bytes);
    cuda_check(call(temp.data(), bytes), what);
}

}  // namespace stPS
