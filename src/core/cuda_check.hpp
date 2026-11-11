#pragma once

// CUDA error → exception helpers shared by the .cu translation units. Include
// only from files compiled with the CUDA toolkit on the include path.

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace stPS {

// After a kernel launch (or a CUB call): throws if the launch itself failed,
// so a bug surfaces at the responsible kernel rather than later as an opaque
// memcpy failure.
inline void cuda_launch_check(const char* what) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA launch error in ") + what +
                                 ": " + cudaGetErrorString(err));
    }
}

// For calls that return a status (cudaMemcpy, CUB): an earlier kernel's fault
// surfaces as their return code, so dropping it would let the caller run on
// garbage values.
inline void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + what +
                                 ": " + cudaGetErrorString(err));
    }
}

// Block until every stream is idle, throwing on any pending device error.
inline void cuda_sync_check(const char* what) {
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA sync error in ") + what +
                                 ": " + cudaGetErrorString(err));
    }
}

}  // namespace stPS
