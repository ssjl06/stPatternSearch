#include "core/device_buffer.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>

namespace stPS {

namespace {
void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw CudaError(std::string("CUDA error in ") + what + ": " +
                        cudaGetErrorString(err));
    }
}
}  // namespace

DeviceBufferImpl::~DeviceBufferImpl() { deallocate(); }

DeviceBufferImpl::DeviceBufferImpl(DeviceBufferImpl&& other) noexcept
    : ptr_(other.ptr_), bytes_(other.bytes_) {
    other.ptr_   = nullptr;
    other.bytes_ = 0;
}

DeviceBufferImpl& DeviceBufferImpl::operator=(DeviceBufferImpl&& other) noexcept {
    if (this != &other) {
        deallocate();
        ptr_         = other.ptr_;
        bytes_       = other.bytes_;
        other.ptr_   = nullptr;
        other.bytes_ = 0;
    }
    return *this;
}

void DeviceBufferImpl::allocate(std::size_t num_bytes) {
    if (num_bytes == 0) {
        ptr_   = nullptr;
        bytes_ = 0;
        return;
    }
    check(cudaMalloc(&ptr_, num_bytes), "DeviceBufferImpl::allocate");
    bytes_ = num_bytes;
}

void DeviceBufferImpl::deallocate() noexcept {
    if (ptr_) {
        const cudaError_t err = cudaFree(ptr_);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "DeviceBufferImpl::deallocate cudaFree failed: %s\n",
                         cudaGetErrorString(err));
        }
        ptr_   = nullptr;
        bytes_ = 0;
    }
}

void DeviceBufferImpl::copy_from_host(const void* src, std::size_t num_bytes) {
    if (num_bytes == 0) return;
    check(cudaMemcpy(ptr_, src, num_bytes, cudaMemcpyHostToDevice),
          "DeviceBufferImpl::copy_from_host");
}

void DeviceBufferImpl::copy_to_host(void* dst, std::size_t num_bytes) const {
    if (num_bytes == 0) return;
    check(cudaMemcpy(dst, ptr_, num_bytes, cudaMemcpyDeviceToHost),
          "DeviceBufferImpl::copy_to_host");
}

void DeviceBufferImpl::zero() {
    if (bytes_ == 0) return;
    check(cudaMemset(ptr_, 0, bytes_), "DeviceBufferImpl::zero");
}

}  // namespace stPS
