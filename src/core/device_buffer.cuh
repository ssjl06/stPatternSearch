#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace fullchipusc {

// Minimal CUDA error check — throws on failure. Used inside DeviceBuffer where
// a leak from a half-constructed allocation would be worse than an exception.
inline void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                                 cudaGetErrorString(err));
    }
}

// RAII owner for a device-side buffer of trivially-copyable T. Move-only.
//
// Purpose: keep every cudaMalloc paired with a cudaFree across exceptions and
// scope exits, with a small typed API that hides the byte-counting boilerplate
// from USCSolver. Resize re-allocates from scratch (no copy of existing data).
template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) { allocate(count); }

    ~DeviceBuffer() { free_internal(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : d_ptr_(other.d_ptr_), count_(other.count_) {
        other.d_ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            free_internal();
            d_ptr_ = other.d_ptr_;
            count_ = other.count_;
            other.d_ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    T*          data()       noexcept { return d_ptr_; }
    const T*    data() const noexcept { return d_ptr_; }
    std::size_t size()  const noexcept { return count_; }
    std::size_t bytes() const noexcept { return count_ * sizeof(T); }

    void resize(std::size_t new_count) {
        if (new_count == count_) return;
        free_internal();
        allocate(new_count);
    }

    // Synchronous host → device copy. Caller-supplied count may be < size().
    void copy_from_host(const T* src, std::size_t count) {
        if (count == 0) return;
        cuda_check(cudaMemcpy(d_ptr_, src, count * sizeof(T),
                              cudaMemcpyHostToDevice),
                   "DeviceBuffer::copy_from_host");
    }

    // Synchronous device → host copy.
    void copy_to_host(T* dst, std::size_t count) const {
        if (count == 0) return;
        cuda_check(cudaMemcpy(dst, d_ptr_, count * sizeof(T),
                              cudaMemcpyDeviceToHost),
                   "DeviceBuffer::copy_to_host");
    }

    // Memset to zero (commonly used for bitsets and freshly-allocated scratch).
    void zero() {
        if (count_ == 0) return;
        cuda_check(cudaMemset(d_ptr_, 0, bytes()), "DeviceBuffer::zero");
    }

private:
    void allocate(std::size_t count) {
        if (count == 0) {
            d_ptr_ = nullptr;
            count_ = 0;
            return;
        }
        void* p = nullptr;
        cuda_check(cudaMalloc(&p, count * sizeof(T)), "DeviceBuffer::allocate");
        d_ptr_ = static_cast<T*>(p);
        count_ = count;
    }

    void free_internal() noexcept {
        if (d_ptr_) {
            // Destructor must not throw — log instead of throwing on failure.
            const cudaError_t err = cudaFree(d_ptr_);
            if (err != cudaSuccess) {
                std::fprintf(stderr, "DeviceBuffer::free cudaFree failed: %s\n",
                             cudaGetErrorString(err));
            }
            d_ptr_ = nullptr;
            count_ = 0;
        }
    }

    T*          d_ptr_ = nullptr;
    std::size_t count_ = 0;
};

}  // namespace fullchipusc
