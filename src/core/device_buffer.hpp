#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace stPS {

// Exception thrown when a CUDA runtime call fails inside DeviceBuffer.
struct CudaError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Non-template, type-erased owner of a device-side byte buffer. Move-only. All
// CUDA runtime calls (cudaMalloc / cudaFree / cudaMemcpy / cudaMemset) live in
// the matching .cu translation unit, so this header is consumable from any
// host C++ TU without pulling in cuda_runtime.h.
class DeviceBufferImpl {
public:
    DeviceBufferImpl() = default;
    ~DeviceBufferImpl();

    DeviceBufferImpl(const DeviceBufferImpl&) = delete;
    DeviceBufferImpl& operator=(const DeviceBufferImpl&) = delete;

    DeviceBufferImpl(DeviceBufferImpl&& other) noexcept;
    DeviceBufferImpl& operator=(DeviceBufferImpl&& other) noexcept;

    void allocate(std::size_t num_bytes);
    void deallocate() noexcept;

    void copy_from_host(const void* src, std::size_t num_bytes);
    void copy_to_host(void* dst, std::size_t num_bytes) const;
    void zero();

    void*       raw()       noexcept { return ptr_; }
    const void* raw() const noexcept { return ptr_; }
    std::size_t bytes() const noexcept { return bytes_; }

private:
    void*       ptr_   = nullptr;
    std::size_t bytes_ = 0;
};

// Typed, move-only RAII view over a contiguous device allocation of T. Pure
// header-only template; all CUDA work delegates to DeviceBufferImpl.
template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { impl_.allocate(count * sizeof(T)); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) noexcept = default;
    DeviceBuffer& operator=(DeviceBuffer&&) noexcept = default;

    T*          data()       noexcept { return static_cast<T*>(impl_.raw()); }
    const T*    data() const noexcept { return static_cast<const T*>(impl_.raw()); }
    std::size_t size()  const noexcept { return impl_.bytes() / sizeof(T); }
    std::size_t bytes() const noexcept { return impl_.bytes(); }

    void resize(std::size_t new_count) {
        const std::size_t new_bytes = new_count * sizeof(T);
        if (new_bytes == impl_.bytes()) return;
        impl_.deallocate();
        impl_.allocate(new_bytes);
    }

    void copy_from_host(const T* src, std::size_t count) {
        impl_.copy_from_host(src, count * sizeof(T));
    }
    void copy_to_host(T* dst, std::size_t count) const {
        impl_.copy_to_host(dst, count * sizeof(T));
    }
    void zero() { impl_.zero(); }

private:
    DeviceBufferImpl impl_;
};

}  // namespace stPS
