#include "core/gpu_profiler.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace stPS::detail {

GpuProfiler::GpuProfiler(std::vector<std::string> stage_names, bool enabled)
    : enabled_(enabled), names_(std::move(stage_names)) {
    if (!enabled_) return;
    const std::size_t n = names_.size();
    start_.resize(n, nullptr);
    stop_.resize(n, nullptr);
    accum_ms_.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        cudaEvent_t s = nullptr;
        cudaEvent_t e = nullptr;
        cudaEventCreate(&s);
        cudaEventCreate(&e);
        start_[i] = s;
        stop_[i]  = e;
    }
}

void GpuProfiler::destroy() noexcept {
    for (void* e : start_) if (e) cudaEventDestroy(static_cast<cudaEvent_t>(e));
    for (void* e : stop_)  if (e) cudaEventDestroy(static_cast<cudaEvent_t>(e));
    start_.clear();
    stop_.clear();
}

GpuProfiler::~GpuProfiler() { destroy(); }

GpuProfiler::GpuProfiler(GpuProfiler&& other) noexcept
    : enabled_(other.enabled_),
      names_(std::move(other.names_)),
      start_(std::move(other.start_)),
      stop_(std::move(other.stop_)),
      accum_ms_(std::move(other.accum_ms_)) {
    other.enabled_ = false;
}

GpuProfiler& GpuProfiler::operator=(GpuProfiler&& other) noexcept {
    if (this != &other) {
        destroy();
        enabled_  = other.enabled_;
        names_    = std::move(other.names_);
        start_    = std::move(other.start_);
        stop_     = std::move(other.stop_);
        accum_ms_ = std::move(other.accum_ms_);
        other.enabled_ = false;
    }
    return *this;
}

void GpuProfiler::begin(int stage) {
    if (!enabled_) return;
    cudaEventRecord(static_cast<cudaEvent_t>(start_[stage]));
}

void GpuProfiler::end(int stage) {
    if (!enabled_) return;
    cudaEvent_t s = static_cast<cudaEvent_t>(start_[stage]);
    cudaEvent_t e = static_cast<cudaEvent_t>(stop_[stage]);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, s, e);
    accum_ms_[stage] += static_cast<double>(ms);
}

void GpuProfiler::report(std::FILE* out, const char* title,
                         std::uint64_t iterations) const {
    if (!enabled_) return;
    const double iters = static_cast<double>(iterations ? iterations : 1);
    std::fprintf(out, "%s iters=%llu  (per-iter μs avg, total ms)\n",
                 title, static_cast<unsigned long long>(iterations));
    for (std::size_t i = 0; i < names_.size(); ++i) {
        std::fprintf(out, "  %-16s %7.1f μs    %7.1f ms\n",
                     names_[i].c_str(),
                     accum_ms_[i] / iters * 1000.0,
                     accum_ms_[i]);
    }
}

}  // namespace stPS::detail
