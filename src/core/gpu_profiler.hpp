#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace stPS::detail {

// Per-stage GPU timing via CUDA events. begin(stage) records a start event on
// the default stream; end(stage) records a stop event, synchronizes it, and
// accumulates cudaEventElapsedTime into that stage's running total. One event
// pair per stage is created up front (at construction) and reused every
// iteration, so the hot loop allocates nothing.
//
// All CUDA runtime calls live in the matching .cu; the event handles are held
// here as type-erased void* so this header stays consumable from any host TU
// without pulling in cuda_runtime.h (matching DeviceBufferImpl's convention).
//
// end() synchronizes the stop event, so enabling the profiler serializes stage
// boundaries and slightly inflates wall time — the relative per-stage shape is
// the signal, as with any sync-based profiler. When constructed with
// enabled=false, no events are created and begin/end/report are no-ops.
class GpuProfiler {
public:
    GpuProfiler() = default;
    GpuProfiler(std::vector<std::string> stage_names, bool enabled);
    ~GpuProfiler();

    GpuProfiler(const GpuProfiler&) = delete;
    GpuProfiler& operator=(const GpuProfiler&) = delete;
    GpuProfiler(GpuProfiler&& other) noexcept;
    GpuProfiler& operator=(GpuProfiler&& other) noexcept;

    bool enabled() const noexcept { return enabled_; }

    void begin(int stage);
    void end(int stage);

    // Print a "<title> iters=N" header followed by a per-stage
    // (per-iter μs avg, total ms) table. No-op when disabled.
    void report(std::FILE* out, const char* title, std::uint64_t iterations) const;

private:
    void destroy() noexcept;

    bool                     enabled_ = false;
    std::vector<std::string> names_;
    std::vector<void*>       start_;     // cudaEvent_t handles
    std::vector<void*>       stop_;      // cudaEvent_t handles
    std::vector<double>      accum_ms_;
};

}  // namespace stPS::detail
