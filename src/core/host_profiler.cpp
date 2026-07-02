#include "core/host_profiler.hpp"

#include <cstddef>
#include <utility>

namespace stPS::detail {

HostProfiler::HostProfiler(std::vector<std::string> stage_names, bool enabled)
    : enabled_(enabled), names_(std::move(stage_names)) {
    if (!enabled_) return;
    starts_.resize(names_.size());
    accum_ns_.assign(names_.size(), 0);
}

void HostProfiler::begin(int stage) {
    if (!enabled_) return;
    starts_[stage] = Clock::now();
}

void HostProfiler::end(int stage) {
    if (!enabled_) return;
    accum_ns_[stage] += std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - starts_[stage]).count();
}

void HostProfiler::report(std::FILE* out, const char* title,
                          std::uint64_t iterations) const {
    if (!enabled_) return;
    const double iters = static_cast<double>(iterations ? iterations : 1);
    std::fprintf(out, "%s iters=%llu  (per-iter μs avg, total ms)\n",
                 title, static_cast<unsigned long long>(iterations));
    for (std::size_t i = 0; i < names_.size(); ++i) {
        const double total_ms = static_cast<double>(accum_ns_[i]) * 1e-6;
        std::fprintf(out, "  %-16s %7.1f μs    %7.1f ms\n",
                     names_[i].c_str(),
                     total_ms / iters * 1000.0,
                     total_ms);
    }
}

}  // namespace stPS::detail
