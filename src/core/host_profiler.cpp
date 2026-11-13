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
    starts_[static_cast<std::size_t>(stage)] = Clock::now();
}

void HostProfiler::end(int stage) {
    if (!enabled_) return;
    const auto s = static_cast<std::size_t>(stage);
    accum_ns_[s] += std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - starts_[s]).count();
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

PhaseTimer::PhaseTimer(bool enabled) : enabled_(enabled) {
    if (enabled_) last_ = Clock::now();
}

void PhaseTimer::mark(const char* name, Where where) {
    if (!enabled_) return;
    const auto now = Clock::now();
    phases_.push_back(Phase{name, where,
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - last_).count()});
    last_ = now;
}

void PhaseTimer::report(std::FILE* out, const char* title, int rank) const {
    if (!enabled_) return;
    static constexpr const char* kWhere[] = {"host", "host-comm", "transfer", "device", "mixed"};
    std::int64_t total_ns = 0;
    std::int64_t by_where[5] = {};
    for (const Phase& ph : phases_) {
        total_ns += ph.ns;
        by_where[static_cast<std::size_t>(ph.where)] += ph.ns;
    }
    const double total_ms = static_cast<double>(total_ns) * 1e-6;
    const auto share = [&](std::int64_t ns) {
        return total_ns > 0 ? 100.0 * static_cast<double>(ns) / static_cast<double>(total_ns) : 0.0;
    };
    std::fprintf(out, "%s rank=%d total=%.1f ms\n", title, rank, total_ms);
    for (const Phase& ph : phases_) {
        std::fprintf(out, "  %-34s %-9s %10.1f ms %5.1f%%\n", ph.name.c_str(),
                     kWhere[static_cast<std::size_t>(ph.where)],
                     static_cast<double>(ph.ns) * 1e-6, share(ph.ns));
    }
    for (std::size_t w = 0; w < 5; ++w) {
        if (by_where[w] == 0) continue;
        std::fprintf(out, "  = %-32s %-9s %10.1f ms %5.1f%%\n", "sum", kWhere[w],
                     static_cast<double>(by_where[w]) * 1e-6, share(by_where[w]));
    }
}

}  // namespace stPS::detail
