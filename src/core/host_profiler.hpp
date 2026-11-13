#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace stPS::detail {

// Per-stage host (CPU wall-clock) timing for stages whose cost lands on the
// host rather than on a timed CUDA stream — e.g. MPI host-space collectives, or
// a device collective (NCCL) whose kernels run on a backend-internal stream
// that default-stream CUDA events cannot see. In those cases the stage's
// contribution to iteration latency is the host-blocking wait(), which a plain
// steady_clock captures faithfully.
//
// begin(stage) captures a start point; end(stage) accumulates the elapsed into
// that stage's running total. No CUDA dependency. When constructed with
// enabled=false, begin/end/report are no-ops.
class HostProfiler {
public:
    HostProfiler() = default;
    HostProfiler(std::vector<std::string> stage_names, bool enabled);

    bool enabled() const noexcept { return enabled_; }

    void begin(int stage);
    void end(int stage);

    // Print a "<title> iters=N" header followed by a per-stage
    // (per-iter μs avg, total ms) table. No-op when disabled.
    void report(std::FILE* out, const char* title, std::uint64_t iterations) const;

private:
    using Clock = std::chrono::steady_clock;

    bool                           enabled_ = false;
    std::vector<std::string>       names_;
    std::vector<Clock::time_point> starts_;
    std::vector<std::int64_t>      accum_ns_;
};

// Compile-time switch shared by every profiler: -DUSC_PROFILE=ON.
#ifdef USC_PROFILE
inline constexpr bool kProfileEnabled = true;
#else
inline constexpr bool kProfileEnabled = false;
#endif

// Wall-clock split of one pipeline run into consecutive phases, each tagged
// with where the time goes — the question it answers is "how long does the
// GPU sit idle?". mark() closes the phase that began at the previous mark (or
// at construction). A Device phase must end on a synchronizing call (a
// blocking copy, a wait()), or its tail is billed to the next phase.
class PhaseTimer {
public:
    enum class Where {
        Host,     // CPU work; GPU idle
        HostComm, // host MPI exchange; GPU idle
        Transfer, // host <-> device copies
        Device,   // GPU kernels and device-direct (NCCL) exchanges
        Mixed,    // GPU work interleaved with host round-trips
    };

    explicit PhaseTimer(bool enabled = kProfileEnabled);

    void mark(const char* name, Where where);

    // "<title> rank=R total=… ms", one line per phase (ms, share of total),
    // then the share per Where. No-op when disabled.
    void report(std::FILE* out, const char* title, int rank) const;

private:
    using Clock = std::chrono::steady_clock;
    struct Phase { std::string name; Where where; std::int64_t ns; };

    bool               enabled_ = false;
    Clock::time_point  last_;
    std::vector<Phase> phases_;
};

}  // namespace stPS::detail
