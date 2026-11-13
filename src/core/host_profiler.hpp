#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace stPS::detail {

// Compile-time switch for the phase timers: -DUSC_PROFILE=ON.
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
