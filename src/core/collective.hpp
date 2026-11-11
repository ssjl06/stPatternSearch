#pragma once

#include <climits>
#include <cstdint>
#include <vector>

namespace stComm { class Comm; }

namespace stPS {

// Collective agreement on a per-rank condition: returns true iff `local_ok`
// holds on EVERY rank. A rank that detects a local problem (bad input slice,
// failed I/O, a size past a limit) must not throw on its own — its peers would
// walk into the next collective and wait forever. Instead every rank calls
// this with its own verdict and all of them throw (or carry on) together.
bool all_ranks_ok(stComm::Comm& comm, bool local_ok);

// Per-rank element counts for an alltoallv, in the int form stComm takes.
struct ExchangeCounts {
    std::vector<int> send;   // send[r] = elements this rank sends to rank r
    std::vector<int> recv;   // recv[s] = elements this rank receives from rank s
};

// Collective: turn this rank's 64-bit send counts (send[r] = elements bound for
// rank r, size == comm size) into the matching int send/recv counts for an
// alltoallv. The full rank × rank matrix is gathered, so every rank evaluates
// the same limit check and — if ANY rank would send or receive more than
// `max_total` elements in total — all throw std::length_error together,
// before any buffer is sized. (An int count or displacement past INT_MAX is
// otherwise undefined behaviour on the counting rank, or an exception on that
// rank alone.) `what` names the exchange in the message.
ExchangeCounts agree_exchange_counts(stComm::Comm& comm,
                                     const std::vector<std::uint64_t>& send,
                                     const char* what,
                                     std::uint64_t max_total = INT_MAX);

}  // namespace stPS
