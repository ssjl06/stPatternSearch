#pragma once

#include <cstddef>
#include <vector>

namespace stComm { class Comm; }

namespace stPS {

// Collective agreement on a per-rank condition: returns true iff `local_ok`
// holds on EVERY rank. A rank that detects a local problem (bad input slice,
// failed I/O, a size past a limit) must not throw on its own — its peers would
// walk into the next collective and wait forever. Instead every rank calls
// this with its own verdict and all of them throw (or carry on) together.
bool all_ranks_ok(stComm::Comm& comm, bool local_ok);

// Per-rank element counts for an alltoallv, in the size_t form stComm's
// unlimited-size collectives take. 64-bit throughout: a full-chip input puts
// far more than INT_MAX elements on one rank, and nothing here may cap that.
struct ExchangeCounts {
    std::vector<std::size_t> send;   // send[r] = elements this rank sends to rank r
    std::vector<std::size_t> recv;   // recv[s] = elements this rank receives from rank s
};

// Collective: given this rank's send counts (send[r] = elements bound for rank
// r, size == comm size), learn the matching receive counts — recv[s] is what
// rank s reported for this rank.
ExchangeCounts exchange_counts(stComm::Comm& comm, std::vector<std::size_t> send);

}  // namespace stPS
