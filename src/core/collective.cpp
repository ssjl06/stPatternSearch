#include "core/collective.hpp"

#include <stComm/stComm.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace stPS {

bool all_ranks_ok(stComm::Comm& comm, bool local_ok) {
    const std::int32_t ok = local_ok ? 1 : 0;
    std::int32_t all_ok = 0;
    comm.allreduce<stComm::Space::Host, std::int32_t>(
        &ok, &all_ok, 1, stComm::ReduceOp::Min)->wait();
    return all_ok != 0;
}

ExchangeCounts exchange_counts(stComm::Comm& comm, std::vector<std::size_t> send) {
    const int  size  = comm.getSize();
    const auto usize = static_cast<std::size_t>(size);
    const auto urank = static_cast<std::size_t>(comm.getRank());
    if (send.size() != usize) {
        throw std::logic_error("exchange_counts: one send count per rank required");
    }

    // matrix[s * size + r] = elements rank s sends to rank r. stComm has no
    // plain allgather; allgatherv with a uniform row length is the same thing.
    // (The matrix itself is size² values — tiny — so int counts are fine here.)
    std::vector<std::size_t> matrix(usize * usize);
    {
        std::vector<int> uniform(usize, size);
        comm.allgatherv<stComm::Space::Host, std::size_t>(
            send.data(), size, matrix.data(), uniform.data())->wait();
    }

    ExchangeCounts counts;
    counts.send = std::move(send);
    counts.recv.resize(usize);
    for (std::size_t s = 0; s < usize; ++s) counts.recv[s] = matrix[s * usize + urank];
    return counts;
}

}  // namespace stPS
