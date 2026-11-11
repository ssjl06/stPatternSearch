#include "core/collective.hpp"

#include <stComm/stComm.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace stPS {

bool all_ranks_ok(stComm::Comm& comm, bool local_ok) {
    const std::int32_t ok = local_ok ? 1 : 0;
    std::int32_t all_ok = 0;
    comm.allreduce<stComm::Space::Host, std::int32_t>(
        &ok, &all_ok, 1, stComm::ReduceOp::Min)->wait();
    return all_ok != 0;
}

ExchangeCounts agree_exchange_counts(stComm::Comm& comm,
                                     const std::vector<std::uint64_t>& send,
                                     const char* what,
                                     std::uint64_t max_total) {
    const int  size  = comm.getSize();
    const auto usize = static_cast<std::size_t>(size);
    const auto urank = static_cast<std::size_t>(comm.getRank());
    if (send.size() != usize) {
        throw std::logic_error("agree_exchange_counts: one send count per rank required");
    }

    // matrix[s * size + r] = elements rank s sends to rank r. stComm has no
    // plain allgather; allgatherv with a uniform row length is the same thing.
    std::vector<std::uint64_t> matrix(usize * usize);
    {
        std::vector<int> uniform(usize, size);
        comm.allgatherv<stComm::Space::Host, std::uint64_t>(
            send.data(), size, matrix.data(), uniform.data())->wait();
    }

    // The matrix is identical everywhere, so this verdict is too.
    for (std::size_t a = 0; a < usize; ++a) {
        std::uint64_t sent = 0, received = 0;
        for (std::size_t b = 0; b < usize; ++b) {
            // Saturating: each term is checked before it can wrap the sum.
            const std::uint64_t row = matrix[a * usize + b];
            const std::uint64_t col = matrix[b * usize + a];
            sent     = (row > max_total || sent > max_total)     ? max_total + 1 : sent + row;
            received = (col > max_total || received > max_total) ? max_total + 1 : received + col;
        }
        if (sent > max_total || received > max_total) {
            throw std::length_error(
                std::string(what) + ": rank " + std::to_string(a) + " would " +
                (sent > max_total ? "send" : "receive") + " more than " +
                std::to_string(max_total) + " elements in one exchange — use more "
                "ranks or a smaller input");
        }
    }

    ExchangeCounts counts;
    counts.send.resize(usize);
    counts.recv.resize(usize);
    for (std::size_t r = 0; r < usize; ++r) {
        counts.send[r] = static_cast<int>(matrix[urank * usize + r]);
        counts.recv[r] = static_cast<int>(matrix[r * usize + urank]);
    }
    return counts;
}

}  // namespace stPS
