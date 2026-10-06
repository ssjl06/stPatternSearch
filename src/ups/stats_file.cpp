#include <stPS/ups_pattern_stats.hpp>

#include "io/parallel_file.hpp"

#include <stComm/stComm.h>

#include <sys/types.h>

#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace stPS {

void write_pattern_stats_file(stComm::Comm& comm, const std::string& path,
                              const std::vector<PatternStat>& stats) {
    const int size = comm.getSize();
    const int rank = comm.getRank();

    // Fixed-width records: every offset is a pure function of the line index,
    // so ranks write disjoint byte ranges with no coordination. %.17e
    // round-trips doubles exactly; width 25 covers sign + 3-digit exponent.
    //   hash 16-hex TAB count 20 TAB x 25 TAB y 25 NL = 90 bytes
    constexpr std::size_t kLineBytes = 90;
    constexpr const char* kLineFmt   = "%016lx\t%20lu\t%25.17e\t%25.17e\n";

    // The header uses only values every rank already agrees on, so its length
    // — and with it every line offset — is known everywhere without comm.
    char header[128];
    const int header_len = std::snprintf(header, sizeof(header),
        "# ups-pattern-stats k=%zu ranks=%d (hash, patch count, rep x, rep y)\n",
        stats.size(), size);
    if (header_len <= 0) throw std::runtime_error("ups-stats: header format failed");

    // This rank's contiguous line range (same split rule as patches-by-rank).
    const std::uint64_t k = stats.size();
    const std::uint64_t begin = (k * static_cast<std::uint64_t>(rank))     / size;
    const std::uint64_t end   = (k * (static_cast<std::uint64_t>(rank)+1)) / size;

    std::string buf;
    buf.reserve((end - begin) * kLineBytes + (rank == 0 ? header_len : 0));
    if (rank == 0) buf.append(header, static_cast<std::size_t>(header_len));
    for (std::uint64_t i = begin; i < end; ++i) {
        char line[kLineBytes + 1];
        const int len = std::snprintf(line, sizeof(line), kLineFmt,
                                      static_cast<unsigned long>(stats[i].hash),
                                      static_cast<unsigned long>(stats[i].count),
                                      stats[i].rep.x, stats[i].rep.y);
        if (len != static_cast<int>(kLineBytes)) {
            throw std::runtime_error("ups-stats: fixed-width line overflow");
        }
        buf.append(line, kLineBytes);
    }

    // Parallel single-file write: rank 0's buffer starts with the header.
    const off_t my_off = (rank == 0)
        ? 0
        : static_cast<off_t>(header_len) + static_cast<off_t>(begin * kLineBytes);
    const off_t total = static_cast<off_t>(header_len) + static_cast<off_t>(k * kLineBytes);
    io::write_shared_file(comm, path, "ups-stats", buf.data(), buf.size(), my_off, total);
}

}  // namespace stPS
