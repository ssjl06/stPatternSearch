#include <stPS/ups_pattern_stats.hpp>

#include <stComm/stComm.h>

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace stPS {

namespace {

[[noreturn]] void fail_io(const std::string& path, const std::string& what) {
    throw std::runtime_error("ups-stats: " + path + ": " + what +
                             ": " + std::strerror(errno));
}

}  // namespace

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

    // Parallel single-file write; assumes a POSIX-coherent shared filesystem
    // (local disk, Lustre, GPFS — the usual cluster cases).
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT, 0644);
    if (fd < 0) fail_io(path, "cannot open");
    const off_t my_off = (rank == 0)
        ? 0
        : static_cast<off_t>(header_len) + static_cast<off_t>(begin * kLineBytes);
    for (std::size_t done = 0; done < buf.size();) {
        const ssize_t w = ::pwrite(fd, buf.data() + done, buf.size() - done,
                                   my_off + static_cast<off_t>(done));
        if (w < 0) { ::close(fd); fail_io(path, "pwrite failed"); }
        done += static_cast<std::size_t>(w);
    }

    // A stale, longer file from a previous run would leave garbage past our
    // records — once everyone has written, rank 0 cuts to the exact size.
    comm.barrier();
    if (rank == 0) {
        const off_t total = static_cast<off_t>(header_len) +
                            static_cast<off_t>(k * kLineBytes);
        if (::ftruncate(fd, total) != 0) { ::close(fd); fail_io(path, "ftruncate failed"); }
    }
    ::close(fd);
    comm.barrier();  // no rank returns before the file is complete
}

}  // namespace stPS
