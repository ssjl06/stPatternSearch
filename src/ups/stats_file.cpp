#include <stPS/ups_pattern_stats.hpp>

#include <stComm/stComm.h>

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace stPS {

namespace {

// Message for a failed POSIX call — read errno right away, before anything
// else can clobber it.
std::string io_error(const char* what) {
    return std::string(what) + ": " + std::strerror(errno);
}

// Collective success check: every rank learns whether any rank's I/O failed,
// so all throw together instead of one rank throwing and stranding its peers
// in the next collective. Doubles as the barrier between write phases.
void agree_or_throw(stComm::Comm& comm, const std::string& path,
                    const std::string& local_err) {
    const std::int32_t ok = local_err.empty() ? 1 : 0;
    std::int32_t all_ok = 0;
    comm.allreduce<stComm::Space::Host, std::int32_t>(
        &ok, &all_ok, 1, stComm::ReduceOp::Min)->wait();
    if (!all_ok) {
        throw std::runtime_error("ups-stats: " + path + ": " +
            (local_err.empty() ? std::string("I/O failed on another rank") : local_err));
    }
}

// Closes on scope exit, so a throw from agree_or_throw never leaks the fd.
struct FdGuard {
    int fd = -1;
    ~FdGuard() { if (fd >= 0) ::close(fd); }
    int release() { const int f = fd; fd = -1; return f; }
};

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
    // (local disk, Lustre, GPFS — the usual cluster cases). Errors are
    // collected, not thrown, until every rank has agreed on the outcome.
    std::string err;
    FdGuard file;
    file.fd = ::open(path.c_str(), O_WRONLY | O_CREAT, 0644);
    if (file.fd < 0) err = io_error("cannot open");
    const off_t my_off = (rank == 0)
        ? 0
        : static_cast<off_t>(header_len) + static_cast<off_t>(begin * kLineBytes);
    for (std::size_t done = 0; err.empty() && done < buf.size();) {
        const ssize_t w = ::pwrite(file.fd, buf.data() + done, buf.size() - done,
                                   my_off + static_cast<off_t>(done));
        if (w < 0) { err = io_error("pwrite failed"); break; }
        done += static_cast<std::size_t>(w);
    }
    agree_or_throw(comm, path, err);  // everyone has written

    // A stale, longer file from a previous run would leave garbage past our
    // records — once everyone has written, rank 0 cuts to the exact size.
    if (rank == 0) {
        const off_t total = static_cast<off_t>(header_len) +
                            static_cast<off_t>(k * kLineBytes);
        if (::ftruncate(file.fd, total) != 0) err = io_error("ftruncate failed");
    }
    // close() can be the first to report a deferred write error (NFS, quota).
    if (::close(file.release()) != 0 && err.empty()) err = io_error("close failed");
    agree_or_throw(comm, path, err);  // no rank returns before the file is complete
}

}  // namespace stPS
