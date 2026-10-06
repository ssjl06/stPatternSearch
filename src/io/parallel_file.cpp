#include "io/parallel_file.hpp"

#include <stComm/stComm.h>

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace stPS::io {

namespace {

// Message for a failed POSIX call — read errno right away, before anything
// else can clobber it.
std::string io_error(const char* what) {
    return std::string(what) + ": " + std::strerror(errno);
}

// Collective success check: every rank learns whether any rank's I/O failed,
// so all throw together. Doubles as the barrier between write phases.
void agree_or_throw(stComm::Comm& comm, const std::string& prefix,
                    const std::string& local_err) {
    const std::int32_t ok = local_err.empty() ? 1 : 0;
    std::int32_t all_ok = 0;
    comm.allreduce<stComm::Space::Host, std::int32_t>(
        &ok, &all_ok, 1, stComm::ReduceOp::Min)->wait();
    if (!all_ok) {
        throw std::runtime_error(prefix +
            (local_err.empty() ? std::string("I/O failed on another rank") : local_err));
    }
}

// Write all `bytes` at `offset`, retrying short writes. "" on success.
std::string pwrite_all(int fd, const void* data, std::size_t bytes, off_t offset) {
    const char* p = static_cast<const char*>(data);
    for (std::size_t done = 0; done < bytes;) {
        const ssize_t w = ::pwrite(fd, p + done, bytes - done,
                                   offset + static_cast<off_t>(done));
        if (w < 0) return io_error("pwrite failed");
        done += static_cast<std::size_t>(w);
    }
    return {};
}

// Closes on scope exit, so a throw from agree_or_throw never leaks the fd.
struct FdGuard {
    int fd = -1;
    ~FdGuard() { if (fd >= 0) ::close(fd); }
    int release() { const int f = fd; fd = -1; return f; }
};

}  // namespace

void write_shared_file(stComm::Comm& comm, const std::string& path, const char* label,
                       const void* data, std::size_t bytes, off_t offset,
                       off_t total_size, const void* head, std::size_t head_bytes) {
    const std::string prefix = std::string(label) + ": " + path + ": ";

    std::string err;
    FdGuard file;
    file.fd = ::open(path.c_str(), O_WRONLY | O_CREAT, 0644);
    if (file.fd < 0) err = io_error("cannot open");
    if (err.empty() && comm.getRank() == 0 && head_bytes > 0) {
        err = pwrite_all(file.fd, head, head_bytes, 0);
    }
    if (err.empty()) err = pwrite_all(file.fd, data, bytes, offset);
    agree_or_throw(comm, prefix, err);  // everyone has written

    if (comm.getRank() == 0 && ::ftruncate(file.fd, total_size) != 0) {
        err = io_error("ftruncate failed");
    }
    if (::close(file.release()) != 0 && err.empty()) err = io_error("close failed");
    agree_or_throw(comm, prefix, err);  // no rank returns before the file is complete
}

}  // namespace stPS::io
