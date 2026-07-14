#include "ups/hash_stats.hpp"

#include <stComm/stComm.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace stPS {

namespace {

// Ordering for statistics output: most patches first, then smaller hash —
// deterministic across ranks so every rank derives the identical top-k list.
bool stat_before(const HashStat& a, const HashStat& b) {
    return a.count != b.count ? a.count > b.count : a.hash < b.hash;
}

[[noreturn]] void fail_io(const std::string& path, const std::string& what) {
    throw std::runtime_error("ups-stats: " + path + ": " + what +
                             ": " + std::strerror(errno));
}

}  // namespace

std::vector<std::pair<Hash, Point>> local_min_locations(
    const std::vector<std::vector<Hash>>& patches,
    const std::vector<std::vector<Point>>& coords) {
    assert(coords.size() == patches.size());

    std::unordered_map<Hash, Point> best;
    for (std::size_t p = 0; p < patches.size(); ++p) {
        assert(coords[p].size() == patches[p].size());
        for (std::size_t i = 0; i < patches[p].size(); ++i) {
            auto [it, inserted] = best.emplace(patches[p][i], coords[p][i]);
            if (!inserted && coords[p][i] < it->second) it->second = coords[p][i];
        }
    }

    std::vector<std::pair<Hash, Point>> out(best.begin(), best.end());
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    return out;
}

std::vector<HashStat> global_top_k(stComm::Comm& comm, const PatchSet& ps,
                                   const std::vector<std::pair<Hash, Point>>& minloc,
                                   std::uint64_t k) {
    const int size = comm.getSize();
    const int rank = comm.getRank();
    const InvertedIndex& inv = ps.inverted_index();

    // The §5.1 mapping assigns IDs in global hash order, so this rank's sorted
    // unique hashes (minloc) and its sorted unique ElementIds (inv.keys) are
    // the same set in the same order — pair them positionally.
    const std::size_t n_local = inv.keys.size();
    if (minloc.size() != n_local) {
        throw std::invalid_argument(
            "global_top_k: minloc/inverted-index size mismatch (" +
            std::to_string(minloc.size()) + " vs " + std::to_string(n_local) +
            ") — was local_min_locations run on the same patches?");
    }

    // Owner boundaries: rank r owns IDs [starts[r], starts[r+1]).
    std::vector<std::uint64_t> starts(static_cast<std::size_t>(size) + 1);
    {
        const std::uint64_t my_start = ps.shard_start();
        std::vector<int> ones(static_cast<std::size_t>(size), 1);
        comm.allgatherv<stComm::Space::Host, std::uint64_t>(&my_start, 1,
                                                starts.data(), ones.data())->wait();
        starts[static_cast<std::size_t>(size)] = ps.N();
    }

    // Route each local (id, degree, minloc) to the id's owner. inv.keys is
    // sorted, so each owner's entries form one contiguous run.
    std::vector<int> sendcounts(static_cast<std::size_t>(size), 0);
    for (std::size_t i = 0; i < n_local; ++i) {
        const int owner = static_cast<int>(
            std::upper_bound(starts.begin() + 1, starts.begin() + size, inv.keys[i]) -
            (starts.begin() + 1));
        ++sendcounts[owner];
    }

    std::vector<std::uint64_t> send_ids(n_local), send_deg(n_local);
    std::vector<double>        send_x(n_local),   send_y(n_local);
    for (std::size_t i = 0; i < n_local; ++i) {
        send_ids[i] = inv.keys[i];
        send_deg[i] = inv.offsets[i + 1] - inv.offsets[i];
        send_x[i]   = minloc[i].second.x;
        send_y[i]   = minloc[i].second.y;
    }

    // Exchange counts (allgatherv of each rank's sendcounts row, as in
    // patch_set.cpp), then ship the four arrays with the same counts.
    std::vector<int> recvcounts(static_cast<std::size_t>(size), 0);
    {
        std::vector<int> all(static_cast<std::size_t>(size) * size);
        std::vector<int> uniform(static_cast<std::size_t>(size), size);
        comm.allgatherv<stComm::Space::Host, int>(sendcounts.data(), size,
                                       all.data(), uniform.data())->wait();
        for (int s = 0; s < size; ++s) recvcounts[s] = all[s * size + rank];
    }
    const std::size_t total_recv = static_cast<std::size_t>(
        std::accumulate(recvcounts.begin(), recvcounts.end(), 0));

    std::vector<std::uint64_t> recv_ids(total_recv), recv_deg(total_recv);
    std::vector<double>        recv_x(total_recv),   recv_y(total_recv);
    comm.alltoallv<stComm::Space::Host, std::uint64_t>(send_ids.data(), sendcounts.data(),
                                            recv_ids.data(), recvcounts.data())->wait();
    comm.alltoallv<stComm::Space::Host, std::uint64_t>(send_deg.data(), sendcounts.data(),
                                            recv_deg.data(), recvcounts.data())->wait();
    comm.alltoallv<stComm::Space::Host, double>(send_x.data(), sendcounts.data(),
                                     recv_x.data(), recvcounts.data())->wait();
    comm.alltoallv<stComm::Space::Host, double>(send_y.data(), sendcounts.data(),
                                     recv_y.data(), recvcounts.data())->wait();

    // Owner-side reduction over this rank's shard: sum the patch counts, keep
    // the lexicographic-min location. shard_hashes maps slot → original hash.
    const std::vector<Hash>& shard = ps.shard_hashes();
    const std::uint64_t shard_start = ps.shard_start();
    std::vector<HashStat> stats(shard.size());
    for (std::size_t i = 0; i < shard.size(); ++i) {
        stats[i].hash = shard[i];
        stats[i].rep  = Point{0.0, 0.0};
    }
    std::vector<bool> seen(shard.size(), false);
    for (std::size_t j = 0; j < total_recv; ++j) {
        const std::size_t slot = static_cast<std::size_t>(recv_ids[j] - shard_start);
        assert(slot < shard.size());
        stats[slot].count += recv_deg[j];
        const Point pt{recv_x[j], recv_y[j]};
        if (!seen[slot] || pt < stats[slot].rep) { stats[slot].rep = pt; seen[slot] = true; }
    }

    // Local top-k of this shard. Any global top-k entry is in its owner's
    // local top-k, so gathering the per-rank candidates is exact.
    const std::size_t k_local = std::min<std::size_t>(k, stats.size());
    std::partial_sort(stats.begin(), stats.begin() + k_local, stats.end(), stat_before);
    stats.resize(k_local);

    // Gather every rank's candidates (variable counts) and reduce identically
    // everywhere. All ranks hold size×k candidates here — fine for a stats
    // tool; revisit with a k-selection exchange if k ever gets huge.
    const int n_cand = static_cast<int>(stats.size());
    std::vector<int> cand_counts(static_cast<std::size_t>(size));
    {
        std::vector<int> ones(static_cast<std::size_t>(size), 1);
        comm.allgatherv<stComm::Space::Host, int>(&n_cand, 1,
                                       cand_counts.data(), ones.data())->wait();
    }
    const std::size_t total_cand = static_cast<std::size_t>(
        std::accumulate(cand_counts.begin(), cand_counts.end(), 0));

    std::vector<Hash>          c_hash(stats.size());
    std::vector<std::uint64_t> c_cnt(stats.size());
    std::vector<double>        c_x(stats.size()), c_y(stats.size());
    for (std::size_t i = 0; i < stats.size(); ++i) {
        c_hash[i] = stats[i].hash;  c_cnt[i] = stats[i].count;
        c_x[i]    = stats[i].rep.x; c_y[i]   = stats[i].rep.y;
    }
    std::vector<Hash>          all_hash(total_cand);
    std::vector<std::uint64_t> all_cnt(total_cand);
    std::vector<double>        all_x(total_cand), all_y(total_cand);
    comm.allgatherv<stComm::Space::Host, Hash>(c_hash.data(), n_cand,
                                   all_hash.data(), cand_counts.data())->wait();
    comm.allgatherv<stComm::Space::Host, std::uint64_t>(c_cnt.data(), n_cand,
                                   all_cnt.data(), cand_counts.data())->wait();
    comm.allgatherv<stComm::Space::Host, double>(c_x.data(), n_cand,
                                   all_x.data(), cand_counts.data())->wait();
    comm.allgatherv<stComm::Space::Host, double>(c_y.data(), n_cand,
                                   all_y.data(), cand_counts.data())->wait();

    std::vector<HashStat> merged(total_cand);
    for (std::size_t i = 0; i < total_cand; ++i) {
        merged[i] = HashStat{all_hash[i], all_cnt[i], Point{all_x[i], all_y[i]}};
    }
    const std::size_t k_global = std::min<std::size_t>(k, merged.size());
    std::partial_sort(merged.begin(), merged.begin() + k_global, merged.end(), stat_before);
    merged.resize(k_global);
    return merged;
}

void write_stats_file(stComm::Comm& comm, const std::string& path,
                      const std::vector<HashStat>& topk) {
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
        "# ups-hash-stats k=%zu ranks=%d (hash, patch count, rep x, rep y)\n",
        topk.size(), size);
    if (header_len <= 0) throw std::runtime_error("ups-stats: header format failed");

    // This rank's contiguous line range (same split rule as patches-by-rank).
    const std::uint64_t k = topk.size();
    const std::uint64_t begin = (k * static_cast<std::uint64_t>(rank))     / size;
    const std::uint64_t end   = (k * (static_cast<std::uint64_t>(rank)+1)) / size;

    std::string buf;
    buf.reserve((end - begin) * kLineBytes + (rank == 0 ? header_len : 0));
    if (rank == 0) buf.append(header, static_cast<std::size_t>(header_len));
    for (std::uint64_t i = begin; i < end; ++i) {
        char line[kLineBytes + 1];
        const int len = std::snprintf(line, sizeof(line), kLineFmt,
                                      static_cast<unsigned long>(topk[i].hash),
                                      static_cast<unsigned long>(topk[i].count),
                                      topk[i].rep.x, topk[i].rep.y);
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
