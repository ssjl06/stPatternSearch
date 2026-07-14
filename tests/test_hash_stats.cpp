// Tests for UPS hash statistics (ups/hash_stats.hpp): global per-hash patch
// counts + representative (lexicographic-min) locations, top-K selection, and
// the fixed-width parallel stats file.
//
// The distributed results are checked against a single-process host reference
// computed over the full (deterministic) synthetic set — every rank builds the
// same reference, so every rank can assert the collective result in full.

#include "ups/hash_stats.hpp"
#include "core/patch_set.hpp"
#include "data/synthetic.hpp"
#include "test_nccl_env.hpp"

#include <stPS/stPS.h>
#include <stComm/stComm.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace stPS;

namespace {

struct SyntheticData {
    std::vector<std::vector<Hash>>  patches;
    std::vector<std::vector<Point>> coords;
};

SyntheticData small_synthetic(std::uint64_t seed = 41) {
    SyntheticParams p;
    p.N = 400; p.M = 60; p.K_mean = 15; p.overlap = 0.5; p.seed = seed;
    SyntheticData d;
    d.patches = generate_synthetic(p);
    d.coords  = generate_synthetic_coords(p, d.patches);
    return d;
}

// Output ordering (count desc, hash asc) — mirrors hash_stats.cpp.
bool stat_before(const HashStat& a, const HashStat& b) {
    return a.count != b.count ? a.count > b.count : a.hash < b.hash;
}

// Single-process reference: per hash, the number of patches containing it
// (per-patch dedupe) and the lexicographic-min occurrence location.
std::vector<HashStat> host_reference(const SyntheticData& d) {
    std::map<Hash, std::uint64_t> counts;
    std::map<Hash, Point>         rep;
    for (std::size_t p = 0; p < d.patches.size(); ++p) {
        for (Hash h : std::set<Hash>(d.patches[p].begin(), d.patches[p].end())) {
            ++counts[h];
        }
        for (std::size_t i = 0; i < d.patches[p].size(); ++i) {
            const Point pt = d.coords[p][i];
            auto [it, inserted] = rep.emplace(d.patches[p][i], pt);
            if (!inserted && pt < it->second) it->second = pt;
        }
    }
    std::vector<HashStat> out;
    out.reserve(counts.size());
    for (const auto& [h, c] : counts) out.push_back(HashStat{h, c, rep.at(h)});
    std::sort(out.begin(), out.end(), stat_before);
    return out;
}

// Build this rank's PatchSet + minloc from the shared synthetic set.
struct DistributedStats {
    std::vector<std::pair<Hash, Point>> minloc;
    PatchSet ps;
};

DistributedStats make_distributed(stComm::Comm& comm, const SyntheticData& d) {
    auto slice = slice_patches_by_rank(d.patches, d.coords,
                                       comm.getRank(), comm.getSize());
    auto minloc = local_min_locations(slice.patches, slice.coords);
    return DistributedStats{std::move(minloc),
                            PatchSet(comm, std::move(slice.patches))};
}

void expect_stats_eq(const std::vector<HashStat>& got,
                     const std::vector<HashStat>& want) {
    ASSERT_EQ(got.size(), want.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_EQ(got[i].hash,  want[i].hash)  << "entry " << i;
        EXPECT_EQ(got[i].count, want[i].count) << "entry " << i;
        EXPECT_EQ(got[i].rep,   want[i].rep)   << "entry " << i;
    }
}

}  // namespace

TEST(HashStats, LocalMinLocations) {
    // Duplicated hash across (and within) patches keeps the smallest (x,y).
    std::vector<std::vector<Hash>>  patches = {{7, 3, 7}, {3, 9}};
    std::vector<std::vector<Point>> coords  = {{{2, 2}, {5, 5}, {2, 1}}, {{4, 4}, {0, 9}}};
    const auto ml = local_min_locations(patches, coords);
    ASSERT_EQ(ml.size(), 3u);                       // hashes 3, 7, 9 sorted
    EXPECT_EQ(ml[0].first, 3u); EXPECT_EQ(ml[0].second, (Point{4, 4}));
    EXPECT_EQ(ml[1].first, 7u); EXPECT_EQ(ml[1].second, (Point{2, 1}));
    EXPECT_EQ(ml[2].first, 9u); EXPECT_EQ(ml[2].second, (Point{0, 9}));
}

TEST(HashStats, TopKMatchesHostReference) {
    auto& comm = test_helpers::comm();
    const auto data = small_synthetic();
    const auto ref  = host_reference(data);
    auto dist = make_distributed(comm, data);

    const std::uint64_t k = 17;
    const auto topk = global_top_k(comm, dist.ps, dist.minloc, k);
    ASSERT_EQ(topk.size(), std::min<std::size_t>(k, ref.size()));
    expect_stats_eq(topk, {ref.begin(), ref.begin() + topk.size()});
}

TEST(HashStats, KPastUniqueCountReturnsAll) {
    auto& comm = test_helpers::comm();
    const auto data = small_synthetic(43);
    const auto ref  = host_reference(data);
    auto dist = make_distributed(comm, data);

    const auto all = global_top_k(comm, dist.ps, dist.minloc, ref.size() * 10);
    expect_stats_eq(all, ref);

    const auto none = global_top_k(comm, dist.ps, dist.minloc, 0);
    EXPECT_TRUE(none.empty());
}

TEST(HashStats, StatsFileRoundTrip) {
    auto& comm = test_helpers::comm();
    const auto data = small_synthetic(47);
    auto dist = make_distributed(comm, data);
    const auto topk = global_top_k(comm, dist.ps, dist.minloc, 23);

    // One shared file, all ranks write their slice in parallel. Pre-seed a
    // longer garbage file to prove a stale run gets fully overwritten.
    const std::string path = ::testing::TempDir() + "ups_stats_shared.txt";
    if (comm.getRank() == 0) {
        std::ofstream junk(path, std::ios::trunc);
        for (int i = 0; i < 1000; ++i) junk << "stale stale stale\n";
    }
    comm.barrier();
    write_stats_file(comm, path, topk);   // barriers internally before returning

    // Every rank parses the whole file back.
    std::ifstream in(path);
    ASSERT_TRUE(in.good());
    std::string header;
    ASSERT_TRUE(std::getline(in, header));
    EXPECT_EQ(header.rfind("# ups-hash-stats", 0), 0u) << header;

    std::vector<HashStat> parsed;
    std::string line;
    while (std::getline(in, line)) {
        HashStat st;
        unsigned long h = 0, c = 0;
        ASSERT_EQ(std::sscanf(line.c_str(), "%lx %lu %le %le",
                              &h, &c, &st.rep.x, &st.rep.y), 4) << line;
        st.hash = h; st.count = c;
        parsed.push_back(st);
    }
    expect_stats_eq(parsed, topk);  // %.17e round-trips doubles exactly

    comm.barrier();  // no rank may still be reading when rank 0 unlinks
    if (comm.getRank() == 0) std::remove(path.c_str());
}
