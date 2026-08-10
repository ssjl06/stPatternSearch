// Tests for UPS pattern statistics (public stPS::UpsPatternStats): global
// per-hash patch counts + representative (lexicographic-min) locations, top-K
// selection, and the fixed-width parallel stats file.
//
// The distributed results are checked against a single-process host reference
// computed over the full (deterministic) synthetic set — every rank builds the
// same reference, so every rank can assert the collective result in full.

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

// Output ordering (count desc, hash asc) — mirrors ups_pattern_stats.cpp.
bool stat_before(const PatternStat& a, const PatternStat& b) {
    return a.count != b.count ? a.count > b.count : a.hash < b.hash;
}

// Single-process reference: per hash, the number of patches containing it
// (per-patch dedupe) and the lexicographic-min occurrence location.
std::vector<PatternStat> host_reference(const SyntheticData& d) {
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
    std::vector<PatternStat> out;
    out.reserve(counts.size());
    for (const auto& [h, c] : counts) out.push_back(PatternStat{h, c, rep.at(h)});
    std::sort(out.begin(), out.end(), stat_before);
    return out;
}

// Run the public collective on this rank's slice of the shared synthetic set.
std::vector<PatternStat> distributed_stats(stComm::Comm& comm,
                                           const SyntheticData& d,
                                           std::uint64_t k) {
    auto slice = slice_patches_by_rank(d.patches, d.coords,
                                       comm.getRank(), comm.getSize());
    UpsPatternStats ups(comm);
    return ups.pattern_stats(std::move(slice.patches), std::move(slice.coords), k);
}

void expect_stats_eq(const std::vector<PatternStat>& got,
                     const std::vector<PatternStat>& want) {
    ASSERT_EQ(got.size(), want.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_EQ(got[i].hash,  want[i].hash)  << "entry " << i;
        EXPECT_EQ(got[i].count, want[i].count) << "entry " << i;
        EXPECT_EQ(got[i].rep,   want[i].rep)   << "entry " << i;
    }
}

}  // namespace

TEST(PatternStats, HandcraftedMinLocationAndCounts) {
    // Duplicated hash across (and within) patches: count is per-patch deduped,
    // the representative keeps the smallest (x, then y) over all occurrences.
    auto& comm = test_helpers::comm();
    SyntheticData d;
    d.patches = {{7, 3, 7}, {3, 9}};
    d.coords  = {{{2, 2}, {5, 5}, {2, 1}}, {{4, 4}, {0, 9}}};

    const auto got = distributed_stats(comm, d, 10);
    const std::vector<PatternStat> want = {
        {3, 2, Point{4, 4}},   // in both patches; min of (5,5),(4,4)
        {7, 1, Point{2, 1}},   // twice in patch 0; (2,1) < (2,2)
        {9, 1, Point{0, 9}},
    };
    expect_stats_eq(got, want);
}

TEST(PatternStats, TopKMatchesHostReference) {
    auto& comm = test_helpers::comm();
    const auto data = small_synthetic();
    const auto ref  = host_reference(data);

    const std::uint64_t k = 17;
    const auto topk = distributed_stats(comm, data, k);
    ASSERT_EQ(topk.size(), std::min<std::size_t>(k, ref.size()));
    expect_stats_eq(topk, {ref.begin(), ref.begin() + topk.size()});
}

TEST(PatternStats, KPastUniqueCountReturnsAll) {
    auto& comm = test_helpers::comm();
    const auto data = small_synthetic(43);
    const auto ref  = host_reference(data);

    const auto all = distributed_stats(comm, data, ref.size() * 10);
    expect_stats_eq(all, ref);

    const auto none = distributed_stats(comm, data, 0);
    EXPECT_TRUE(none.empty());
}

TEST(PatternStats, RejectsShapeMismatch) {
    auto& comm = test_helpers::comm();
    std::vector<std::vector<Hash>>  patches = {{1, 2}};
    std::vector<std::vector<Point>> coords  = {{{0, 0}}};  // one coord short
    UpsPatternStats ups(comm);
    EXPECT_THROW(ups.pattern_stats(std::move(patches), std::move(coords), 5),
                 std::invalid_argument);
}

TEST(PatternStats, StatsFileRoundTrip) {
    auto& comm = test_helpers::comm();
    const auto data = small_synthetic(47);
    const auto topk = distributed_stats(comm, data, 23);

    // One shared file, all ranks write their slice in parallel. Pre-seed a
    // longer garbage file to prove a stale run gets fully overwritten.
    const std::string path = ::testing::TempDir() + "ups_stats_shared.txt";
    if (comm.getRank() == 0) {
        std::ofstream junk(path, std::ios::trunc);
        for (int i = 0; i < 1000; ++i) junk << "stale stale stale\n";
    }
    comm.barrier();
    write_pattern_stats_file(comm, path, topk);   // barriers internally before returning

    // Every rank parses the whole file back.
    std::ifstream in(path);
    ASSERT_TRUE(in.good());
    std::string header;
    ASSERT_TRUE(std::getline(in, header));
    EXPECT_EQ(header.rfind("# ups-pattern-stats", 0), 0u) << header;

    std::vector<PatternStat> parsed;
    std::string line;
    while (std::getline(in, line)) {
        PatternStat st;
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
