#include "data/synthetic.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_set>

namespace stPS {

namespace {

// SplitMix64-based deterministic pool of N distinct uint64 hashes.
std::vector<Hash> build_hash_pool(std::uint64_t N, std::uint64_t seed) {
    std::vector<Hash> pool(N);
    std::uint64_t z = seed * 0x9E3779B97F4A7C15ULL + 0xDEADBEEFCAFEBABEULL;
    for (std::uint64_t i = 0; i < N; ++i) {
        z += 0x9E3779B97F4A7C15ULL;
        std::uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        x =  x ^ (x >> 31);
        pool[i] = x;
    }
    return pool;
}

}  // namespace

std::vector<std::vector<Hash>> generate_synthetic(const SyntheticParams& p) {
    std::vector<std::vector<Hash>> patches(p.M);
    if (p.N == 0 || p.M == 0) return patches;

    const std::vector<Hash> pool = build_hash_pool(p.N, p.seed);

    // Overlap model: every patch samples from the same "shared region" of the pool.
    // region_size shrinks linearly with overlap, bounded below by K_mean so a single
    // patch can still fit.
    //   overlap=0 → region = N (uniform; patches independent)
    //   overlap=1 → region = K_mean (all patches draw from the same K elements)
    const double overlap = std::clamp(p.overlap, 0.0, 1.0);
    const std::uint64_t k_floor = std::max<std::uint64_t>(1, p.K_mean);
    const double region_d = static_cast<double>(p.N) * (1.0 - overlap)
                          + static_cast<double>(k_floor) * overlap;
    const std::uint64_t region_size = std::min<std::uint64_t>(
        p.N, std::max<std::uint64_t>(k_floor, static_cast<std::uint64_t>(region_d)));

    std::mt19937_64 rng(p.seed);
    std::uniform_int_distribution<std::uint64_t> idx_dist(0, region_size - 1);

    // K varies ±25% around K_mean for some realism, capped by region_size.
    const std::int64_t k_min = std::max<std::int64_t>(1, p.K_mean - p.K_mean / 4);
    const std::int64_t k_max_pre = static_cast<std::int64_t>(p.K_mean) + p.K_mean / 4;
    const std::int64_t k_cap = static_cast<std::int64_t>(region_size);
    const std::int64_t k_max = std::max<std::int64_t>(k_min, std::min<std::int64_t>(k_max_pre, k_cap));
    std::uniform_int_distribution<std::int64_t> k_dist(k_min, k_max);

    for (std::uint64_t pi = 0; pi < p.M; ++pi) {
        const std::int64_t k = k_dist(rng);
        std::unordered_set<std::uint64_t> picked;
        picked.reserve(static_cast<std::size_t>(k * 2));
        while (static_cast<std::int64_t>(picked.size()) < k) {
            picked.insert(idx_dist(rng));
        }
        auto& out = patches[pi];
        out.reserve(picked.size());
        for (std::uint64_t idx : picked) out.push_back(pool[idx]);
    }

    return patches;
}

std::vector<std::vector<Point>> generate_synthetic_coords(
    const SyntheticParams& p, const std::vector<std::vector<Hash>>& patches) {
    // Independent RNG stream (seed offset) so coordinates don't perturb — and
    // aren't perturbed by — the hash sampling above. Uniform over a synthetic
    // 1mm × 1mm die in nm units; occurrences of the same hash land at
    // different spots, exercising the representative-location (lexicographic
    // min) reduction.
    std::mt19937_64 rng(p.seed ^ 0xC0A2D5EEDULL);  // "coord seed" stream offset
    std::uniform_real_distribution<double> pos(0.0, 1.0e6);

    std::vector<std::vector<Point>> coords(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) {
        coords[pi].resize(patches[pi].size());
        for (auto& pt : coords[pi]) pt = Point{pos(rng), pos(rng)};
    }
    return coords;
}

}  // namespace stPS
