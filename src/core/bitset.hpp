#pragma once

#include "core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace fullchipusc {

class DenseBitset {
public:
    DenseBitset() = default;
    explicit DenseBitset(std::uint64_t num_bits);

    void resize(std::uint64_t num_bits);
    void clear() noexcept;

    std::uint64_t num_bits()  const noexcept { return num_bits_; }
    std::uint64_t num_words() const noexcept { return words_.size(); }

    void set(ElementId id) noexcept {
        words_[id >> 6] |= (std::uint64_t{1} << (id & 63));
    }
    bool test(ElementId id) const noexcept {
        return (words_[id >> 6] >> (id & 63)) & 1ULL;
    }

    void or_inplace(const DenseBitset& other) noexcept;

    std::uint64_t popcount() const noexcept;

    // popcount(this & other) — used to compute |patch ∩ newly_covered|
    // when both sides are dense; rarely used by the hot path but handy for tests.
    std::uint64_t popcount_intersect(const DenseBitset& other) const noexcept;

    // Count how many IDs in `ids` have their bit set in this bitset.
    // Hot path: |patch ∩ newly_covered| with patch represented as a sparse ID span.
    std::uint64_t popcount_intersect_with_ids(std::span<const ElementId> ids) const noexcept;

    // Iterate the set bits of `this` and emit their IDs into `out`.
    // Used to convert newly_covered to a sparse ID list for broadcast.
    void collect_set_ids(std::vector<ElementId>& out) const;

    const std::uint64_t* data() const noexcept { return words_.data(); }
    std::uint64_t*       data()       noexcept { return words_.data(); }

private:
    std::uint64_t num_bits_ = 0;
    std::vector<std::uint64_t> words_;
};

}  // namespace fullchipusc
