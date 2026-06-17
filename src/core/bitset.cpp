#include "core/bitset.hpp"

#include <bit>
#include <cassert>
#include <cstring>

namespace stPS {

namespace {
constexpr std::uint64_t words_for(std::uint64_t num_bits) noexcept {
    return (num_bits + 63) / 64;
}
}  // namespace

DenseBitset::DenseBitset(std::uint64_t num_bits)
    : num_bits_(num_bits), words_(words_for(num_bits), 0) {}

void DenseBitset::resize(std::uint64_t num_bits) {
    num_bits_ = num_bits;
    words_.assign(words_for(num_bits), 0);
}

void DenseBitset::clear() noexcept {
    if (!words_.empty()) {
        std::memset(words_.data(), 0, words_.size() * sizeof(std::uint64_t));
    }
}

void DenseBitset::or_inplace(const DenseBitset& other) noexcept {
    assert(num_bits_ == other.num_bits_);
    const std::size_t n = words_.size();
    for (std::size_t i = 0; i < n; ++i) {
        words_[i] |= other.words_[i];
    }
}

std::uint64_t DenseBitset::popcount() const noexcept {
    std::uint64_t total = 0;
    for (std::uint64_t w : words_) {
        total += static_cast<std::uint64_t>(std::popcount(w));
    }
    return total;
}

std::uint64_t DenseBitset::popcount_intersect(const DenseBitset& other) const noexcept {
    assert(num_bits_ == other.num_bits_);
    std::uint64_t total = 0;
    const std::size_t n = words_.size();
    for (std::size_t i = 0; i < n; ++i) {
        total += static_cast<std::uint64_t>(std::popcount(words_[i] & other.words_[i]));
    }
    return total;
}

std::uint64_t DenseBitset::popcount_intersect_with_ids(std::span<const ElementId> ids) const noexcept {
    std::uint64_t total = 0;
    for (ElementId id : ids) {
        total += (words_[id >> 6] >> (id & 63)) & 1ULL;
    }
    return total;
}

void DenseBitset::collect_set_ids(std::vector<ElementId>& out) const {
    out.clear();
    out.reserve(static_cast<std::size_t>(popcount()));
    const std::uint64_t nw = words_.size();
    for (std::uint64_t wi = 0; wi < nw; ++wi) {
        std::uint64_t w = words_[wi];
        const ElementId base = static_cast<ElementId>(wi) * 64;
        while (w != 0) {
            const int bit = std::countr_zero(w);
            out.push_back(base + static_cast<ElementId>(bit));
            w &= (w - 1);  // clear lowest set bit
        }
    }
}

}  // namespace stPS
