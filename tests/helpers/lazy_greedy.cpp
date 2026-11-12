#include "helpers/lazy_greedy.hpp"

#include "core/bitset.hpp"

#include <queue>
#include <utility>
#include <vector>

namespace stPS::test_helpers {

namespace {

struct Entry {
    std::int64_t score;
    PatchId      patch;
};

// priority_queue keeps the "largest" on top: higher score first, then the
// smaller PatchId.
struct WorseThan {
    bool operator()(const Entry& a, const Entry& b) const {
        return a.score != b.score ? a.score < b.score : a.patch > b.patch;
    }
};

}  // namespace

PatchSelection lazy_greedy_select(const PatchCsr& patches, std::uint64_t N) {
    PatchSelection result;
    result.covered_count = 0;
    result.iterations    = 0;

    std::vector<Entry> initial;
    initial.reserve(patches.M());
    for (PatchId p = 0; p < patches.M(); ++p) {
        const auto size = static_cast<std::int64_t>(patches.patch(p).size());
        if (size > 0) initial.push_back(Entry{size, p});
    }
    std::priority_queue<Entry, std::vector<Entry>, WorseThan> heap(
        WorseThan{}, std::move(initial));

    DenseBitset covered(N);
    while (result.covered_count < N && !heap.empty()) {
        const Entry top = heap.top();
        heap.pop();

        std::int64_t score = 0;
        for (ElementId e : patches.patch(top.patch)) {
            if (!covered.test(e)) ++score;
        }
        if (score == 0) continue;
        if (score != top.score) {  // stale: re-queue with the true score
            heap.push(Entry{score, top.patch});
            continue;
        }

        for (ElementId e : patches.patch(top.patch)) {
            if (!covered.test(e)) {
                covered.set(e);
                ++result.covered_count;
            }
        }
        result.selected.push_back(top.patch);
        ++result.iterations;
    }
    return result;
}

}  // namespace stPS::test_helpers
