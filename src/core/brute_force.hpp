#pragma once

#include "core/csr.hpp"
#include "core/usc_solver.hpp"  // SolverResult
#include "core/types.hpp"

#include <cstdint>

namespace stPS {

// Reference greedy set-cover implementation.
//
// Each iteration recomputes |patch[p] \ covered| for ALL patches and picks the
// argmax (tie: smaller PatchId, matching MPI_MAXLOC semantics). No inverted index,
// no incremental score maintenance. O(S · M · K) overall — slow but obviously
// correct. Used as ground truth in equivalence tests against solve_greedy.
SolverResult solve_brute_force(const PatchCsr& patches, std::uint64_t N);

}  // namespace stPS
