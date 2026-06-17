#pragma once

/**
 * @file stPS.h
 * @brief Umbrella header for stPS (stPatternSearch) — distributed greedy
 *        minimum set-cover over stComm.
 *
 * Include this to use the library:
 *   - Solver / SolverResult  (the single solve entry point)
 *   - slice_patches_by_rank  (partition patches across ranks)
 *   - the public scalar types (Hash, PatchId, ...)
 *
 * A device-enabled stComm::Comm (stComm::Comm::onDevice) is required; include
 * <stComm/stComm.h> to build one.
 */

#include <stPS/types.hpp>
#include <stPS/partition.hpp>
#include <stPS/solver.hpp>
