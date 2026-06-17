#pragma once

/**
 * @file stPS.h
 * @brief Umbrella header for stPS (stPatternSearch) — distributed greedy
 *        minimum set-cover over stComm.
 *
 * Include this to use the library:
 *   - UscPatchSelector / PatchSelection  (the single patch-select entry point)
 *   - slice_patches_by_rank  (partition patches across ranks)
 *   - the public scalar types (Hash, PatchId, ...)
 *
 * A device-enabled stComm::Comm (stComm::Comm::onDevice) is required; include
 * <stComm/stComm.h> to build one.
 */

#include <stPS/types.hpp>
#include <stPS/partition.hpp>
#include <stPS/usc_patch_selector.hpp>
