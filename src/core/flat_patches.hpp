#pragma once

#include <stPS/partition.hpp>

#include <string>

namespace stPS {

// Structural check of a FlatPatches' hashes / offsets / coords (not its
// global_ids — whether those are required is up to the entry point). Returns
// an empty string when well-formed, otherwise what is wrong.
std::string flat_patches_error(const FlatPatches& patches);

}  // namespace stPS
