#pragma once

#include <stPS/partition.hpp>   // PatchSlice
#include <stPS/types.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace stPS {

// Interface for reading patches from a file (ROADMAP M7). A reader is opened
// once per rank; setup then asks for a cheap header probe (patch_count) and a
// single contiguous slice read — the same contiguous split rule as
// slice_patches_by_rank, so file input and synthetic input hand the solver
// identically-shaped slices. Global PatchId of patch i = its file-order index.
//
// Concrete formats implement this interface; open_patch_file() sniffs the file
// and dispatches. Today there is one format (.stps binary CSR, stps_binary.cpp);
// the real OPC reader plugs in here once the OPC team's format spec lands.
class PatchReader {
public:
    virtual ~PatchReader() = default;

    // Total patch count M (header only — no bulk I/O).
    virtual std::uint64_t patch_count() = 0;

    // Read patches [begin, end) plus their global IDs. Not collective — each
    // rank calls independently with its own range. Throws std::runtime_error
    // on I/O errors or a corrupt file.
    virtual PatchSlice read_slice(std::uint64_t begin, std::uint64_t end) = 0;
};

// Open a patch file, sniffing the format by its magic. Throws
// std::runtime_error if the file is missing, unreadable, or not a known format.
std::unique_ptr<PatchReader> open_patch_file(const std::string& path);

// Write the v1 .stps binary CSR file (format documented in stps_binary.cpp).
// Overwrites `path`. Throws std::runtime_error on I/O failure.
void write_patch_file(const std::string& path,
                      const std::vector<std::vector<Hash>>& patches);

// v2: same, plus per-occurrence coordinates (coords must mirror `patches`
// shape; read_slice hands them back in PatchSlice::coords).
void write_patch_file(const std::string& path,
                      const std::vector<std::vector<Hash>>& patches,
                      const std::vector<std::vector<Point>>& coords);

}  // namespace stPS
