#include "io/patch_reader.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// stPS binary patch file (.stps) — CSR on disk, little-endian:
//
//   header (32 B):
//     magic     8 B   "STPSPAT1" (v1) or "STPSPAT2" (v2)
//     M         8 B   uint64 patch count
//     total_K   8 B   uint64 total hash count (== offsets[M])
//     reserved  8 B   0 (future flags/version bumps)
//   offsets     (M+1) × 8 B uint64   patch p's hashes = hashes[offsets[p] .. offsets[p+1])
//   hashes      total_K × 8 B uint64 raw hash values, patches concatenated in order
//   coords      total_K × 16 B (x,y double)   v2 only: coords[j] is where
//                                             occurrence hashes[j] sits (same
//                                             CSR offsets index both sections)
//
// The layout deliberately mirrors the in-memory PatchCsr (design doc §3): a
// rank reading patches [begin, end) seeks to its offsets window, then to its
// hash window (and, for v2, its coords window) — contiguous reads only, no
// scan of the rest of the file. That keeps multi-rank setup I/O proportional
// to each rank's own share (M7 goal), with plain fseek/fread and no MPI-IO
// dependency.
//
// Hashes are stored exactly as produced (duplicates within a patch allowed);
// PatchSet's setup dedupes per patch, same as the synthetic path. v1 files
// carry no locations — read_slice returns empty coords and location-consuming
// callers (UPS) reject the input with a clear error.

namespace stPS {

namespace {

constexpr char        kMagicV1[8]  = {'S','T','P','S','P','A','T','1'};
constexpr char        kMagicV2[8]  = {'S','T','P','S','P','A','T','2'};
constexpr std::size_t kHeaderBytes = 32;

// RAII FILE* so every throw path closes the handle.
struct FileCloser { void operator()(std::FILE* f) const { if (f) std::fclose(f); } };
using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

[[noreturn]] void fail(const std::string& path, const std::string& what) {
    throw std::runtime_error("stps: " + path + ": " + what);
}

void seek_to(std::FILE* f, std::uint64_t pos, const std::string& path) {
    if (fseeko(f, static_cast<off_t>(pos), SEEK_SET) != 0) {
        fail(path, "seek failed: " + std::string(std::strerror(errno)));
    }
}

void read_exact(std::FILE* f, void* dst, std::size_t bytes, const std::string& path) {
    if (bytes && std::fread(dst, 1, bytes, f) != bytes) {
        fail(path, "truncated file (unexpected EOF)");
    }
}

void write_exact(std::FILE* f, const void* src, std::size_t bytes, const std::string& path) {
    if (bytes && std::fwrite(src, 1, bytes, f) != bytes) {
        fail(path, "write failed: " + std::string(std::strerror(errno)));
    }
}

class StpsBinaryReader final : public PatchReader {
public:
    StpsBinaryReader(FilePtr file, std::string path)
        : file_(std::move(file)), path_(std::move(path)) {
        char magic[8];
        std::uint64_t header[3];  // M, total_K, reserved
        seek_to(file_.get(), 0, path_);
        read_exact(file_.get(), magic, sizeof(magic), path_);
        if      (std::memcmp(magic, kMagicV1, sizeof(kMagicV1)) == 0) has_coords_ = false;
        else if (std::memcmp(magic, kMagicV2, sizeof(kMagicV2)) == 0) has_coords_ = true;
        else fail(path_, "bad magic (not a .stps patch file)");
        read_exact(file_.get(), header, sizeof(header), path_);
        M_       = header[0];
        total_K_ = header[1];

        // Cheap structural sanity: the declared sections must match the file
        // size exactly, so a truncated or padded file fails at open, not
        // mid-slice-read.
        if (fseeko(file_.get(), 0, SEEK_END) != 0) fail(path_, "seek failed");
        const std::uint64_t file_bytes = static_cast<std::uint64_t>(ftello(file_.get()));
        // Bound M and total_K by the file size before multiplying: every
        // offset and hash takes 8 B, so a header claiming more than fits is
        // corrupt — and rejecting it here keeps the sum below from wrapping.
        if (M_ >= file_bytes / 8 || total_K_ > file_bytes / 8) {
            fail(path_, "corrupt header (M=" + std::to_string(M_) + ", total_K=" +
                        std::to_string(total_K_) + " exceed file size " +
                        std::to_string(file_bytes) + ")");
        }
        const std::uint64_t want =
            kHeaderBytes + (M_ + 1) * 8 + total_K_ * 8 +
            (has_coords_ ? total_K_ * 16 : 0);
        if (file_bytes != want) {
            fail(path_, "size mismatch: header declares " + std::to_string(want) +
                        " bytes, file has " + std::to_string(file_bytes));
        }

        // CSR endpoints: read_slice checks monotonicity per window, but only
        // these two pin the offsets to the hash section — without them a file
        // with offsets[0] > 0 or offsets[M] < total_K silently drops hashes.
        std::uint64_t first = 0, last = 0;
        seek_to(file_.get(), kHeaderBytes, path_);
        read_exact(file_.get(), &first, sizeof(first), path_);
        seek_to(file_.get(), kHeaderBytes + M_ * 8, path_);
        read_exact(file_.get(), &last, sizeof(last), path_);
        if (first != 0 || last != total_K_) {
            fail(path_, "corrupt offsets (offsets[0]=" + std::to_string(first) +
                        ", offsets[M]=" + std::to_string(last) +
                        ", want 0 and total_K=" + std::to_string(total_K_) + ")");
        }
    }

    std::uint64_t patch_count() override { return M_; }
    bool has_coords() override { return has_coords_; }

    FlatPatches read_flat(std::uint64_t begin, std::uint64_t end) override {
        if (begin > end || end > M_) {
            fail(path_, "slice [" + std::to_string(begin) + ", " + std::to_string(end) +
                        ") out of range (M=" + std::to_string(M_) + ")");
        }
        const std::uint64_t n = end - begin;
        FlatPatches out;

        // This slice's offsets window: n+1 entries starting at offsets[begin].
        out.offsets.resize(n + 1);
        seek_to(file_.get(), kHeaderBytes + begin * 8, path_);
        read_exact(file_.get(), out.offsets.data(), (n + 1) * 8, path_);
        for (std::uint64_t i = 0; i < n; ++i) {
            if (out.offsets[i] > out.offsets[i + 1]) fail(path_, "corrupt offsets (not monotone)");
        }
        if (out.offsets[n] > total_K_) fail(path_, "corrupt offsets (past hash section)");
        const std::uint64_t first = out.offsets[0];
        const std::uint64_t count = out.offsets[n] - first;

        // The hash window, one contiguous read straight into its final place.
        const std::uint64_t hash_base = kHeaderBytes + (M_ + 1) * 8;
        out.hashes.resize(count);
        seek_to(file_.get(), hash_base + first * 8, path_);
        read_exact(file_.get(), out.hashes.data(), count * 8, path_);

        // v2: same window into the coords section (indexed by the same offsets).
        if (has_coords_) {
            const std::uint64_t coord_base = hash_base + total_K_ * 8;
            out.coords.resize(count);
            seek_to(file_.get(), coord_base + first * 16, path_);
            read_exact(file_.get(), out.coords.data(), count * 16, path_);
        }

        for (std::uint64_t& o : out.offsets) o -= first;
        out.global_ids.resize(n);
        for (std::uint64_t p = 0; p < n; ++p) out.global_ids[p] = begin + p;
        return out;
    }

private:
    FilePtr       file_;
    std::string   path_;
    std::uint64_t M_          = 0;
    std::uint64_t total_K_    = 0;
    bool          has_coords_ = false;
};

}  // namespace

PatchSlice PatchReader::read_slice(std::uint64_t begin, std::uint64_t end) {
    FlatPatches flat = read_flat(begin, end);
    const std::uint64_t n = flat.patch_count();
    const bool with_coords = has_coords();
    PatchSlice slice;
    slice.patches.reserve(n);
    if (with_coords) slice.coords.reserve(n);
    for (std::uint64_t p = 0; p < n; ++p) {
        const auto b = static_cast<std::ptrdiff_t>(flat.offsets[p]);
        const auto e = static_cast<std::ptrdiff_t>(flat.offsets[p + 1]);
        slice.patches.emplace_back(flat.hashes.begin() + b, flat.hashes.begin() + e);
        if (with_coords) {
            slice.coords.emplace_back(flat.coords.begin() + b, flat.coords.begin() + e);
        }
    }
    slice.global_ids = std::move(flat.global_ids);
    return slice;
}

FlatPatches read_rank_flat(PatchReader& reader, int rank, int size) {
    const RankRange mine = rank_range(reader.patch_count(), rank, size);
    return reader.read_flat(mine.begin, mine.end);
}

PatchSlice read_rank_slice(PatchReader& reader, int rank, int size) {
    const RankRange mine = rank_range(reader.patch_count(), rank, size);
    return reader.read_slice(mine.begin, mine.end);
}

std::unique_ptr<PatchReader> open_patch_file(const std::string& path) {
    FilePtr f(std::fopen(path.c_str(), "rb"));
    if (!f) fail(path, "cannot open: " + std::string(std::strerror(errno)));
    // Single format today — StpsBinaryReader validates the magic itself. When a
    // second format lands (real OPC / HDF5), sniff the magic here and dispatch.
    return std::make_unique<StpsBinaryReader>(std::move(f), path);
}

namespace {

// Shared v1/v2 writer body: `coords` null → v1, non-null → v2 (must mirror
// `patches` shape).
void write_patch_file_impl(const std::string& path,
                           const std::vector<std::vector<Hash>>& patches,
                           const std::vector<std::vector<Point>>* coords) {
    static_assert(sizeof(Point) == 16, "Point must serialize as two packed doubles");
    const std::uint64_t M = patches.size();
    std::vector<std::uint64_t> offsets(M + 1, 0);
    for (std::uint64_t p = 0; p < M; ++p) {
        offsets[p + 1] = offsets[p] + patches[p].size();
    }
    const std::uint64_t total_K = offsets[M];

    if (coords) {
        if (coords->size() != M) fail(path, "coords/patches shape mismatch");
        for (std::uint64_t p = 0; p < M; ++p) {
            if ((*coords)[p].size() != patches[p].size()) {
                fail(path, "coords/patches shape mismatch at patch " + std::to_string(p));
            }
        }
    }

    FilePtr f(std::fopen(path.c_str(), "wb"));
    if (!f) fail(path, "cannot create: " + std::string(std::strerror(errno)));

    write_exact(f.get(), coords ? kMagicV2 : kMagicV1, 8, path);
    const std::uint64_t header[3] = {M, total_K, 0};
    write_exact(f.get(), header, sizeof(header), path);
    write_exact(f.get(), offsets.data(), offsets.size() * 8, path);
    for (const auto& p : patches) {
        write_exact(f.get(), p.data(), p.size() * 8, path);
    }
    if (coords) {
        for (const auto& c : *coords) {
            write_exact(f.get(), c.data(), c.size() * 16, path);
        }
    }
    if (std::fflush(f.get()) != 0) fail(path, "flush failed");
}

}  // namespace

void write_patch_file(const std::string& path,
                      const std::vector<std::vector<Hash>>& patches) {
    write_patch_file_impl(path, patches, nullptr);
}

void write_patch_file(const std::string& path,
                      const std::vector<std::vector<Hash>>& patches,
                      const std::vector<std::vector<Point>>& coords) {
    write_patch_file_impl(path, patches, &coords);
}

}  // namespace stPS
