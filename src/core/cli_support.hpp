#pragma once

// Process-level helpers shared by the CLI drivers (usc-patch-select,
// ups-pattern-stats). Compiled into the executables only, not into libstPS —
// device placement and fatal-error policy belong to the application, not the
// library.

#include "data/synthetic.hpp"

#include <string>

namespace stPS::cli {

// The synthetic-data options every driver accepts: --N --M --K --overlap
// --seed. Looks at argv[i]; if it is one of them, stores the value into
// `params` and advances `i` past it.
enum class OptionParse {
    NotMine,       // argv[i] is some other option — the caller handles it
    Consumed,      // parsed; i now points at the option's value
    MissingValue,  // the option is the last argument, with no value after it
};
OptionParse parse_synthetic_option(int argc, char** argv, int& i, SyntheticParams& params);

// Route every uncaught exception straight to MPI_Abort. With no matching
// handler, the C++ runtime calls std::terminate *without* unwinding the stack,
// so CUDA buffers and the NCCL communicator are never destroyed on the way
// out — their destructors (cudaFree, ncclCommDestroy) can block forever on a
// collective a failed peer will never finish. Call right after MPI init, and
// don't wrap the work in a catch-all (that would unwind first).
void install_mpi_abort_on_uncaught();

// Pick this process's GPU from its node-local rank (ranks sharing a node via
// MPI_COMM_TYPE_SHARED), so placement is right for any launcher mapping
// (--map-by node, uneven ranks per node). Modulo the visible count keeps
// per-process masking (CUDA_VISIBLE_DEVICES = one GPU per rank) working.
// Sets the device and returns its id for Comm::onDevice; aborts the job if no
// GPU is visible or the device can't be set.
int pick_device_for_local_rank(const char* app_name);

// Whether two output paths name the same file: device + inode when both
// already exist (catches hard links), otherwise lexical + symlink
// normalization (the files need not exist yet), so "out.bin", "./out.bin"
// and "dir/../out.bin" all match.
// Lets a driver reject two outputs aimed at one file, where the second writer
// would silently overwrite the first.
bool same_output_path(const std::string& a, const std::string& b);

}  // namespace stPS::cli
