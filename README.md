# stPatternSearch

Distributed greedy minimum set cover for OPC segment dedup (USC: Unique Segment Correction).

Splits a semiconductor layout's full chip into patches, where each patch's segments carry hash values
based on local pattern context. This solver finds the **minimum number of patches whose union covers
every unique hash** — a classical Minimum Set Cover problem solved with the greedy approximation.

Ships as the **`stPS`** (stPatternSearch) library: one header, one
`UscPatchSelector` class — see [Use as a library](#use-as-a-library).

Design rationale lives in [`greedy_set_cover.md`](./greedy_set_cover.md).

## Status

| Milestone | Title | State |
|---|---|---|
| M1 | Single-rank end-to-end | ✅ |
| M2 | Multi-rank MPI via stComm | ✅ |
| M3 | Distributed sample sort setup (§5.1) + sparse bcast (§7.2) | ✅ |
| M4 | GPU port (CUDA kernels, CUB ArgMax) | ✅ |
| M5 | NCCL device-direct broadcast | ✅ |
| M5.5 | score_update strategy A (inverted-index, affected-only) | ✅ |
| M6 | Element-partitioned covered bitset (§7.3) — `PartitionMode::ByElement` | ✅ opt-in |
| M7 | Real OPC input parser + 2D partition (§7.4) | ⏳ Planned |

Branches:
- `main` — CPU-only baseline (through M3). Stable regression reference.
- `gpu` — GPU work (M4+). M5+ continues here.

## Documentation

- **[STATUS.md](STATUS.md)** — what's been built, architecture summary, verification baseline
- **[ROADMAP.md](ROADMAP.md)** — M5+ planned work, new-environment setup, open decisions
- **[CONVENTIONS.md](CONVENTIONS.md)** — collaboration patterns and design preferences

## Build & install

```bash
# One-time: install build prereqs (cmake>=3.25, OpenMPI, NCCL, GTest, stComm).
# CUDA Toolkit + driver must be pre-installed; everything else is handled.
./scripts/setup-env.sh

# Build (stComm is a dependency; point at its prefix if not in ~/install/stComm)
STCOMM_PREFIX=~/install/stComm ./build.sh        # or: ./build.sh --clean / --debug

# Install the stPS library + headers + CMake package
./install.sh ~/install/stPS                      # default prefix: /usr/local

# Tests (one MPI rank per visible GPU)
cd build && ctest --output-on-failure

# Smoke (the CLI driver, which uses the public stPS::UscPatchSelector under the hood)
mpirun -n 4 --oversubscribe ./build/src/usc-patch-select \
    --N 10000 --M 1000 --K 50 --overlap 0.4 --seed 42
# Expected: selected=353 covered=6019 iterations=353

# File input (M7): dump a patch set to the .stps binary format once, then feed
# it back — each rank reads only its own slice; results are bit-identical.
./build/src/usc-patch-select --N 10000 --M 1000 --K 50 --overlap 0.4 --seed 42 --dump patches.stps
mpirun -n 2 ./build/src/usc-patch-select --input patches.stps

# UPS hash statistics: top-K hashes by patch count, with each hash's
# representative (lexicographic-min) location, written to one text file by all
# ranks in parallel. --dump here emits .stps v2 (with per-occurrence coords).
mpirun -n 2 ./build/src/ups-hash-stats --N 10000 --M 1000 --K 50 \
    --output stats.txt --output-limit 100
```

`build.sh` configures + builds against stComm and prints the artifact paths;
the raw `cmake -S . -B build -DCMAKE_PREFIX_PATH=~/install/stComm` flow works too.

## Use as a library

stPS installs a CMake package, so consumers just `find_package` and link:

```cmake
find_package(stPS REQUIRED)        # also pulls in stComm, MPI, CUDA
target_link_libraries(your_target PRIVATE stPS::stPS)
```

The whole API is one umbrella header and one class. A device-enabled
`stComm::Comm` (from `stComm::Comm::onDevice`) drives the distributed solve:

```cpp
#include <stPS/stPS.h>
#include <stComm/stComm.h>

stComm::Comm::initialize(&argc, &argv);
{
    stComm::Comm comm = stComm::Comm::onDevice(/*device_id=*/rank % num_gpus);

    stPS::UscPatchSelector selector(comm);

    // This rank's patches (each a list of element hashes) + their global IDs.
    // Use slice_patches_by_rank to partition a full list across ranks:
    auto slice = stPS::slice_patches_by_rank(std::move(all_patches), rank, size);

    // One collective call = load + distributed setup + greedy select.
    // Every rank must call together and gets the same result.
    stPS::PatchSelection r =
        selector.patch_select(std::move(slice.patches), std::move(slice.global_ids));

    // r.selected (chosen patch IDs), r.covered_count, r.iterations
}
stComm::Comm::finalize();
```

All algorithm internals (CSR, inverted index, device buffers, CUDA kernels) are
hidden behind a pImpl — the public headers (`include/stPS/`) pull in no
MPI/NCCL/CUDA.

Detailed build / environment instructions: [STATUS.md](STATUS.md) and [ROADMAP.md](ROADMAP.md).
