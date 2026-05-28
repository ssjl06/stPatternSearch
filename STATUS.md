# fullchipUSC — Status (as of M4 completion, 2026-05-28)

## Project

Distributed greedy minimum set cover for OPC segment dedup. Splits a semiconductor
layout's full chip into patches, then finds the minimum patch set whose union
covers every unique hash. Design rationale lives in [`greedy_set_cover.md`](greedy_set_cover.md).

## Milestone status

| M | Title | Status | Branch |
|---|---|---|---|
| M1 | Single-rank end-to-end greedy set cover | ✅ Done | `main` |
| M2 | Multi-rank MPI via stComm (USCSolver template) | ✅ Done | `main` |
| M3 | Distributed sample sort setup (§5.1) + sparse bcast (§7.2) | ✅ Done | `main` |
| M4 | GPU port (CUDA kernels, CUB ArgMax, DeviceBuffer) | ✅ Done | `gpu` |
| M5 | NCCL device-direct broadcast | ⏳ Planned | will continue on `gpu` |
| M6 | Element-partitioned covered bitset (§7.3) | ⏳ Planned | |
| M7 | Real OPC input parser + 2D partition (§7.4) | ⏳ Planned | |

## Branch model

```
main  d4e866c  M3 baseline (CPU-only, distributed sample sort) — stable reference
gpu   e859c26  M4-7 (current tip) — GPU port; M5+ continues here
```

- `main`: CPU-only. Useful as a regression reference and for environments without GPU.
- `gpu`: All GPU work. M5/M6 commit here directly.

## Architecture summary

```
                 main()  (src/main.cpp)
                   │
                   ▼
                 USCSolver<stComm::MPIComm>     ← template strategy, M5 adds NCCLComm
                   │
       ┌───────────┴───────────┐
       │                       │
   load(raw,gids)         load(raw,gids)        ← per-rank local patches
       │                       │
       ▼                       ▼
   setup()                 setup()              ← distributed sample sort §5.1
       │                       │                  hash%size → alltoallv → Exscan →
       │                       │                  reverse alltoallv → host PatchCsr +
       │                       │                  H2D to device buffers
       ▼                       ▼
   solve()                 solve()              ← multi-rank greedy main loop
       │                       │
       └───────────┬───────────┘
                   ▼
              SolverResult                      ← identical on every rank
              (selected, covered_count, iters)
```

### Source layout

```
src/
├── main.cpp                       # CLI entry point
├── core/
│   ├── types.hpp                  # ElementId, PatchId, Score, Hash, kDisabledScore
│   ├── bitset.{hpp,cpp}           # DenseBitset (host) — used by brute_force only
│   ├── csr.{hpp,cpp}              # PatchCsr (CSR layout, design doc §3)
│   ├── inverted_index.{hpp,cpp}   # Sparse CSR transpose (§4)
│   ├── usc_solver.{hpp,cu}        # USCSolver<CommT> template (algorithm core)
│   ├── brute_force.{hpp,cpp}      # Single-process reference for tests
│   ├── device_buffer.{hpp,cu}     # DeviceBuffer<T> RAII wrapper (M4)
│   └── (M5+ may add: nccl_context.*)
└── data/
    └── synthetic.{hpp,cpp}        # Deterministic synthetic generator + rank slicer

tests/
├── test_*.cpp                     # Unit tests per module
├── test_solver_equivalence.cpp    # USCSolver vs brute_force (the key correctness test)
├── test_main.cpp                  # Custom GTest main with stComm::MPIComm::initialize
└── helpers/
    └── local_setup.{hpp,cpp}      # Single-process setup for brute_force ref
```

### Dependencies

| Dep | Where | Notes |
|---|---|---|
| MPI (OpenMPI/MPICH) | system | `mpicxx` driver, `find_package(MPI)` |
| stComm | `~/install/stComm` | Source at `~/tickets/stComm`. PR #1 adds bcast/maxloc/exscan |
| CUDA Toolkit | `/usr/local/cuda` (12.8) | M4+ requires. nvcc compiles `usc_solver.cu` and `device_buffer.cu` |
| NCCL | system (`libnccl-dev`) | Required by stComm even if we don't use NCCLComm yet |
| GoogleTest | FetchContent (v1.15.2) | tests only |

### Algorithm hot-path data placement (M4 final)

| Data | Host | Device |
|---|---|---|
| PatchCsr | Yes (offsets read at build kernel launch) | `d_patch_data_`, `d_patch_offsets_` |
| InvertedIndex | Yes (currently unused by solve) | `d_inv_*` (currently unused) |
| covered (cumulative) | — | `d_covered_` |
| scores | — | `d_scores_` |
| newly_covered_ids (sparse) | Yes (MPI bcast stage) | `d_newly_covered_ids` |
| newly_covered bitset | — | `d_newly_covered_` |

Per-iteration MPI staging (M5 will replace with NCCL device-direct):
- D2H of 16 B (CUB ArgMax result) → MPI_Allreduce<MAXLOC>
- MPI_Bcast<PatchId> winner_global (8 B host scalar)
- D2H winner_score×8 B → MPI_Bcast<ElementId> → H2D on non-winners
- H2D 8 B (Score = -1) for winner disable

## Key design decisions (recorded for future maintainers)

1. **Strategy pattern via template** — `USCSolver<CommT>` is parameterized on
   the comm backend. M2–M4 instantiate `<stComm::MPIComm>`. M5 adds
   `<stComm::NCCLComm>`. Explicit instantiation in `usc_solver.cu` keeps the
   header free of CUDA includes.

2. **DeviceBuffer split** — `device_buffer.hpp` (CUDA-free) defines
   `DeviceBufferImpl` (non-template) and `DeviceBuffer<T>` (header-only
   template wrapper). `device_buffer.cu` is the only TU that touches
   `cuda_runtime.h`. This keeps `usc_solver.hpp` consumable by host TUs
   (main.cpp, tests) without CUDA pollution.

3. **All collective comm lives in stComm** — fullchipUSC never wraps MPI/NCCL
   calls. Custom protocols (e.g., sparse bcast) become free functions or
   stComm extensions, not USCSolver methods.

4. **Sample sort partitioning** — `hash % size` was the first cut; replaced
   with proper distributed sample sort in M3 for robustness against skewed
   hash distributions. `compute_splitters()` in `usc_solver.cu` implements
   the 6 standard sample sort steps.

5. **Score update strategy B (full sweep)** — design doc §6.3 alternatives:
   A (inverted-index, affected-only) and B (full sweep). B was chosen for
   GPU because A's irregular gather is poor on device. B's wasted work on
   non-affected patches short-circuits via `if (scores[p] <= 0)` and the
   `block_delta > 0` write guard.

6. **Host MPI stays in M4** — moving collectives to NCCL is M5. M4's
   D2H/H2D boundaries around MPI calls are documented in `solve()` comments.

## Verification baseline

All sizes produce **bit-identical** results between size=1 and size=4 mpirun.

| Scale | N | M | K_mean | seed | selected | covered |
|---|---|---|---|---|---|---|
| Small | 10000 (raw) → 6019 (universe) | 1000 | 50 | 42 | 353 | 6019/6019 |
| Medium | 50000 → 30001 | 2000 | 100 | 1 | 1121 | 30001/30001 |
| Large | 200000 → 120045 | 5000 | 200 | 1 | 2526 | 120045/120045 |

27 ctest cases all pass under `mpirun -n 4 --oversubscribe`.

## Build & run (current environment)

```bash
# Prerequisites
# - mpicxx (OpenMPI or MPICH)
# - nvcc 12.x at /usr/local/cuda/bin/nvcc
# - libnccl-dev (apt install libnccl2 libnccl-dev)
# - stComm installed at ~/install/stComm (see ~/tickets/stComm)

# Configure
CMAKE_PREFIX_PATH=~/install/stComm cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j 4

# Run tests (size=4)
cd build && ctest --output-on-failure

# Smoke run
mpirun -n 4 --oversubscribe ./build/src/fullchipusc-solve \
    --N 10000 --M 1000 --K 50 --overlap 0.4 --seed 42
```

Expected smoke output at any size: `selected=353 covered=6019/6019 iterations=353`.

## Known performance notes (single MX450)

- `solve()` GPU time exceeds the host equivalent at our test scales (K≈50–200).
  Kernel launch overhead dominates; GPU wins are expected at K≥1000 or
  N≥10^8 with proper multi-GPU.
- `setup()` shows ~800 ms first-iteration cold-start from CUDA runtime
  initialization. Subsequent calls in the same process would be fast.
- `mpirun -n 4` on a single GPU serializes work across processes (no MPS
  enabled). Real multi-GPU testing requires N GPUs or MPS daemon.
