# fullchipUSC — Status (as of M5 completion, 2026-05-28)

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
| M5 | NCCL device-direct broadcast for newly_covered_ids | ✅ Done | `gpu` |
| M5.5 | score_update kernel rework (strategy A or fusion) — driven by profile | ⏳ Next | `gpu` |
| M6 | Element-partitioned covered bitset (§7.3) | ⏳ Planned | |
| M7 | Real OPC input parser + 2D partition (§7.4) | ⏳ Planned | |

## Branch model

```
main  d4e866c  M3 baseline (CPU-only, distributed sample sort) — stable reference
gpu   <M5 tip> M5 complete (NCCL device-direct + profile instrumentation)
```

- `main`: CPU-only. Useful as a regression reference and for environments without GPU.
- `gpu`: All GPU work. M5.5+ commits here directly.

## M5 architecture (current)

```
                 main()  (src/main.cpp)
                   │  ── cudaSetDevice(rank % nGPU)
                   │  ── ncclUniqueId from rank 0, MPI_Bcast, NCCLComm::initialize
                   ▼
                 USCSolver<stComm::MPIComm>(mpi_comm, shared_ptr<NCCLComm>)
                   │
       ┌───────────┴───────────┐
       │                       │
   load(raw,gids)         load(raw,gids)        ← per-rank local patches
       │                       │
       ▼                       ▼
   setup()                 setup()              ← distributed sample sort §5.1
       │                       │                  via MPIComm (host buffers).
       │                       │                  Ends with H2D to device buffers.
       ▼                       ▼
   solve()                 solve()              ← multi-rank greedy main loop;
       │                       │                  see "Per-iteration boundary" below.
       └───────────┬───────────┘
                   ▼
              SolverResult                      ← identical on every rank
              (selected, covered_count, iters)
```

### Source layout

```
src/
├── main.cpp                       # CLI entry; CUDA device pick + NCCLComm bootstrap
├── core/
│   ├── types.hpp                  # ElementId, PatchId, Score, Hash, kDisabledScore
│   ├── bitset.{hpp,cpp}           # DenseBitset (host) — used by brute_force only
│   ├── csr.{hpp,cpp}              # PatchCsr (CSR layout, design doc §3)
│   ├── inverted_index.{hpp,cpp}   # Sparse CSR transpose (§4)
│   ├── usc_solver.{hpp,cu}        # USCSolver<CommT> template (algorithm core)
│   ├── brute_force.{hpp,cpp}      # Single-process reference for tests
│   └── device_buffer.{hpp,cu}     # DeviceBuffer<T> RAII wrapper (M4)
└── data/
    └── synthetic.{hpp,cpp}        # Deterministic synthetic generator + rank slicer

tests/
├── test_main.cpp                  # GTest main; bootstraps process-global NCCLComm
├── test_nccl_env.hpp              # accessor for the global NCCLComm
├── test_*.cpp                     # Unit tests per module
├── test_solver_equivalence.cpp    # USCSolver vs brute_force (correctness test)
└── helpers/
    └── local_setup.{hpp,cpp}      # Single-process setup for brute_force ref
```

### Dependencies

| Dep | Where | Notes |
|---|---|---|
| MPI (OpenMPI/MPICH) | system | `mpicxx`, `find_package(MPI)` |
| stComm | `~/install/stComm` | branch `add-nccl-bcast` — adds `NCCLComm::bcast<T>` on top of `add-bcast-maxloc-exscan` |
| CUDA Toolkit | `/usr/local/cuda` (12.x) | nvcc compiles `usc_solver.cu`, `device_buffer.cu` |
| NCCL | system (`libnccl-dev`) | required at runtime for the hot-path bcast |
| GoogleTest | system (`libgtest-dev`) for stComm; FetchContent 1.15.2 for fullchipUSC | |

### Per-iteration host/device boundary (M5)

| Step | Where | Cost |
|---|---|---|
| CUB ArgMax + 16 B D2H | device + host stub | 16 B/iter |
| MPI Allreduce&lt;MAXLOC&gt; on (score,rank) | host | NCCL has no MAXLOC |
| MPI Bcast&lt;PatchId&gt; winner_global (8 B) | host | tiny |
| build_newly_covered_kernel (winner only) | device | — |
| **NCCL Bcast&lt;ElementId&gt; on d_newly_covered_ids** | **device-direct** | **no host staging** |
| set_bits ×2 + score_update_kernel | device | — |
| H2D 8 B (Score = -1) winner disable | host stub | tiny |

The payload (`d_newly_covered_ids`, sized winner_score × 8 B) **never visits host
memory** in M5. Only the 16 B MAXLOC metadata is host-side, dictated by the NCCL
API (root/count must be host scalars).

## Key design decisions (recorded for future maintainers)

1. **`USCSolver<MPIComm>` + injected `shared_ptr<NCCLComm>`** — host setup +
   tiny metadata stay on MPIComm (NCCL lacks MAXLOC/EXSCAN); the per-iter big
   payload goes through the injected NCCLComm. Single template instantiation;
   the alternative (`USCSolver<NCCLComm>` for the full path) would have
   required reworking setup for device-resident data, which is M6+ territory.

2. **GPU is required, no fallbacks** — fullchipUSC is GPU-first by premise.
   No CPU-only path, no `--backend mpi` toggle. The CPU baseline lives on the
   `main` branch for bisect, not in product code.

3. **Smart pointers for host ownership** — host-side optional/owned objects
   use `std::shared_ptr` or `std::unique_ptr`. No raw owning pointers, no
   `new`/`delete`. References are fine for non-owning, non-nullable views.

4. **DeviceBuffer split** — `device_buffer.hpp` (CUDA-free) defines
   `DeviceBufferImpl` (non-template) and `DeviceBuffer<T>` (header-only
   template wrapper). `device_buffer.cu` is the only TU that touches
   `cuda_runtime.h`.

5. **All collective comm lives in stComm** — fullchipUSC never wraps MPI/NCCL
   calls. Custom protocols (e.g., sparse bcast) become free functions or
   stComm extensions, not USCSolver methods.

6. **Sample sort partitioning** — `hash % size` was the first cut; replaced
   with proper distributed sample sort in M3 for robustness against skewed
   hash distributions. `compute_splitters()` implements the 6 standard
   sample sort steps.

7. **Score update strategy B (full sweep) — for now** — design doc §6.3
   alternatives: A (inverted-index, affected-only) and B (full sweep). B is
   currently used because of its regular kernel pattern. M5 profile shows
   `score_update_kernel` dominates (78% of solve at the 20× workload), so
   M5.5 will revisit and likely adopt A or a fused variant.

8. **Optional profile instrumentation in solve()** — gated by env var
   `FULLCHIPUSC_PROFILE=1`; off path has no sync overhead. Used to pinpoint
   per-stage cost. See "Profile baseline" below.

## Verification baseline (M5)

All sizes produce **bit-identical** results between size=1 and size=2.

| Scale | N (raw) | M | K | seed | universe | iters | selected |
|---|---|---|---|---|---|---|---|
| Small | 10000 | 1000 | 50 | 42 | 6019 | 353 | 353 |
| LARGE (1×) | 500000 | 10000 | 300 | 1 | 350023 | 5378 | 5378 |
| 4× | 2000000 | 20000 | 500 | 1 | 1399007 | 14889 | 14889 |
| 20× | 10000000 | 100000 | 1000 | 1 | 7000295 | 35824 | 35824 |

27 ctest cases all pass under `mpirun -n 2` (1 process per A100).

## Performance — dual A100-SXM4-80GB

### Solve time (ms) — size=1 vs size=2 with 1 process per GPU

| Workload | size=1 | size=2 | speedup |
|---|---|---|---|
| Small (N=10K) | 18 | 33 | 0.55× (NCCL latency floor dominates) |
| LARGE (N=350K univ) | 400 | 458 | 0.87× |
| 4× (N=1.4M univ) | 2413 | 2301 | 1.05× |
| **20× (N=7M univ)** | **32917** | **19349** | **1.70×** |

→ Multi-GPU value emerges as workload grows: NCCL bcast latency (~27 μs/iter)
is constant, so its share of solve time shrinks with scale, while
`score_update`'s work-distribution benefit grows.

### Setup time (ms) — size=2 scaling

| Workload | size=1 | size=2 | speedup |
|---|---|---|---|
| LARGE | 1195 | 643 | 1.86× |
| 4× | 4605 | 2759 | 1.67× |
| 20× | 60394 | 30533 | **1.98×** |

Setup scales near-ideal at 20× — sample sort + alltoallv + H2D are all
work-distributable across ranks.

### Profile baseline — 20× workload, `FULLCHIPUSC_PROFILE=1`

| Stage | size=2 μs/iter | size=2 ms total | share |
|---|---|---|---|
| **score_update** | 421 | **15063** | **78%** ← dominant |
| argmax + 16 B D2H | 28 | 996 | 5% |
| nccl_bcast | 27 | 969 | 5% |
| set_bits ×2 | 17 | 624 | 3% |
| maxloc | 14 | 505 | 3% |
| build_kernel | 8 | 299 | 2% |
| winner_disable, patchid_bcast | — | 343 | 2% |

`score_update` is now the unambiguous next target (M5.5). NCCL latency is
constant ~27 μs/iter and only matters at small workloads.

## Build & run (current environment)

```bash
# One-shot bootstrap (apt + cmake + stComm)
scripts/setup-env.sh

# Configure (override CUDA arch for your GPU; 80 = A100, 86 = RTX 30/A4500, etc.)
export CMAKE_PREFIX_PATH=$HOME/install/stComm:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$HOME/install/stComm/lib:$LD_LIBRARY_PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80

# Build
cmake --build build -j

# Run tests — needs num_gpus >= mpirun -n. Default ctest uses -n 2.
# Override with -DFULLCHIPUSC_TEST_NRANKS=<n> at configure time if you have n GPUs.
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
(cd build && ctest --output-on-failure)

# Smoke
mpirun -n 2 ./build/src/fullchipusc-solve --N 10000 --M 1000 --K 50 --overlap 0.4 --seed 42

# Profile breakdown
FULLCHIPUSC_PROFILE=1 mpirun -n 2 -x FULLCHIPUSC_PROFILE \
    ./build/src/fullchipusc-solve --N 500000 --M 10000 --K 300 --overlap 0.3 --seed 1
```
