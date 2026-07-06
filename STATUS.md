# stPatternSearch — Status (as of M6 completion, 2026-07-06)

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
| M5.5 | score_update strategy A (inverted-index, affected-only) | ✅ Done | `gpu` |
| — | Segmented argmax (tried, no gain — loop is latency-bound) | ⛔ Reverted | |
| M6 | Element-partitioned covered bitset (§7.3) | ✅ Done (opt-in `PartitionMode::ByElement`) | `m6-element-partition` |
| M6.5 | Device-resident ByElement loop (Full-GPU iteration track) | ✅ Done | `device-resident-loop` |
| M7 | Real OPC input parser + 2D partition (§7.4) | ⏳ In progress (io infra parked on `m7-io-infra`) | |

> **Next-work note (2026-05-29):** the per-iteration hot loop is **latency/
> launch-bound, not compute-bound** (argmax ≈ 20 µs flat across a 5× M range).
> A **device-resident single-rank loop** (no per-iter D2H/H2D, batched
> termination) was prototyped and measured **2.7× / 2.06× faster** at size=1,
> bit-identical — then reverted to keep solve() simple (re-implement from the
> ROADMAP note if single-GPU throughput matters). The **multi-rank all-NCCL**
> version was attempted and reverted (correctness bug at scale, too complex).
> Both lessons + the segmented-argmax post-mortem are in ROADMAP.md
> "Full-GPU iteration track".

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
include/stPS/                      # public API (no MPI/NCCL/CUDA leaks; pImpl)
├── stPS.h                         # umbrella header
├── types.hpp                      # ElementId, PatchId, Score, Hash, kDisabledScore
├── partition.hpp                  # slice_patches_by_rank
└── usc_patch_selector.hpp         # UscPatchSelector, PatchSelection, PartitionMode

src/
├── core/                          # shared, algorithm-neutral primitives
│   ├── bitset.{hpp,cpp}           # DenseBitset (host) — used by brute_force only
│   ├── csr.{hpp,cpp}              # PatchCsr (CSR layout, design doc §3)
│   ├── inverted_index.{hpp,cpp}   # Sparse CSR transpose (§4)
│   ├── patch_set.{hpp,cpp}        # map_hashes_to_element_ids (§5.1) + PatchSet
│   ├── device_buffer.{hpp,cu}     # DeviceBuffer<T> RAII wrapper (M4)
│   └── {gpu,host}_profiler.*      # -DUSC_PROFILE stage timers
├── usc/                           # greedy minimum set-cover algorithm
│   ├── main.cpp                   # CLI entry; device pick + Comm::onDevice
│   ├── usc_patch_selector_impl.{hpp,cu}  # ByPatch + ByElement impls, kernels, facade
│   └── brute_force.{hpp,cpp}      # single-process reference for tests
└── data/
    ├── synthetic.{hpp,cpp}        # deterministic synthetic generator
    └── partition.cpp              # slice_patches_by_rank impl

tests/
├── test_main.cpp                  # GTest main; bootstraps process-global Comm
├── test_nccl_env.hpp              # accessor for the global Comm
├── test_*.cpp                     # unit tests per module
├── test_solver_equivalence.cpp    # both PartitionModes vs brute_force
└── helpers/
    └── local_setup.{hpp,cpp}      # single-process setup for brute_force ref
```

### Dependencies

| Dep | Where | Notes |
|---|---|---|
| MPI (OpenMPI/MPICH) | system | `mpicxx`, `find_package(MPI)` |
| stComm | `~/install/stComm` | branch `add-getstream` — element-wise `allreduce<T>(op)` (merged PR #6) + `NCCLComm::getStream()` for the device-resident loop |
| CUDA Toolkit | `/usr/local/cuda` (12.x) | nvcc compiles `usc_solver.cu`, `device_buffer.cu` |
| NCCL | system (`libnccl-dev`) | required at runtime for the hot-path bcast |
| GoogleTest | system (`libgtest-dev`) for stComm; FetchContent 1.15.2 for stPS | |

### Per-iteration host/device boundary (M5)

| Step | Where | Cost |
|---|---|---|
| CUB ArgMax + 16 B D2H | device + host stub | 16 B/iter |
| MPI Allreduce&lt;MAXLOC&gt; on (score,rank) | host | NCCL has no MAXLOC |
| MPI Bcast&lt;PatchId&gt; winner_global (8 B) | host | tiny |
| build_newly_covered_kernel (winner only) | device | — |
| **NCCL Bcast&lt;ElementId&gt; on d_newly_covered_ids** | **device-direct** | **no host staging** |
| set_bits (d_covered_ only) + score_update_invidx_kernel (strategy A) | device | — |
| H2D 8 B (Score = -1) winner disable | host stub | tiny |

The payload (`d_newly_covered_ids`, sized winner_score × 8 B) **never visits host
memory** in M5. Only the 16 B MAXLOC metadata is host-side, dictated by the NCCL
API (root/count must be host scalars).

## M6 architecture (`PartitionMode::ByElement`, opt-in)

Design doc §7.3 — the N-scaling mode. The element ID space [0, N) is statically
sharded across ranks; each rank holds only its shard of the covered bitset
(N/(8·P) bytes) plus every patch's shard-restricted sub-CSR + inverted index
(each incidence on exactly one rank: M·K/P per rank — tighter than §7.5's full
patch replication). Scores are per-rank partials over the local shard.

Per-iteration hot loop — ONE fixed-size, root-less collective:

| Step | Where | Cost |
|---|---|---|
| **`allreduce<Device,SUM>` partial→full scores** | **device-direct (stComm/NCCL)** | **M_global × 8 B** |
| CUB ArgMax over identical full scores + 16 B D2H | device + host stub | every rank picks the same winner |
| build_newly / set_bits / score_update (strategy A) | device, shard-local | same three kernels as ByPatch |
| winner partial disable | 8 B H2D | tiny |

MAXLOC, winner-id bcast and the newly-covered payload bcast are all gone;
communication is O(M_global)/iter, independent of N. Selection is provably
identical to ByPatch/brute-force: integer sums are exact, and the
smallest-slot tie-break (rank-major slots) equals the smallest-PatchId rule.

**When to use**: N large enough that the replicated covered bitset (or the
ByPatch newly-covered bcast) hurts — the design-doc trigger is N ≥ 10^10+.
At current test scales it is *slower* than ByPatch (the M×8 B allreduce
dominates), exactly as the design doc predicted:

| Workload (`-n 2`, 2×RTX 4000 Ada, patch-select total ms) | ByPatch | ByElement |
|---|---|---|
| Small (N=10K, M=1K) | 318 | 319 |
| LARGE (univ 350K, M=10K) | 1087 | 1205 |
| 4× (univ 1.4M, M=20K) | 3278 | 4501 |

All three scales bit-identical between modes and sizes 1/2 (`selected` =
353 / 5378 / 14889).

### M6.5 — the ByElement loop is DEVICE-RESIDENT (2026-07-06)

Because ByElement's one collective has a fixed count and no root (the exact
blockers that killed the earlier multi-rank device-resident attempt), the host
now enqueues **batches of 128 whole iterations blind** on the NCCL stream —
ncclAllReduce + CUB ArgMax + a single-thread `elem_commit_winner` kernel
(winner append, covered_count, winner disable, done flag, all on device) +
three fixed-grid grid-stride update kernels — and syncs once per batch to read
a 24 B loop state. **Zero per-iteration D2H/H2D round-trips.** At size == 1
the allreduce is skipped (partials are the full scores) — this is the
single-rank device-resident loop the ROADMAP prototype validated at 2.7×.
Iterations enqueued past termination no-op on device; `done` derives only from
allreduced values, so all ranks leave after the same batch.

Measured (patch-select total = setup + solve, ms; same-mode host-loop "before"
from the M6 table — identical setup, so the delta is pure solve):

| Workload | elem `-n 1` before → after | elem `-n 2` before → after | ByPatch `-n 2` (ref) |
|---|---|---|---|
| Small | 34 → **22** | 319 → 483 ⚠ | 318 |
| LARGE | 1553 → **1341** | 1205 → **1034** | 1087 |
| 4× | — → 5393 | 4501 → **3914** | 3278 |
| 20× | | — → 43224 | 31470 |

All bit-identical (353 / 5378 / 14889 / 35824), 31/31 ctest. ⚠ Small `-n 2`
regresses: at a few-hundred-iteration scale the deeply pipelined NCCL enqueue
costs more than the host round-trips it removes (unprofiled; irrelevant to the
mode's target regime). ByElement stays comm-bound by the M×8 B allreduce on
this PCIe/SHM box (20×: 800 KB × 35.8 K iters), so **ByPatch remains the
faster multi-rank mode on this hardware** — the device-resident win (~15%
end-to-end, ~30% solve-only at LARGE/4×) compounds where ByElement is actually
needed: huge N, NVLink-class interconnects. Next comm levers if pursued:
int32 wire scores (halves the allreduce), CUDA-graph capture of the batch
(cuts launch overhead), NVLink validation.

`-DUSC_PROFILE` stage timers are absent from this loop by design — no
per-iteration syncs means no host-visible stage boundaries; profile with
Nsight Systems.

## Key design decisions (recorded for future maintainers)

1. **`USCSolver<MPIComm>` + injected `shared_ptr<NCCLComm>`** — host setup +
   tiny metadata stay on MPIComm (NCCL lacks MAXLOC/EXSCAN); the per-iter big
   payload goes through the injected NCCLComm. Single template instantiation;
   the alternative (`USCSolver<NCCLComm>` for the full path) would have
   required reworking setup for device-resident data, which is M6+ territory.

2. **GPU is required, no fallbacks** — stPS is GPU-first by premise.
   No CPU-only path, no `--backend mpi` toggle. The CPU baseline lives on the
   `main` branch for bisect, not in product code.

3. **Smart pointers for host ownership** — host-side optional/owned objects
   use `std::shared_ptr` or `std::unique_ptr`. No raw owning pointers, no
   `new`/`delete`. References are fine for non-owning, non-nullable views.

4. **DeviceBuffer split** — `device_buffer.hpp` (CUDA-free) defines
   `DeviceBufferImpl` (non-template) and `DeviceBuffer<T>` (header-only
   template wrapper). `device_buffer.cu` is the only TU that touches
   `cuda_runtime.h`.

5. **All collective comm lives in stComm** — stPS never wraps MPI/NCCL
   calls. Custom protocols (e.g., sparse bcast) become free functions or
   stComm extensions, not USCSolver methods.

6. **Sample sort partitioning** — `hash % size` was the first cut; replaced
   with proper distributed sample sort in M3 for robustness against skewed
   hash distributions. `compute_splitters()` implements the 6 standard
   sample sort steps.

7. **Score update strategy A (inverted-index, affected-only)** — design doc
   §6.3. M5 used strategy B (full sweep) and its profile showed
   `score_update_kernel` dominating (78% of solve at the 20× workload). M5.5
   replaced it: `score_update_invidx_kernel` launches one block per *newly
   covered element*, looks the element up in this rank's inverted index
   (`d_inv_*`, built since M3 but unused until now) and decrements the score of
   every patch containing it. Because each element is covered exactly once over
   the whole solve, every (element, patch) incidence is touched at most once —
   total work is O(local inverted-index size) instead of O(iters × M·K). The
   result is **bit-identical** to strategy B: a patch's score is its uncovered-
   element count, so per-element −1 equals popcount(patch ∩ newly_covered), and
   a score-0 patch's elements can never reappear in a future newly_covered list
   (matching B's `score <= 0` early-out). int64 scores use a 64-bit `atomicAdd`
   of −1 (no 64-bit `atomicSub` exists); decrements commute, so scores stay
   deterministic. The strategy-B kernel lives on the `main` branch for bisect.

8. **Optional profile instrumentation in patch_select()** — compiled in by
   `-DUSC_PROFILE=ON` (`GpuProfiler` CUDA-event timers + `HostProfiler` host
   wall-clock timers); off path has no sync overhead. Used to pinpoint
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

### Profile baseline — 20× workload, `-DUSC_PROFILE=ON`

The original M5 profile (2×A100) had `score_update` at **78% of solve**:

| Stage (M5, 2×A100) | size=2 μs/iter | size=2 ms total | share |
|---|---|---|---|
| **score_update (strategy B)** | 421 | **15063** | **78%** ← dominant |
| argmax + 16 B D2H | 28 | 996 | 5% |
| nccl_bcast | 27 | 969 | 5% |
| set_bits ×2 | 17 | 624 | 3% |
| maxloc | 14 | 505 | 3% |
| build_kernel | 8 | 299 | 2% |
| winner_disable, patchid_bcast | — | 343 | 2% |

### M5.5 result — strategy A, 20× workload (2×L4-24GB, this box)

> Note: the M5.5 measurements below are on **2×NVIDIA L4** (the hardware
> available for this milestone), not the 2×A100 used for the M5 table above, so
> absolute solve times are not directly comparable across the two tables. The
> *stage shares* and the within-L4 before/after comparison (next table) are the
> apples-to-apples signal.

| Stage (M5.5, 2×L4) | size=2 μs/iter | size=2 ms total | share |
|---|---|---|---|
| argmax + 16 B D2H | 19.9 | 713 | **29%** ← new top stage |
| nccl_bcast | 14.9 | 535 | 22% |
| **score_update (strategy A)** | 12.5 | **447** | **18%** (was 78%) |
| set_bits (×1 now) | 5.7 | 204 | 8% |
| build_kernel | 4.4 | 158 | 7% |
| winner_disable | 3.1 | 113 | 5% |
| maxloc | 2.6 | 92 | 4% |

solve total at 20× = **2425 ms**; `selected=35824` — bit-identical to M5.

### Strategy B → A, same hardware (2×L4), `score_update` total ms

| Workload | B (was) | A (now) | speedup | solve total B→A |
|---|---|---|---|---|
| LARGE | 106 | 48 | 2.2× | 428 → 349 ms |
| 4× | 853 | 146 | 5.8× | 1765 → 980 ms |

The win grows with M (patch count) because B re-sweeps all M patches every
iteration while A touches each incidence once total. At the 20× scale (M=100K)
B's `score_update` would dominate as on A100; A drops it to 18% of solve.
`argmax + 16 B D2H` (full O(M_local) CUB scan every iter) is now the top stage
— a candidate for a future bucket/segmented argmax, but well behind where
`score_update` was.

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
# Override with -DSTPS_TEST_NRANKS=<n> at configure time if you have n GPUs.
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# On hosts where GPU-to-GPU PCIe P2P is broken (see "Known issues" below) the
# NCCL bcast deadlocks — export NCCL_P2P_DISABLE=1 first. Harmless on NVLink
# boxes other than a small bandwidth hit; required on the 2×L4 container.
export NCCL_P2P_DISABLE=1
(cd build && ctest --output-on-failure)

# Smoke
mpirun -n 2 -x NCCL_P2P_DISABLE ./build/src/usc-patch-select --N 10000 --M 1000 --K 50 --overlap 0.4 --seed 42

# Profile breakdown — configure with -DUSC_PROFILE=ON, then run normally
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 -DUSC_PROFILE=ON && cmake --build build -j
mpirun -n 2 -x NCCL_P2P_DISABLE \
    ./build/src/usc-patch-select --N 500000 --M 10000 --K 300 --overlap 0.3 --seed 1
```

## Known issues

### NCCL P2P deadlock on broken-P2P hosts (2×L4 container) — workaround: `NCCL_P2P_DISABLE=1`

**Symptom.** `mpirun -n 2` hangs forever; `-n 1` is fine. With per-rank tracing
the hang is at the **first NCCL broadcast of iteration 0** (`usc_solver.cu`,
the `nccl_comm_->bcast<ElementId>(...)->wait()` device-direct payload bcast).
Both ranks agree on `winner_score`/`winner_rank` via the MPI MAXLOC allreduce
and clear the MPI patch-id bcast, then the **root rank returns from the NCCL
bcast while the non-root rank blocks in `cudaStreamSynchronize` forever** — a
mismatched-collective deadlock with identical args on both sides.

**Root cause.** Not a code bug — the `solve()` collective order is correct.
On this container NCCL selects `P2P/CUMEM` (PCIe peer-to-peer; the two L4s have
no NVLink) but P2P data transfer never actually completes — a common ACS/IOMMU
limitation in virtualized/container GPU setups. NCCL believes P2P works and
picks it, so the broadcast stalls.

**Confirmation.** With `NCCL_P2P_DISABLE=1` (forces NCCL onto the SHM/socket
transport) the full suite passes reliably (27/27, ~42 s) and the standalone
solver reproduces the documented baselines (`selected=324` for the smoke case,
`selected=5378` for LARGE). With P2P enabled the result is *flaky*: the suite
sometimes passes by lucky timing and sometimes deadlocks at iter 0.

**When to set it.** Required on any host whose GPU-pair P2P is broken (this 2×L4
box). On real NVLink hardware (the 2×A100 box used for the M5 perf tables) leave
P2P enabled — disabling it there only costs a little bcast bandwidth. It is a
runtime env var, deliberately *not* baked into the binary, so NVLink hosts keep
the fast path.
