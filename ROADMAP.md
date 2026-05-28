# fullchipUSC — Roadmap (M5.5+)

This file describes what's still planned. See [STATUS.md](STATUS.md) for
completed work and the M5 profile baseline that drives the new ordering.

## New-environment prerequisites

### Hardware
- **2+ NVIDIA GPUs per node** (the M5+ verification baseline assumes 1 process
  per GPU; NCCL refuses to map two ranks to one device). Multi-node clusters
  with NVLink/InfiniBand are also fine.
- Single-GPU hosts can still run `mpirun -n 1`, but won't exercise the
  NCCL device-direct path and won't reproduce the multi-GPU speedups.

### Software
| Package | Tested version | Notes |
|---|---|---|
| Linux | Ubuntu 24.04 / WSL2 OK | |
| GCC / G++ | 12.4 | C++20 required |
| CMake | 3.25+ | `CMAKE_CUDA_STANDARD 20`. `scripts/setup-env.sh` installs Kitware 3.29 if apt cmake is too old. |
| CUDA Toolkit | 12.4+ | nvcc must be at `/usr/local/cuda/bin/nvcc` or override `CMAKE_CUDA_COMPILER` |
| NCCL | 2.18+ (libnccl2 + libnccl-dev) | `apt install libnccl2 libnccl-dev` |
| OpenMPI | 3.1+ / 4.1 / 5.x | `mpicxx` available in PATH |
| GoogleTest | 1.15.2 (FetchContent) | |

### One-shot setup

```bash
git clone <fullchipUSC remote>
cd fullchipUSC && git checkout gpu
./scripts/setup-env.sh
# then build per STATUS.md "Build & run"
```

The script auto-detects `CUDA_ARCH` from `nvidia-smi`. Override via env:
`STCOMM_PREFIX`, `STCOMM_SRC`, `CUDA_ARCH`, `CMAKE_VERSION`.

Manual stComm install (alternative):
```bash
git clone https://github.com/ssjl06/stComm ~/tickets/stComm
cd ~/tickets/stComm
git checkout add-nccl-bcast    # has NCCLComm::bcast (M5 upstream)
export CUDA_ARCH=<your GPU arch, e.g. 80 for A100>
./build.sh
./install.sh ~/install/stComm
```

---

## M5.5 — `score_update_kernel` rework (next, profile-driven)

### Why this jumped ahead of M6

M5 profile at the 20× workload (N=10M, M=100K, K=1000) shows
`score_update_kernel` is **78% of solve time** at size=2 (15 of 19 s).
Everything else combined is <22%. Until this is addressed, no other change
moves the needle meaningfully.

### Candidate approaches (pick after a microbench)

| Approach | Idea | Expected effect | Risk |
|---|---|---|---|
| **Strategy A: inverted-index, affected-only** | Per design doc §6.3. After each iter, gather only patches that share an element with `newly_covered`; skip the rest. Inverted index `d_inv_*` is already built but unused. | Big in later iters as `affected` shrinks; small early. -30–50% on `score_update`. | Irregular gather; needs careful kernel design. atomicSub on scores. |
| **Strategy B + fusion** | Keep full sweep but fuse `set_bits` + `score_update` into one kernel (read `newly_covered_ids` once, then sweep). | -20–30% via fewer launches and one less global pass. | Smaller win; familiar pattern. |
| **CUDA Graph capture of the iteration** | Capture the 7-kernel iter pattern as a graph, launch with `cudaGraphLaunch`. Re-record only when `winner_score` changes "shape." | -1–2 s (launch overhead). Complements either A or B. | Recapture cost when shape changes; needs guard. |
| **Hybrid A/B** | Choose per-iter based on `popcount(newly_covered)` — design doc §6.3 explicitly suggests this. | Best in theory; both kernels must exist. | More code. |

Recommended order: (1) **strategy A first** because it's the design-doc
default and exploits already-built `d_inv_*`; (2) measure; (3) add graph
capture if launch overhead still shows up.

### Verification target

- bit-identical against M5 baseline at all four scales (Small / LARGE / 4× /
  20×) — `selected` sequence must match exactly.
- 27 ctest pass at size=2.
- 20× size=2 solve time should drop from ~19.3 s to ≤12 s if strategy A
  delivers the expected affected-set scaling in late iters.

---

## M6 — Element-partitioned covered bitset (§7.3)

### Goal
When N is large enough that `d_covered_` (N/64 words per rank) becomes a
real memory pressure, partition the covered bitset across ranks so each
holds only its shard.

### Trigger conditions
- N ≥ 10^10 → d_covered_ = 1.25 GB per rank (fits A100 80 GB easily)
- N ≥ 10^11 → 12.5 GB per rank (still fits A100)
- N ≥ 10^12 → only on H100 80 GB / multi-GPU per rank

For the M5 test scales (N ≤ 10^7) this is unnecessary. M6 is a scaling
upgrade, not a perf win at current sizes.

### Algorithm change (design doc §7.3)
- Element ID space [0, N) divided into P shards (size N/P each).
- Rank r owns shard r.
- Per iteration:
  - Each rank computes **partial scores** for its patches against only its
    shard of covered.
  - Partial scores `AllReduce<SUM>` across ranks → full scores.
  - argmax on full scores.
  - Winner broadcasts `newly_covered`; each rank only OR's IDs belonging to
    its shard.

### USCSolver impact
- `d_covered_` becomes per-shard (size N/(64·P)).
- Score kernel maps each element ID to a shard, counts intersections
  within own shard only.
- Requires adding `allreduce<T>(op)` to stComm (currently only
  `allreduceMaxloc<T>`).

### Open decisions
- Static shard partition vs hash-based? Static is simpler when IDs are
  approximately uniform (sample sort already enforces this).
- Score AllReduce bandwidth (M × 4 B per iter) vs current N-bit bcast.

---

## M7 — Real OPC input parser + 2D partition (§7.4)

### Goal
- Replace synthetic generator with real OPC binary/HDF5 reader.
- If M × K exceeds single-node RAM (~2 TB / 8 = 256 GB of patches per node),
  add 2D partition (patch shard × element shard) per design doc §7.4.

### Likely structure
- `src/io/` with a `PatchReader` interface and concrete implementations
  (binary, HDF5, …).
- Move `generate_synthetic` + `slice_patches_by_rank` to `tests/helpers/`
  since synthetic is no longer the production path.
- App accepts `--input <path>` and dispatches to the appropriate reader.

### Note
Most uncertain milestone — depends on what the actual OPC toolchain emits.
Capture format spec from the OPC team before designing the reader.

---

## Cross-cutting follow-ups (no milestone assigned)

- **stComm `Allreduce<T>(op)` PR**: blocks M6. Currently only `allreduceMaxloc<T>`.
- **CI / automated regression**: GitHub Actions or similar with MPI + CUDA
  matrix. Trigger on PR. Currently a manual `ctest` discipline.
- **`patches_` / `inv_` host memory free after H2D**: currently both stay in
  host RAM after `setup()` copies to device, doubling host footprint.
  `build_newly_covered_kernel` still reads `patches_.offsets[lbp]` on host —
  could pull from `d_patch_offsets_` via a tiny D2H to drop the host
  dependency.
- **`d_inv_*` allocated but unused at M5** — used by M5.5 strategy A if we
  go that route; otherwise remove the device mirror.
- **Determinism audit**: `build_newly_covered_kernel`'s `atomicAdd` makes
  `d_newly_covered_ids` ordering non-deterministic. The *set* is identical
  and downstream consumers are order-insensitive, but if anything ever
  relies on order this needs a stable sort or grid-strided write.
- **Stream awareness**: All kernels use the default stream. Per-step streams
  + explicit `cudaStreamSynchronize` would document intent and unlock
  comm/compute overlap.
- **Profile gating**: `FULLCHIPUSC_PROFILE=1` instrumentation lives in
  `solve()`; off path has no sync. Keep until M5.5 is done so we can
  re-measure stage shares.
