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
- **Broken GPU-pair PCIe P2P** (common in containers/VMs without NVLink, e.g.
  the 2×L4 box) makes the per-iteration NCCL bcast deadlock at iteration 0.
  Export `NCCL_P2P_DISABLE=1` to fall back to the SHM/socket transport. See
  STATUS.md "Known issues" for the full diagnosis. Leave P2P enabled on real
  NVLink hardware.

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

## M5.5 — `score_update` strategy A ✅ DONE (2026-05-29)

Replaced the strategy-B full sweep with `score_update_invidx_kernel` (design
doc §6.3 strategy A): one block per newly covered element, inverted-index
lookup, per-patch atomic −1. See STATUS.md decision #7 and the M5.5 profile
table for the full write-up.

**Result (2×L4):** bit-identical `selected` at all four scales; 27 ctest pass
at size=2. `score_update` total dropped 2.2× (LARGE) / 5.8× (4×) on the same
hardware, and its 20× profile share fell from **78% → 18%** of solve. The new
top stage is `argmax + 16 B D2H` (29%).

Not pursued (recorded for later, in priority order if argmax becomes the
constraint):
- **Bucket / segmented argmax** to replace the full O(M_local) CUB scan every
  iteration — now the largest single stage.
- **CUDA Graph capture** of the per-iteration kernel chain to shave launch
  overhead (complements any argmax change).
- **Hybrid A/B** (design doc §6.3): pick per-iter by `popcount(newly_covered)`.
  Skipped because strategy A's total work is provably ≤ B's in every iteration
  (A touches a subset of incidences), so B never wins — the hybrid would be
  dead weight at current scales.

---

## M6 — Element-partitioned covered bitset (§7.3)  ← next

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
- ~~**`d_inv_*` allocated but unused at M5**~~ — RESOLVED in M5.5: the device
  inverted index is the backbone of `score_update_invidx_kernel`.
- ~~**`d_newly_covered_` per-iter bitset**~~ — REMOVED in M5.5: strategy A reads
  the sparse list + inverted index directly, so the dense per-iter bitset (and
  its second `set_bits` launch) is gone. `d_covered_` (cumulative) stays.
- **Determinism audit**: `build_newly_covered_kernel`'s `atomicAdd` makes
  `d_newly_covered_ids` ordering non-deterministic. The *set* is identical
  and downstream consumers are order-insensitive (strategy A's per-patch
  decrements commute), but if anything ever relies on order this needs a
  stable sort or grid-strided write.
- **Stream awareness**: All kernels use the default stream. Per-step streams
  + explicit `cudaStreamSynchronize` would document intent and unlock
  comm/compute overlap.
- **Profile gating**: `FULLCHIPUSC_PROFILE=1` instrumentation lives in
  `solve()`; off path has no sync. Kept through M5.5 (used to confirm the
  78% → 18% shift); next useful for a possible argmax rework.
