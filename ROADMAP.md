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

- **Hybrid A/B** (design doc §6.3): pick per-iter by `popcount(newly_covered)`.
  Skipped because strategy A's total work is provably ≤ B's in every iteration
  (A touches a subset of incidences), so B never wins — the hybrid would be
  dead weight at current scales.

---

## Segmented argmax — TRIED, reverted (2026-05-29)

`argmax + 16 B D2H` was the top profile stage after M5.5 (29% of solve), so we
implemented a group-wise (two-level) argmax with dirty-group tracking: cache one
max per fixed-size group of patch scores, recompute only groups whose
*max-holding* patch changed (scores only decrease, so a non-max decrement can't
change a group's max), then a level-1 argmax over the per-group maxima. It was
**bit-identical** (27/27 ctest) but measured **no gain — slightly negative**
(4×: 296→285 ms; 20×: 728→745 ms; solve totals up ~2-4%). Reverted.

**Why it failed — the lesson:** argmax per-iter is **~20 µs at both 4× (M=10k)
and 20× (M=50k)** — flat across a 5× M range. So argmax is **not** O(M)-compute-
bound; the O(M) CUB scan over 50k int64 is <1 µs (400 KB / ~1 TB/s). The ~20 µs
is **per-iteration kernel-launch + blocking 16 B D2H sync latency**, which is
constant in M. The segmented version cut a non-bottleneck and added launch
overhead (1 kernel → 3). **Lesson: the per-iteration hot loop on this box is
latency/overhead-bound, not compute-bound — optimize launches/syncs, not FLOPs.**

## Full-GPU iteration track — single-rank PROTOTYPED (2.7×), multi-rank reverted

Bigger lever for the latency-bound loop: keep the **entire iteration device-
resident** so the host enqueues the kernel chain without blocking and the GPU
pipeline stays full, eliminating the per-iteration device→host→device round-trips.

### Single-rank result (2026-05-29) — validated, then reverted for simplicity

A prototype `solve_device_resident()` (size==1) moved the whole per-iteration
flow onto the device: a `resolve_winner` kernel reads the CUB argmax result and
looks up the patch range/global-id on device; build/set_bits/score_update use
fixed grids + grid-stride loops (no host-known winner_score); a `commit` kernel
advances covered_count / selected / winner-disable. The host only read a device
`done` flag once per 128-iteration batch. This removed the two per-iter host
round-trips the size=1 profile flagged — the 16 B argmax **D2H** (725 ms at 20×)
and the 8 B winner-disable **H2D** (186 ms).

**Bit-identical** (selected sequence matched baseline; 27/27 ctest intact) and
**measured 2.7× (4×: 800→297 ms) / 2.06× (20×: 2346→1138 ms) on 2×L4 at size=1.**
So the host-round-trip elimination is a real win — confirming the loop was
latency/sync-bound, not compute-bound. **Reverted** (kept baseline) because the
multi-rank version below didn't pan out and the team preferred to keep solve()
simple; re-implement from this note if single-GPU throughput matters.

### Multi-rank attempt (2026-05-29) — reverted

Tried an all-NCCL, all-kernel multi-rank loop: replace the MPI MAXLOC + patch-id
bcast with a packed-key `ncclAllReduce<MAX>`, and distribute newly_covered via a
fixed-size `ncclAllReduce<MIN>` with sentinel padding (root-less + fixed count →
no host scalars), all kernels on the NCCL stream (added `NCCLComm::allreduce` +
`getStream()` to stComm). Single-rank-style batched termination, no per-iter D2H.
Small cases matched, but at LARGE the selected count diverged (3184 vs baseline
5378) — a real bug (likely in the sentinel buffer build/reduce or tie-break
cascade) that wasn't root-caused before it grew too complex. **Reverted** (code
+ stComm changes). Revisit with a cleaner design and a per-iteration
winner-sequence diff harness to catch the first divergence.

### If resumed

- **Blockers to a fully-device loop** (not just MAXLOC):
  1. NCCL bcast `root`/`count` must be host scalars, but the sparse
     newly_covered `count` (= winner_score) is data-dependent.
  2. Loop termination `winner_score <= 0` is a host branch.
- **Enabling tech:** CUDA Graph **conditional nodes** (CUDA 12.3+, present here)
  for a device-driven while loop; **NVSHMEM** for device-initiated communication
  (GPU triggers comm from inside kernels — no host round-trip, no host-side
  count/root).
- **MAXLOC on GPU is possible** (pack `(score, global_patch_id)` into a
  comparable 64-bit key, `ncclAllReduce(ncclMax)`; ties → smallest patch id,
  matching brute_force) but **counterproductive on this 2×L4 box**: NCCL small-
  message latency is ~16 µs vs MPI host ~2.5 µs (no NVLink → socket/shm
  transport), and the per-iter data dependency forces *two* sequential
  collectives (find winner → broadcast winner's data). Moving the tiny MAXLOC to
  NCCL would *add* ~13 µs/iter (~470 ms at 20×).
- **Where it pays off:** NVLink/NVSwitch (A100/H100, NCCL latency ~3-5 µs). There
  the device-resident + graph-fused iteration could collapse the ~69 µs/iter
  toward 2×NCCL + tiny compute. **Cannot be validated on L4** — the latency
  trade-off runs the wrong way here (same hardware-dependence trap as the
  segmented-argmax attempt). Prototype on NVLink hardware.

---

## M6 — Element-partitioned covered bitset (§7.3) — DEFERRED (speculative)

> Deferred 2026-05-29: M6 swaps M5's patch-partition for **element-partition +
> patch-replicate**. But patch replication holds M×K incidence data on every rank
> (M×K ≥ N always: every element is in ≥1 patch), so it shrinks the *smaller*
> replicated structure (covered, N/8 B) while keeping the *larger* one (patches,
> ≥ 8N B) fully replicated. It only helps in the narrow regime where N breaks GPU
> covered memory yet M×K still fits one node's RAM. Whether the real OPC workload
> is in that regime is **unmeasured** (design doc §9: N and M×K unknown). At our
> test scales (N ≤ 10⁷) it gives no benefit and adds an M×8 B AllReduce per iter.
> Revisit once real N / M×K numbers exist; the general scaling answer is the
> §7.4 2D partition (M7), not M6.

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
