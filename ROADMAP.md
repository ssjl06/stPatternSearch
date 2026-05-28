# fullchipUSC — Roadmap (M5+)

This file describes what's still planned and what the new environment needs
to do M5+ work. See [STATUS.md](STATUS.md) for completed work.

## New-environment prerequisites

The current dev host (single MX450) cannot exercise multi-GPU NCCL. M5+ needs:

### Hardware
- **2+ NVIDIA GPUs** in one node (for size=2 NCCL testing), or
- **Multi-node cluster** with NVLink/InfiniBand for size=4+ NCCL
- Failing that, **CUDA MPS** enabled so multiple processes share a single GPU
  via NCCL (degraded performance but functionally testable)

### Software
| Package | Tested version | Notes |
|---|---|---|
| Linux | Ubuntu 24.04 / WSL2 OK | |
| GCC / G++ | 12.4 | C++20 required |
| CMake | 3.22+ | `enable_language(CUDA)` |
| CUDA Toolkit | 12.8 | nvcc must be at `/usr/local/cuda/bin/nvcc` or override `CMAKE_CUDA_COMPILER` |
| NCCL | 2.18+ (libnccl2 + libnccl-dev) | `apt install libnccl2 libnccl-dev` |
| OpenMPI | 4.1 or 5.x | `mpicxx` available in PATH |
| GoogleTest | 1.15.2 (auto via FetchContent) | |

### stComm install
```bash
git clone https://github.com/ssjl06/stComm ~/tickets/stComm
cd ~/tickets/stComm
git checkout add-bcast-maxloc-exscan        # has bcast/maxloc/exscan (PR #1)
export CUDA_ARCH=<your GPU arch, e.g. 80 for A100>
./build.sh
./install.sh ~/install/stComm
```

Then for fullchipUSC: `CMAKE_PREFIX_PATH=~/install/stComm cmake ...`

### Repo bootstrap
```bash
git clone <fullchipUSC remote>   # or copy directory if no remote yet
cd fullchipUSC
git checkout gpu                  # M4 tip
# Build / test as in STATUS.md
```

---

## M5 — NCCL device-direct broadcast

### Goal
Replace the per-iteration D2H → MPI_Bcast → H2D round-trip for
`newly_covered_ids` with a NCCL device-to-device broadcast. Eliminates the
biggest remaining bottleneck: host memory bus traffic per iteration.

### Required upstream changes

**stComm needs `NCCLComm::bcast<T>`** (currently has send/recv/allgatherv/
alltoallv but no broadcast). Open as a follow-up PR to ssjl06/stComm.

Sketch:
```cpp
// stComm/include/stComm/nccl_comm.h
template<typename T>
RequestPtr bcast(T* data, std::size_t count, int root);

// stComm/src/stComm/nccl_comm.cpp (or .h if inline)
// → ncclBroadcast(data, data, count*sizeof(T), ncclChar, root, comm_, stream_)
// Or use the appropriate ncclDataType for T.
```

### fullchipUSC changes

1. **Explicit instantiation**: add `template class USCSolver<stComm::NCCLComm>;`
   at the bottom of `usc_solver.cu`.

2. **Constructor / setup signature**: NCCLComm requires explicit
   `initialize(rank, size, device_id, comm_id)`. The current
   `USCSolver(CommT& comm)` constructor still works — the caller (main) is
   responsible for initializing NCCLComm before passing it in.

3. **Hybrid main**:
   ```cpp
   stComm::MPIComm::initialize(&argc, &argv);
   stComm::MPIComm world;
   
   // NCCL setup (boilerplate from stComm README)
   ncclUniqueId nccl_id;
   if (world.getRank() == 0) nccl_id = stComm::NCCLComm::getUniqueId();
   MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
   int device_id = world.getRank() % num_gpus_per_node;
   cudaSetDevice(device_id);
   stComm::NCCLComm gpu_comm;
   gpu_comm.initialize(world.getRank(), world.getSize(), device_id, nccl_id);
   
   USCSolver<stComm::NCCLComm> solver(gpu_comm);
   // load_synthetic + setup + solve as before
   ```

4. **solve() loop changes**:
   - Replace the `D2H newly_covered_ids → MPI_Bcast → H2D` block with a single
     `gpu_comm.bcast<ElementId>(d_newly_covered_ids.data(), winner_score, winner_rank).wait()`.
   - winner_global PatchId bcast: also device, or keep as MPI (8 bytes is tiny).
   - MPI MAXLOC stays — NCCL has no MAXLOC.

5. **Setup phase comm choice**: setup() uses allgatherv/alltoallv/exscan/bcast,
   not all of which NCCL supports. Two options:
   - (a) Setup remains on MPIComm; only solve()'s big broadcast uses NCCL.
     Cleanest, requires holding both comms.
   - (b) Migrate everything to NCCL where supported, fallback to MPI for
     missing ops. More complex.
   
   Recommend (a) for the first M5 iteration.

### Verification target

- 27 ctest pass with **two backend variants** (MPIComm and NCCLComm).
  Consider parameterizing `test_solver_equivalence.cpp` over the backend.
- Multi-GPU smoke at size=2 (minimum to exercise NCCL): selected sequence
  must match the M3/M4 baselines bit-identical.
- Performance: solve() time should now drop measurably as the per-iteration
  D2H/H2D goes away. Compare against M4 numbers in STATUS.md.

### Open decisions for M5

- Per-rank GPU assignment: round-robin (`rank % gpus_per_node`) vs explicit
  config? Round-robin is simplest.
- NCCL communicator group create logic: must call `getUniqueId()` only on
  rank 0, then MPI_Bcast it. Document carefully.
- What to do if size > num GPUs and MPS isn't available? Fail loud at init.

---

## M6 — Element-partitioned covered bitset (§7.3)

### Goal
When N is large enough that `d_covered_` (N/64 words per rank) doesn't fit
in GPU memory, partition the covered bitset across ranks so each holds only
its shard.

### Trigger conditions
- N ≥ 10^10 → d_covered_ = 1.25 GB per rank (still fits, marginal)
- N ≥ 10^11 → 12.5 GB per rank (typical A100 80GB fits, smaller GPUs fail)
- N ≥ 10^12 → only on H100 80GB / multi-GPU per rank

For M4 test scales (N ~ 10^5), this is unnecessary. M6 is purely a scaling
upgrade.

### Algorithm change (design doc §7.3)
- Element ID space [0, N) is divided into P shards (size N/P each)
- Rank r owns shard r
- Per iteration:
  - Each rank computes **partial scores** for its patches against only its
    shard of covered (i.e., element IDs that map to its shard)
  - Partial scores AllReduce SUM across ranks → full scores
  - argmax on full scores (now M elements per rank — but M ≤ M_global)
  - Winner broadcasts its newly_covered as before, BUT each rank only OR's
    the IDs that belong to its shard

### USCSolver impact
- `d_covered_` becomes per-shard (size N/(64*P))
- Score kernel needs to map each element ID to a shard and only count
  intersections within own shard
- Add an `Allreduce<SUM>` over the M scores array — requires stComm to add
  `allreduce<T>` (currently only allreduceMaxloc<T>)

### Open decisions for M6
- Static shard partition (rank r owns IDs [r*N/P, (r+1)*N/P)) vs hash-based?
  Static is simpler and sufficient if IDs are uniform.
- Score update kernel needs M-sized partial result array per rank → AllReduce
  — confirm bandwidth budget vs current N-bit bcast.

---

## M7 — Real OPC input parser + 2D partition (§7.4)

### Goal
- Replace synthetic generator with real OPC binary/HDF5 reader.
- If M*K exceeds single-node RAM (~2 TB / 8 = 256 GB of patches per node),
  add 2D partition (patch shard × element shard) per design doc §7.4.

### Likely structure
- Add `src/io/` directory with a reader interface (`PatchReader`) and one or
  more concrete implementations (binary, HDF5, …).
- Move `generate_synthetic` + `slice_patches_by_rank` to `tests/helpers/`
  since synthetic is no longer the production path.
- App accepts `--input <path>` and dispatches to the appropriate reader.

### Note
This is the most uncertain milestone — it depends on what the actual OPC
toolchain emits. Capture format spec from the OPC team before designing
the reader.

---

## Other follow-ups (no milestone assigned yet)

- **NCCLComm bcast PR upstream to stComm**: blocks M5 directly.
- **stComm Allreduce<T> with op (SUM/MIN/MAX, not just MAXLOC)**: blocks M6.
- **MPS / multi-GPU testbed**: needed for end-to-end M5+ verification.
- **CI / automated regression**: currently rely on manual `mpirun -n 4 ctest`.
  Set up GitHub Actions or similar with MPI + CUDA when there's a public repo.
- **`patches_` / `inv_` host memory freed after H2D**: currently both stay
  in host memory after setup() copies to device. For large M_local this
  doubles RAM use. Free them at end of setup() once device buffers are
  populated, with a guard for when host code still needs them (currently:
  build_newly_covered_kernel reads `patches_.offsets[lbp]` on host — that
  could pull from `d_patch_offsets_` via a tiny D2H, eliminating the host
  dependency).
- **`d_inv_*` is allocated but unused** (score update uses strategy B). Either
  remove the device mirror or implement a dynamic A/B switching kernel.
- **Determinism audit**: M4-6 introduced atomicAdd in build_newly_covered_kernel
  which makes the order of `d_newly_covered_ids` non-deterministic. The set
  is the same, the final solver result is the same — but if anything
  downstream relies on element order (which it doesn't today), this needs
  revisiting.
- **Stream awareness**: All kernels currently use the default stream.
  For overlap with MPI / NCCL communication, M5+ should consider per-step
  streams. Synchronization currently relies on cudaMemcpy being implicit
  fence — explicit `cudaStreamSynchronize` would document intent better.
