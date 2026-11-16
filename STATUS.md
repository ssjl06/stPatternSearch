# stPatternSearch — Status (2026-11-16)

What the code does today. The dated record of how it got here — milestone
notes, older measurements, abandoned attempts — is in [HISTORY.md](HISTORY.md);
the algorithm's design rationale is in [`greedy_set_cover.md`](greedy_set_cover.md).

## Project

Two distributed GPU components over a set of *patches*, each a list of element
hashes (an OPC layout cut into windows):

- **USC** (`UscPatchSelector`, `usc-patch-select`): greedy minimum set cover —
  the smallest set of patches whose union holds every distinct hash.
- **UPS** (`UpsPatternStats`, `ups-pattern-stats`): per-hash statistics — how
  many patches contain each hash and a representative location — plus an
  export of every distinct hash.

One rank per GPU. Ranks talk through [stComm](../stComm)'s `Comm`
(`Space::Host` = MPI, `Space::Device` = NCCL). In device mode everything from
the upload on runs in GPU memory: the phase timers of a file-driven run show
no host phase, only transfers.

| Component | State |
|---|---|
| Setup (hash → ID mapping, patch CSR, inverted index) | on the device, shared by USC and UPS |
| USC `PartitionMode::ByPatch` (default) | device-resident loop, several selections per round |
| USC `PartitionMode::ByElement` | device-resident loop, several selections per round |
| UPS statistics | device pipeline; k-sized final merge on the host |
| Unique-hash export | host-only mode available (no GPU) |
| Real OPC format reader + 2D partition (M7b) | blocked on the OPC team's format spec |

## Input

Every entry point takes either form (`<stPS/partition.hpp>`):

- nested — one `std::vector<Hash>` per patch;
- flat — `FlatPatches`: hashes patch after patch + CSR offsets (+ optional
  per-hash coords, per-patch global IDs).

Flat is the native one: the `.stps` reader fills a `FlatPatches` straight from
the file and both CLI tools use it, so the hashes go from disk to the GPU in
one upload. The nested overloads flatten on the host and take the same path.

## Setup (device)

`PatchSet` (`src/core/patch_set.*`) and the ByElement selector both start with
`map_hashes_to_element_ids_device`:

1. Sort + unique this rank's hashes (radix sort, head flags + prefix sum).
2. Sample-sort splitters (128 samples per rank — the only hashes that visit
   the host), then a device-direct exchange sends every distinct hash to its
   owner rank.
3. Owners sort + unique what they got — their *shard* — and number it; shards
   are contiguous and rank-ordered, so **global ID order is global hash
   order**. The IDs travel back the same way.
4. Every occurrence is translated; two stable radix passes order the pairs by
   (patch, ID) and duplicates inside a patch are dropped → a device CSR.

Then the inverted index (element → patches) is built on the device: one stable
radix sort of (element, patch) pairs, keys/offsets from the run boundaries.

Only 64-bit-count CUB entry points are used here. Peak device memory in these
stages is 32 B per incidence (two key + two value buffers of a pair sort).
The host mapping (`map_hashes_to_element_ids`) remains as the test reference
and for the host-only unique-hash export.

## USC — the round (both modes)

Sequential greedy picks the best patch, updates every score, and repeats. A
round here takes the K best patches at once, ordered the way greedy breaks
ties — score descending, then rank, then position on the rank — and walks them
in that order:

- A candidate sharing no still-uncovered element with the patches accepted so
  far this round keeps its score; every other score can only have dropped, so
  it is exactly the patch greedy would pick next: **accept**.
- A candidate that shares some is **skipped**; the largest score a skipped
  candidate can still have is remembered.
- When that bound reaches the next candidate's score the round **ends**.

The accepted patches, in order, are sequential greedy's selections; every test
compares against a host reference (brute force or lazy greedy).

Shared machinery (`src/usc/usc_patch_selector_impl.cu`):

- **Block maxima.** Scores are cut into blocks of 2^shift patches, each with
  its maximum and first position; kernels that change a score queue the block
  and only queued blocks are recomputed.
- **Candidate selection in one pass.** The K best patches lie in the K blocks
  with the best block maxima, so two selections — over the block maxima, then
  inside the chosen blocks — see only `nb + K × block_size` values. Each finds
  a threshold score by a search with one parallel count per step.
- **Device-resident batches.** The host enqueues a batch of rounds on the NCCL
  stream and reads 24 B of loop state per batch. Scores are int32 on the device.

### ByPatch

Each rank holds its own patches and the full covered bitset.

| Step | Where |
|---|---|
| refresh block maxima; select this rank's K best | device |
| `ncclAllReduce<SUM>` of P segments × K keys → every rank merges the same K best overall | device |
| owners write their candidates' still-uncovered elements into payload slots | device |
| `ncclAllReduce<SUM>` of K slots × (bound + 2) words | device |
| walk; disable the accepted; score update through the inverted index | device |

The walk first enters every listed element into a small hash table; an element
entered twice is shared between candidates. A candidate with no shared element
is accepted without a barrier; the others count the elements already taken by
an accepted candidate. `bound` is the previous batch's last accepted score; K
is what fits a 32768-word payload at that bound (at most 256). One rank skips
both reductions.

### ByElement

Elements are split into contiguous shards, one per rank; each rank holds, for
*every* patch, the part inside its shard, and every rank holds every score. It
trades the replicated covered bitset (N / 8 bytes per rank in ByPatch) for
replicated scores (4 bytes × all patches per rank).

| Step | Where |
|---|---|
| send / apply the pending score decrements (delta exchange, one `ncclAllReduce<SUM>`) | device |
| refresh block maxima; select the K best (identical on every rank, no communication) | device |
| count shared uncovered elements per candidate pair inside this shard | device |
| `ncclAllReduce<SUM>` of the K × K table | device |
| walk; this shard's elements of the accepted → covered bits + newly covered; disable the accepted | device |
| new delta stream (the inverted-index lists of the newly covered elements) | device |

The delta exchange ships only what a selection changes — the patch slots whose
score drops — through a fixed-size window; a longer stream continues in the
following steps and nothing is selected until every rank has sent everything.
Its setup (incidence redistribution, sub-CSR) is on the device as well.

## UPS (device)

`src/ups/ups_pattern_stats.cu`: upload occurrences once → sort by hash and
reduce to one representative (lexicographic-min point) per local hash → the
PatchSet (built from the same device hash buffer) gives per-hash patch counts
from the inverted-index offsets → counts and representatives are routed to each
hash's owner shard (device exchanges) and reduced there → each shard's top k →
a k-sized merge on the host. The statistics file is fixed-width, so all ranks
write disjoint ranges of one file in parallel.

Outputs are byte-identical across rank counts.

## Limits

| What | Limit | Why |
|---|---|---|
| Patches on one rank | INT_MAX | positions are int on the device |
| Elements in one patch | INT_MAX | int32 scores |
| ByElement: patches over all ranks | INT_MAX | patch slots are 32-bit in its exchanges |
| ByPatch: ranks | 65535 | candidate key layout |
| UPS: occurrences on one rank / candidates received / shard size | INT_MAX | CUB sort and reduce-by-key take an int count (at 24 B+ per occurrence this is past a 100 GB GPU anyway) |

Host exchanges and the setup stages have no INT_MAX limit (64-bit counts).

## Performance

2×A100-SXM4-80GB, `NCCL_P2P_DISABLE=1`, synthetic input, 2M patches. USC total
(setup + solve), timed after the input is in memory:

| Input | Mode | 1 rank | 2 ranks |
|---|---|---|---|
| 32 per patch, overlap 0.5 (638530 selected) | ByPatch | 1.7 s | 2.0 s |
| | ByElement | 2.8 s | 4.3 s |
| 200 per patch, overlap 0.5 (123636 selected) | ByPatch | 2.1 s | 1.9 s |
| | ByElement | 2.8 s | 9.9 s |
| 32 per patch, overlap 0.9 (96136 selected) | ByPatch | 0.62 s | 0.54 s |
| | ByElement | 0.73 s | 1.53 s |

For scale: the first row took 53.6 s (1 rank) / 51.9 s (2 ranks) in ByPatch
before the 2026-11 work, and ByElement at 2 ranks took 545 s (2×A40).

Where a round goes at 2 ranks (2M patches, 2528 rounds, stage view with a
~10 µs synchronisation floor per stage): ByPatch — selection ~0.21 ms, walk
~0.15 ms, score update ~0.11 ms, the two reductions ~0.10 ms. ByElement —
delta reduce ~0.40 ms, pair-overlap counting ~0.29 ms, selection ~0.28 ms,
walk ~0.24 ms, overlap reduce ~0.21 ms.

Two ranks are not faster than one on this box: the work per round is small and
each reduction costs tens of microseconds. More ranks buy capacity (data that
does not fit one GPU), not speed, at these sizes.

## Verification

- `ctest` runs the GPU suite at 1 and 2 ranks (one rank per GPU; skipped where
  there are fewer GPUs) and a host-only suite at 1/2/4/8 ranks. CI builds
  everything and runs the host-only suite; the GPU suite needs a GPU runner.
- Every distributed result is compared with a rank-independent host reference,
  so running at several rank counts is the rank-invariance check.
  `STPS_TEST_SCALE_M` enlarges the scale tests (verified at 1M patches).
- Test hooks force the paths one GPU cannot reach otherwise: the ByElement
  delta exchange with tiny windows, and candidate caps from 1 to 1024.

**Not verified:** more than 2 ranks on GPUs; NVLink transport (P2P has been
disabled on every box used); real, non-synthetic data; behaviour when a rank
dies mid-run.

## Source layout

```
include/stPS/            public API (no MPI/NCCL/CUDA in the headers; pImpl)
  stPS.h  types.hpp  partition.hpp (PatchSlice, FlatPatches)
  usc_patch_selector.hpp  ups_pattern_stats.hpp  unique_hashes.hpp
src/core/                shared, algorithm-neutral
  patch_set.*            PatchSet + the host hash → ID mapping
  hash_id_map_device.*   the device hash → ID mapping
  inverted_index_device.* inverted_index.*   device builder / host reference
  device_primitives.*    radix sort, unique, lower_bound, owner expansion
  device_buffer.*  cub_run.hpp  cuda_check.hpp  collective.*  flat_patches.hpp
  host_profiler.*        PhaseTimer (-DUSC_PROFILE)
  cli_support.*          GPU pick per rank, abort-on-uncaught (executables only)
src/usc/                 usc_patch_selector_impl.{hpp,cu}  brute_force.*  main.cpp
src/ups/                 ups_pattern_stats.cu  unique_hashes.cpp  stats_file.cpp  main.cpp
src/io/                  patch_reader.hpp  stps_binary.cpp  parallel_file.*
src/data/                synthetic.*  partition.cpp
tests/                   test_main.cpp (GPU suite)  host_test_main.cpp (host-only suite)
  helpers/               lazy_greedy, local_setup, distribute, test_scale, …
```

## Build & run

```bash
scripts/setup-env.sh          # apt packages, cmake, stComm → ~/install/stComm

export CMAKE_PREFIX_PATH=$HOME/install/stComm:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$HOME/install/stComm/lib:$LD_LIBRARY_PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80   # 80 = A100
cmake --build build -j

export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export NCCL_P2P_DISABLE=1     # see "Known issues"
(cd build && ctest --output-on-failure)
#   -DSTPS_TEST_NRANKS="1;2;4;8" at configure time for more GPU ranks
#   STPS_TEST_SCALE_M=<patches> enlarges the *Scale* tests

mpirun -n 2 -x NCCL_P2P_DISABLE build/src/usc-patch-select --N 10000 --M 1000 --K 50 --seed 42
mpirun -n 2 -x NCCL_P2P_DISABLE build/src/usc-patch-select --input patches.stps --partition element
mpirun -n 2 -x NCCL_P2P_DISABLE build/src/ups-pattern-stats --input patches.stps --output stats.txt
```

### Smoke baseline

Deterministic for a given seed, in either mode and at any rank count:

```
usc-patch-select --N 20000 --M 3000 --K 40 --seed 42   →  selected=934 covered=14011
```

### Profiling

Configure with `-DUSC_PROFILE=ON`. `[stps-profile …]` tables on stderr split
setup and solve into phases tagged host / transfer / device. For the loops,
`STPS_LOOP_PROFILE=batch` (default) reports host enqueue time against the wait
for the stream, and `STPS_LOOP_PROFILE=stages` drains the stream after every
stage and bills the wait to it (each stage then carries a synchronisation
floor, so the sum exceeds the normal loop time).

## Known issues

- **Broken GPU-to-GPU P2P** (containers/VMs without NVLink): the first 2-rank
  NCCL send/receive hangs with both GPUs at 100%. Set `NCCL_P2P_DISABLE=1`. It
  is a runtime variable on purpose, so NVLink hosts keep the fast path.
  Details in HISTORY.md "Known issues".
- **MIG slices are not separate GPUs.** A process sees one MIG slice, so "x2"
  MIG instances of one card cannot host two ranks.
- **NCCL 2.25.1 fails on Blackwell-generation GPUs** (RTX PRO 4500 / 6000):
  `Cuda failure 1 'invalid argument'` in `ncclGroupEnd` at 2 ranks, for `main`
  as well. One rank works there.

## Design decisions still in force

1. **GPU is required** in device mode — no CPU fallback path in the product
   code; the host implementations exist as test references and for the
   host-only unique-hash export.
2. **All collective communication goes through stComm.** stPS wraps no MPI or
   NCCL call of its own.
3. **Nothing in a device loop may need a host scalar.** Root-and-count
   broadcasts became fixed-length reductions sized from values every rank
   already agrees on; that is what lets the host enqueue whole batches blind.
4. **A rank never throws alone.** Local problems are agreed collectively
   (`all_ranks_ok`) so every rank leaves together instead of stranding its
   peers in the next collective.
5. **Sizes are 64-bit** wherever the algorithm allows it; the remaining int
   ranges are listed under "Limits".
6. **Smart pointers / RAII for ownership**; `DeviceBuffer<T>` keeps
   `cuda_runtime.h` out of the headers.
