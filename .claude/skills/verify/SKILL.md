---
name: verify
description: Build/launch/drive recipe for verifying stPatternSearch changes end-to-end (usc-patch-select, ups-pattern-stats)
---

# Verifying stPatternSearch changes

## Build

```bash
export CMAKE_PREFIX_PATH=$HOME/install/stComm:$CMAKE_PREFIX_PATH
cmake --build build -j 8          # incremental; targets: stPS, usc-patch-select, ups-pattern-stats, stps_tests
# clean configure (slow: ~155s configure + several min nvcc):
cmake -S . -B <dir> -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_PREFIX_PATH=$HOME/install/stComm
```

If the build errors on missing `stComm::Comm` members, the installed stComm
(`~/install/stComm`) is stale — update `~/tickets/stComm` to origin/main,
`cmake --build build -j 8 && cmake --install build --prefix ~/install/stComm`.

## Environment gotchas (this box: WSL2, 1× MX450)

- **NCCL cannot run 2 ranks on 1 GPU** — anything calling `Comm::onDevice`
  (usc-patch-select, ups-pattern-stats since UPS-2, stps_tests via test_main)
  is single-rank only here. Multi-rank NCCL verification needs the dual-GPU
  reference machine (`NCCL_P2P_DISABLE=1` on broken-P2P hosts).
- ctest runs the whole gtest suite as ONE mpiexec-wrapped entry
  (`-DSTPS_TEST_NRANKS=1` on this box; default 2 = the reference box).

## Drive

```bash
# USC smoke (deterministic; regression reference for a given seed):
mpirun -np 1 build/src/usc-patch-select --N 20000 --M 3000 --K 40 --seed 42
#   → selected=934 covered=14011 iterations=934

# UPS stats: strongest check is rank-count invariance — bodies must be
# byte-identical for -np 1/2/... on the same params (header differs by ranks=).
# NOTE: multi-rank needs multiple GPUs since UPS-2 (NCCL device-direct).
mpirun -np 1 build/src/ups-pattern-stats --N 5000 --M 2000 --K 20 \
    --output stats.txt --output-limit 50
sed 1d stats.txt   # 90-byte fixed lines: hash \t count \t x \t y

# File path round trip (ups --dump writes .stps v2 with coords):
mpirun -np 1 build/src/ups-pattern-stats --M 2000 --dump p.stps
mpirun -np 1 build/src/ups-pattern-stats --input p.stps --output s.txt
# body of s.txt == synthetic-path body for same params

# Independent cross-check: parse the .stps v2 dump in Python (struct: 8B magic
# "STPSPAT2", 3×u64 header, (M+1)×u64 offsets, K×u64 hashes, K×2×f64 coords),
# recompute per-hash patch counts + lexicographic-min coords, compare topmost
# stats.txt lines.
```
