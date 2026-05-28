# fullchipUSC

Distributed greedy minimum set cover for OPC segment dedup (USC: Unique Segment Correction).

Splits a semiconductor layout's full chip into patches, where each patch's segments carry hash values
based on local pattern context. This solver finds the **minimum number of patches whose union covers
every unique hash** — a classical Minimum Set Cover problem solved with the greedy approximation.

Design rationale lives in [`greedy_set_cover.md`](./greedy_set_cover.md).

## Status

| Milestone | Title | State |
|---|---|---|
| M1 | Single-rank end-to-end | ✅ |
| M2 | Multi-rank MPI via stComm | ✅ |
| M3 | Distributed sample sort setup (§5.1) + sparse bcast (§7.2) | ✅ |
| M4 | GPU port (CUDA kernels, CUB ArgMax) | ✅ on `gpu` branch |
| M5 | NCCL device-direct broadcast | ⏳ Planned |
| M6 | Element-partitioned covered bitset (§7.3) | ⏳ Planned |
| M7 | Real OPC input parser + 2D partition (§7.4) | ⏳ Planned |

Branches:
- `main` — CPU-only baseline (through M3). Stable regression reference.
- `gpu` — GPU work (M4+). M5+ continues here.

## Documentation

- **[STATUS.md](STATUS.md)** — what's been built, architecture summary, verification baseline
- **[ROADMAP.md](ROADMAP.md)** — M5+ planned work, new-environment setup, open decisions
- **[CONVENTIONS.md](CONVENTIONS.md)** — collaboration patterns and design preferences

## Quick build

```bash
# Prereqs: mpicxx, CUDA 12.x (for gpu branch), NCCL, stComm at ~/install/stComm
CMAKE_PREFIX_PATH=~/install/stComm cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Tests (size=4 multi-rank)
cd build && ctest --output-on-failure

# Smoke
mpirun -n 4 --oversubscribe ./build/src/fullchipusc-solve \
    --N 10000 --M 1000 --K 50 --overlap 0.4 --seed 42
# Expected: selected=353 covered=6019/6019 iterations=353
```

Detailed build / environment instructions: [STATUS.md](STATUS.md) and [ROADMAP.md](ROADMAP.md).
