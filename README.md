# fullchipUSC

Distributed greedy minimum set cover for OPC segment dedup (USC: Unique Segment Correction).

Splits a semiconductor layout's full chip into patches, where each patch's segments carry hash values
based on local pattern context. This solver finds the **minimum number of patches whose union covers
every unique hash** — a classical Minimum Set Cover problem solved with the greedy approximation.

Design rationale lives in [`greedy_set_cover.md`](./greedy_set_cover.md).

## Status — Milestone 1 (single-rank end-to-end)

- Single-process greedy solver (`size=1` MPI shim)
- CSR-backed patches + inverted index
- Synthetic data generator for testing
- Brute-force reference for correctness validation

Multi-rank MPI / GPU / NCCL ports are scheduled for milestones 2–5.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Requires CMake ≥ 3.22, a C++20 compiler, and MPI (OpenMPI or MPICH). Linux only.

## Test

```bash
ctest --test-dir build --output-on-failure
```

## Run

```bash
mpirun -n 1 ./build/src/fullchipusc-solve \
    --N 10000 --M 1000 --K 50 --overlap 0.4 --seed 42

# correctness comparison against brute-force reference (small instances)
mpirun -n 1 ./build/src/fullchipusc-solve \
    --N 500 --M 100 --K 20 --overlap 0.3 --seed 7 --brute-force
```
