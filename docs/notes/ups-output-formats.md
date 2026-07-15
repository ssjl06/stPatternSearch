# Note: UPS stats output — parallel file formats (discussion record, 2026-07-15)

Context: UPS-2 landed the fixed-width parallel text writer
(`write_pattern_stats_file`). This note records the format discussion so the
reasoning isn't re-derived when the output grows beyond top-K.

## What we have and why it works

One text file, all ranks writing in parallel:

- Records are **fixed-width** (90 B: hash 16-hex, count 20, x/y `%25.17e`), so
  every byte offset is a pure function of the line index — ranks `pwrite`
  disjoint ranges with zero communication; rank 0 truncates stale bytes after
  a barrier.
- This trick **only exists because records are fixed-width** and the payload
  is tiny (top-K, hundreds of lines). Human-readable, greppable, zero
  dependencies. For this use case the "parallelism" is a design nicety, not a
  throughput need.
- Assumes a POSIX-coherent shared filesystem (local disk, Lustre, GPFS).

Verified all-ranks participation: pre-fill the file with exact-final-size
garbage → run `-np 4` → zero garbage bytes survive, while each rank's write
buffer only ever contains its own line range.

## When this stops being the right tool

The moment UPS output becomes a **full dump** (all N patterns, or per-pattern
occurrence lists — potentially billions of rows), fixed-width text is wrong on
every axis: size, write bandwidth, and downstream consumption.

## Options compared

| Option | Parallel-write story | Notes |
|---|---|---|
| **pHDF5** | True cooperative single-file writes via the MPI-IO driver (collective buffering / aggregators on Lustre/GPFS) | Correction recorded: HDF5's parallelism is **MPI-process-based (pHDF5)**, not multithread — the library serializes behind one global lock even in thread-safe builds. Parallel writes + compression filters don't mix well. Heavy-ish dependency. |
| **Parquet** (Arrow C++) | Single-file cooperative writes are **structurally impossible** (row-group metadata lives in a single footer). The ecosystem pattern is **one part file per rank** (`stats/part-0000.parquet`, ...) — the directory *is* the dataset for Spark/Dask/DuckDB/pandas. | Drop the "single file" constraint and the parallel-write problem disappears entirely; columnar (hash u64, count u64, x/y f64) fits perfectly and compresses well. Moderate dependency. |
| **libcudf** (RAPIDS, C++ core of cuDF) | Per-rank part files, same as Parquet — but `cudf::io::write_parquet` encodes **directly from device memory**. | cuDF is a GPU DataFrame *library*, not a format. Since UPS-2 the shard stats already live on device (`d_shard_cnt`, `d_shard_pts`), so this removes the D2H copy; KvikIO/GDS can go storage-direct. Heavy dependency (rapids-cmake/conda); fine on CC ≥ 6.0. |
| **Raw MPI-IO binary** (.stps-style CSR) | Single file, collective writes, maximum throughput | No ecosystem tooling; right if the only consumer is our own reader. |

## Recommendation (agreed 2026-07-15)

- **Top-K stats (current)**: keep the fixed-width text writer. Right-sized.
- **Full dump (future)**: **Parquet part-per-rank via Arrow C++ is the
  first choice** — parallelism becomes trivial and the analysis ecosystem
  (DuckDB/pandas) comes for free. **libcudf device-direct writing is the
  natural follow-up optimization** on top of the same layout, not a separate
  direction. pHDF5 / raw MPI-IO only if a hard single-file requirement
  appears.
