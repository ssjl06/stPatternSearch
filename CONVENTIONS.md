# stPatternSearch — Collaboration Conventions

Notes for future Claude sessions (and human collaborators) about how this
project has been developed. Patterns observed across M1–M4 that should
continue.

## Communication style with the user

- **Korean is the default response language.** Use English only for code,
  commit messages, and inline comments.
- **Explain before editing.** New patterns, non-trivial design choices, or
  any unfamiliar C++/CUDA/MPI idiom should be explained with brief
  examples *before* writing the code. The user will ask follow-up questions
  ("이게 뭐지?", "왜 필요하지?") — answer them clearly, then re-ask
  confirmation to proceed.
- **Use trade-off tables and short option lists.** When there's a decision
  to make, lay out 2–3 options with pros/cons in a table and clearly mark
  the recommendation with "(권장)". Use `AskUserQuestion` for genuine
  binary decisions; don't multiply choices artificially.
- **Cite design doc sections.** `greedy_set_cover.md` is the source of
  truth for algorithmic decisions. Reference sections (§5.1, §6.2, etc.)
  when implementing related code.

## Design preferences (from user feedback)

### 1. All communication code lives in stComm
- stPS must not wrap MPI/NCCL calls in its own classes. There was
  briefly a `Communicator` wrapper in M1; M2 removed it.
- If a comm protocol is missing from stComm, add it to stComm (upstream
  PR), don't reinvent in stPS.
- The one allowed pattern: USCSolver holds a `CommT&` (template) and calls
  `comm_.bcast<T>(...)`, `comm_.allreduceMaxloc<T>(...)` directly.

### 2. USCSolver is algorithm-pure
- No MPI lifecycle methods on USCSolver (`initialize_mpi` / `finalize_mpi`).
  Those go in `main()`.
- No `rank()` / `size()` accessors on USCSolver. If `main()` needs them,
  it keeps its own `stComm::MPIComm world` handle.
- USCSolver only exposes `load`, `setup`, `solve`, `print_solution`,
  and algorithm-state inspectors (N, M_local, patches, inverted_index).

### 3. Header hygiene
- Public headers (`*.hpp`) should not include CUDA headers when possible.
  When CUDA types are needed in members, split: a CUDA-free `*.hpp` plus
  a `.cu` implementation. The `DeviceBuffer` / `DeviceBufferImpl` pair is
  the canonical example.
- Templates: explicit instantiation in `.cu` / `.cpp` (`template class
  USCSolver<stComm::MPIComm>;`), not header-only template. Methods,
  including trivial getters, live in the .cu/.cpp.

### 4. No premature abstraction (YAGNI)
- No globals. No lazy singletons. No pimpl unless you can name the concrete
  reason (e.g., hiding CUDA from a header). The user has repeatedly chosen
  the simpler value-semantic option.
- Don't create directories for a single file. `src/main.cpp` was at
  `src/cli/main.cpp` briefly; M4 flattened it.
- Test-only code lives in `tests/helpers/`, not `src/core/`.

### 5. Synthetic data is for tests, but pragmatic
- `generate_synthetic` lives in `src/data/` even though it's semantically
  "test scaffolding", because the app at this stage has no other input
  source. When a real-data reader exists (M7+), revisit.
- `slice_patches_by_rank` lives in `src/data/` because it's a general data
  distribution utility (will be reused by real loaders).

### 6. Determinism + bit-identical equivalence
- The solver result must be **bit-identical** between size=1 and size>1,
  and against `solve_brute_force` on the same input. `test_solver_equivalence.cpp`
  enforces this — keep it green at every commit.
- MAXLOC tie-breaking is smaller rank wins (MPI_MAXLOC semantics).
- CUB ArgMax tie-breaking is also smaller index — verified to match.
- atomicAdd on the build_newly_covered_kernel makes order non-deterministic
  but the *set* is identical; downstream consumers (atomicOr) are order-
  insensitive, so the global result stays deterministic.
- After M5, the verification matrix is `size=1` vs `size=N` where N = visible
  GPUs (no MPS / oversubscription — NCCL needs 1:1 process:device mapping).

### 7. GPU is required, no fallbacks
- stPS is GPU-first by premise. Don't add CPU fallbacks, optional-GPU
  toggles, or `--backend mpi` flags. If GPUs (or NCCL) aren't available, fail
  loud at startup.
- The CPU-only M3 baseline still lives on `main` branch for bisect-style
  regression hunting — that's the only "fallback" we need.
- Required GPU resources (e.g. `NCCLComm`) are ctor arguments, not optional
  setters; constructing the solver without them is a bug.
- Test fixtures `GTEST_SKIP` (or fail) when `num_gpus < world.size()`,
  never silently fall back to a host-only path.

### 8. Smart-pointer-only host ownership
- Host-side optional or owned objects use `std::shared_ptr` / `std::unique_ptr`.
  No raw owning pointers, no bare `new` / `delete`.
- References (`T&`) are still fine for non-owning, non-nullable views that
  share the caller's lifetime — that's how `USCSolver::comm_` works.
- Device-side allocations (`cudaMalloc` ranges) are managed by `DeviceBuffer<T>`
  RAII; this rule covers host-side state only.
- Standard containers (`std::vector`, `std::unordered_map`, …) already
  satisfy the rule.

## Workflow patterns

### Iterative milestone commits
Each milestone is split into 3–8 small commits with `M{N}-{step}: ...`
prefix. Each commit:
- Has a single conceptual change.
- Passes `ctest` (run between commits) — no broken intermediate states.
- Includes a commit message with verification result ("27/27 pass,
  selected=353 bit-identical").

This makes git bisect cheap when regressions appear.

### Tasks tracked via TaskCreate/Update
Each milestone declares all its sub-tasks up front via `TaskCreate`. The
agent marks `in_progress` before starting a step and `completed` after
verification passes. The user can see live progress in the spinner.

### Build verification matrix
- `mpirun -n 1 ctest` — single-rank correctness
- `mpirun -n <num_gpus> ctest` — multi-rank correctness (NCCL needs 1
  process per GPU, no oversubscription). Default ctest is `-n 2`; override
  via `-DSTPS_TEST_NRANKS=<n>`.
- `mpirun -n <num_gpus> ... usc-patch-select --N ...` — smoke at scale.
- Configure with `-DUSC_PROFILE=ON`, then `mpirun -n <num_gpus> ... usc-patch-select ...`
  — per-stage breakdown when investigating regressions or new bottlenecks.

Smoke output baseline lives in [STATUS.md](STATUS.md). When changing
hot-path code, verify all three before committing.

### When stuck on a comm library issue
- Read stComm's actual headers (`~/install/stComm/include/stComm/`) for
  the API; the README is condensed but the headers are authoritative.
- If stComm is missing a primitive, add it upstream — the project owner
  (ssjl06) is also the user, so PR review is quick.

## Anti-patterns to avoid (lessons learned)

- ❌ "Just-in-case" wrapper classes around stComm (M1 Communicator removed)
- ❌ Globals or lazy singletons for MPI/CUDA handles (`world_handle()` removed)
- ❌ Mixed CPU/GPU code paths in one class to "preserve fallback" (rollback
  is what git is for; the CPU baseline lives on `main` branch)
- ❌ Inline implementations in headers when they're not trivial
- ❌ Implementation-detail directories with one file (`src/cli/` flattened)
- ❌ Test helpers leaking into `src/core/`
- ❌ Adding `dense vs sparse adaptive bcast` (§7.2 adaptive part) when the
  break-even is past our scale — keep sparse-only until measurement says
  otherwise

## How to onboard in a new environment

1. Read this file (CONVENTIONS.md) first.
2. Read [STATUS.md](STATUS.md) for current state.
3. Read [ROADMAP.md](ROADMAP.md) for what's planned next.
4. Skim [`greedy_set_cover.md`](greedy_set_cover.md) §5–§7 for algorithmic
   context.
5. Check out the right branch: `gpu` for current work, `main` for CPU
   reference.
6. Build per STATUS.md, run ctest, confirm baseline matches.
7. Pick a milestone from ROADMAP.md and proceed iteratively.
