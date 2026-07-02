---
name: feedback-gpu-required
description: fullchipUSC assumes a GPU environment. Do not add CPU/host-stage fallbacks or optional GPU paths in production code — make GPU resources required and fail loud if missing.
metadata: 
  node_type: memory
  type: feedback
  originSessionId: fdcd8508-256e-4ae9-bb0a-1832057b94af
---

fullchipUSC assumes a GPU-enabled environment is always available. Do
not add CPU fallbacks, host-stage alternative paths, or optional-GPU
toggles in production code.

**Why:** User stopped me during M5 when I was about to wire NCCL as an
*optional* feature with a host-staged MPI fallback branch behind it.
Their guidance: the project is GPU-first by premise, so optionality
just adds branching, dead code, and configuration surface. If GPUs (or
NCCL) aren't available, fail at startup — don't degrade silently.

**How to apply:**
- Constructors should *require* GPU resources (e.g. `NCCLComm` as a
  shared_ptr ctor arg), not accept them via optional setters.
- Don't keep an "M4 host-staged path" branch alongside the NCCL path
  "just in case." Delete the fallback when promoting NCCL device-direct
  bcast to the production path.
- CLI flags like `--backend nccl` / `--backend mpi` are pointless when
  there's only one supported backend. Don't add them.
- Test fixtures should `GTEST_SKIP` (or fail) when GPU count is
  insufficient, never silently fall back to a CPU path.
- The CPU-only M3 baseline still exists on `main` branch for git-bisect
  regression hunting — that's the only "fallback" we need.

Applies to: `src/core/usc_solver.cu`, `src/main.cpp`, test fixtures, and
any future communication or compute path. The CPU-resident `brute_force`
in `tests/` is fine — it's a correctness oracle, not a fallback.
