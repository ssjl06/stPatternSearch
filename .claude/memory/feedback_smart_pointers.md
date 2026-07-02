---
name: feedback-smart-pointers
description: "Host-side pointer ownership must go through smart pointers (std::unique_ptr / std::shared_ptr); raw owning pointers and `new`/`delete` are not allowed in fullchipUSC host code."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: fdcd8508-256e-4ae9-bb0a-1832057b94af
---

Host-side pointer management in fullchipUSC must use `std::unique_ptr` or
`std::shared_ptr`. Raw owning pointers and bare `new`/`delete` are not
allowed.

**Why:** User explicitly required this during M5 work after I added a raw
`stComm::NCCLComm*` member with a setter — they stopped me and asked to
use smart pointers instead. Treat this as a project-wide rule, not just
that one site.

**How to apply:**
- For optional / nullable host objects (e.g. the M5 `NCCLComm` injected
  into `USCSolver`), use `std::shared_ptr<T>` — the caller in `main.cpp`
  creates with `std::make_shared<T>()` and passes the shared_ptr in.
- For unique ownership, use `std::unique_ptr<T>` (move semantics).
- References (`T&`) are still fine for non-owning, non-nullable views
  that share the caller's lifetime — that's how `USCSolver` already
  holds `comm_` and should stay.
- Device-side pointers (`cudaMalloc` ranges) are out of scope for this
  rule; they're managed by the `DeviceBuffer<T>` RAII wrapper, not raw.
- Standard containers (`std::vector`, `std::unordered_map`, ...) already
  satisfy the rule — they own their storage with destructors.

Applies to host code in `src/` and `tests/`. The CUDA kernel TUs (.cu)
follow the same rule for host-side state; only the kernel arguments
themselves are raw device pointers because the CUDA API demands it.
