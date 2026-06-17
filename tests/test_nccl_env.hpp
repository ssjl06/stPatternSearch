#pragma once

namespace stComm { class Comm; }

namespace stPS::test_helpers {

// Access the process-global, device-enabled Comm that test_main.cpp builds at
// startup (via Comm::onDevice). All tests that construct a UscPatchSelector use this.
// The reference is valid only while test_main owns it — it is destroyed before
// MPI/NCCL finalize, which test_main handles.
stComm::Comm& comm();

}  // namespace stPS::test_helpers
