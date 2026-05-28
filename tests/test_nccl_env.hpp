#pragma once

#include <memory>

namespace stComm { class NCCLComm; }

namespace fullchipusc::test_helpers {

// Access the process-global NCCL communicator that test_main.cpp initialized
// at startup. All tests that build a USCSolver use this. The returned
// shared_ptr is co-owned with the test_main singleton; release it before
// MPI/NCCL finalize (which test_main handles).
std::shared_ptr<stComm::NCCLComm> nccl_comm();

}  // namespace fullchipusc::test_helpers
