#!/bin/bash
# fullchipUSC (stPS) installation script

set -e

DEFAULT_PREFIX="/usr/local"
INSTALL_PREFIX="${1:-$DEFAULT_PREFIX}"

echo "=========================================="
echo "Installing stPS to: ${INSTALL_PREFIX}"
echo "=========================================="

if [ ! -d "build" ]; then
    echo "Build directory not found. Running build first..."
    ./build.sh
fi

${CMAKE:-cmake} --install build --prefix "${INSTALL_PREFIX}"

echo ""
echo "=========================================="
echo "Installation completed successfully!"
echo "=========================================="
echo "Headers: ${INSTALL_PREFIX}/include/stPS/"
echo "Library: ${INSTALL_PREFIX}/lib/libstPS.a"
echo "CMake:   ${INSTALL_PREFIX}/lib/cmake/stPS/"
echo ""
echo "To use stPS in your CMake project, add:"
echo "  find_package(stPS REQUIRED)   # also pulls in stComm, MPI, CUDA"
echo "  target_link_libraries(your_target PRIVATE stPS::stPS)"
echo ""
echo "Then, in code:"
echo "  #include <stPS/stPS.h>"
echo "  stPS::UscPatchSelector selector(comm);     // comm = stComm::Comm::onDevice(...)"
echo "  auto result = selector.patch_select(patches, ids);"
echo "=========================================="
