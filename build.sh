#!/bin/bash
# fullchipUSC (stPS) build script

set -e

BUILD_TYPE="Release"
CLEAN_BUILD=false
NUM_JOBS=$(nproc 2>/dev/null || echo 4)

# stComm is a build dependency. Point at its install prefix (overridable).
STCOMM_PREFIX="${STCOMM_PREFIX:-$HOME/install/stComm}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug) BUILD_TYPE="Debug"; shift ;;
        --clean) CLEAN_BUILD=true; shift ;;
        -j)      NUM_JOBS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--debug] [--clean] [-j NUM_JOBS]"; exit 1 ;;
    esac
done

if [ ! -d "${STCOMM_PREFIX}/lib/cmake/stComm" ]; then
    echo "Warning: stComm not found at ${STCOMM_PREFIX}."
    echo "Build/install stComm first, or set STCOMM_PREFIX=/path/to/stComm."
fi

# The test-discovery step runs the test binary, which loads libstComm.so.
export LD_LIBRARY_PATH="${STCOMM_PREFIX}/lib:${LD_LIBRARY_PATH}"

[ "$CLEAN_BUILD" = true ] && { echo "Cleaning build directory..."; rm -rf build; }

echo "Configuring with CMake (${BUILD_TYPE})..."
${CMAKE:-cmake} -S . -B build \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_PREFIX_PATH="${STCOMM_PREFIX}"

echo "Building with ${NUM_JOBS} jobs..."
${CMAKE:-cmake} --build build -j "${NUM_JOBS}"

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo "Library: build/src/libstPS.a"
echo "Driver:  build/src/fullchipusc-solve"
echo "Tests:   build/tests/fullchipusc_tests"
echo "=========================================="
