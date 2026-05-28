#!/usr/bin/env bash
# fullchipUSC environment bootstrap
#
# Installs the build prerequisites that aren't shipped with the repo and stages
# the stComm dependency so `cmake -S . -B build` can find it. Re-runnable: each
# step skips if its artifact already exists. CUDA Toolkit and the NVIDIA driver
# are NOT installed by this script — they're assumed to be present on the host.
#
# Usage:
#   scripts/setup-env.sh                       # default install paths
#   scripts/setup-env.sh --force-stcomm        # rebuild stComm even if installed
#   STCOMM_PREFIX=/opt/stComm scripts/setup-env.sh
#
# Tested on Ubuntu 24.04 / WSL2 / Runpod containers (root, no sudo needed).
# On a non-root host, run with `sudo` or pre-install the apt packages manually.

set -euo pipefail

# ----- configuration (override via env) -----
STCOMM_REPO_URL="${STCOMM_REPO_URL:-https://github.com/ssjl06/stComm.git}"
STCOMM_BRANCH="${STCOMM_BRANCH:-add-bcast-maxloc-exscan}"
STCOMM_SRC="${STCOMM_SRC:-$HOME/tickets/stComm}"
STCOMM_PREFIX="${STCOMM_PREFIX:-$HOME/install/stComm}"
FORCE_STCOMM=0

for arg in "$@"; do
    case "$arg" in
        --force-stcomm) FORCE_STCOMM=1 ;;
        -h|--help)
            sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m  %s\n' "$*"; }

# ----- 1. apt packages -----
# Note: cmake from apt on Ubuntu 22.04 is 3.22, which doesn't know
# CMAKE_CUDA_STANDARD 20 (needs 3.25+). We install cmake separately below.
need_apt=()
command -v mpicxx  >/dev/null 2>&1 || need_apt+=(openmpi-bin libopenmpi-dev)
[ -f /usr/include/nccl.h ]         || need_apt+=(libnccl2 libnccl-dev)
command -v git     >/dev/null 2>&1 || need_apt+=(git)
command -v wget    >/dev/null 2>&1 || need_apt+=(wget ca-certificates)
command -v curl    >/dev/null 2>&1 || need_apt+=(curl)
# stComm's CMake calls find_package(GTest) — apt provides prebuilt libs on Ubuntu 22.04+.
[ -f /usr/include/gtest/gtest.h ]  || need_apt+=(libgtest-dev libgmock-dev)

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

if [ "${#need_apt[@]}" -gt 0 ]; then
    log "apt installing: ${need_apt[*]}"
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get update -qq
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends "${need_apt[@]}"
else
    log "apt prerequisites already present"
fi

# ----- 1b. cmake >= 3.25 (project sets CMAKE_CUDA_STANDARD 20) -----
need_cmake=1
if command -v cmake >/dev/null 2>&1; then
    cmake_ver=$(cmake --version 2>/dev/null | head -1 | awk '{print $3}')
    cmake_major=${cmake_ver%%.*}
    cmake_minor=${cmake_ver#${cmake_major}.}; cmake_minor=${cmake_minor%%.*}
    if [ "${cmake_major:-0}" -gt 3 ] || \
       { [ "${cmake_major:-0}" -eq 3 ] && [ "${cmake_minor:-0}" -ge 25 ]; }; then
        need_cmake=0
        log "cmake $cmake_ver OK"
    else
        warn "cmake $cmake_ver is too old (need >= 3.25)"
    fi
fi

if [ "$need_cmake" -eq 1 ]; then
    CMAKE_VERSION="${CMAKE_VERSION:-3.29.6}"
    CMAKE_PREFIX_DIR="${CMAKE_PREFIX_DIR:-/opt/cmake}"
    log "installing cmake $CMAKE_VERSION to $CMAKE_PREFIX_DIR (from Kitware release)"
    tmp=$(mktemp -d)
    arch=$(uname -m)   # x86_64 or aarch64
    case "$arch" in
        x86_64|aarch64) ;;
        *) warn "unsupported arch $arch for cmake binary"; exit 1 ;;
    esac
    url="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${arch}.tar.gz"
    wget -q -O "$tmp/cmake.tgz" "$url"
    $SUDO mkdir -p "$CMAKE_PREFIX_DIR"
    $SUDO tar -xzf "$tmp/cmake.tgz" -C "$CMAKE_PREFIX_DIR" --strip-components=1
    $SUDO ln -sf "$CMAKE_PREFIX_DIR/bin/cmake"  /usr/local/bin/cmake
    $SUDO ln -sf "$CMAKE_PREFIX_DIR/bin/ctest"  /usr/local/bin/ctest
    $SUDO ln -sf "$CMAKE_PREFIX_DIR/bin/cpack"  /usr/local/bin/cpack
    rm -rf "$tmp"
    hash -r
    log "cmake now: $(cmake --version | head -1)"
fi

# ----- 1c. gh (GitHub CLI) for PR / issue workflows -----
if ! command -v gh >/dev/null 2>&1; then
    log "installing GitHub CLI (gh) from official apt repo"
    keyring=/usr/share/keyrings/githubcli-archive-keyring.gpg
    list=/etc/apt/sources.list.d/github-cli.list
    $SUDO mkdir -p -m 755 /etc/apt/keyrings
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | $SUDO dd of="$keyring" status=none
    $SUDO chmod go+r "$keyring"
    echo "deb [arch=$(dpkg --print-architecture) signed-by=$keyring] https://cli.github.com/packages stable main" \
        | $SUDO tee "$list" >/dev/null
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get update -qq
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends gh
fi
log "gh $(gh --version 2>/dev/null | head -1 | awk '{print $3}')"

# ----- 2. CUDA / nvcc sanity check (must be pre-installed) -----
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
if [ ! -x "$NVCC" ] && ! command -v nvcc >/dev/null 2>&1; then
    warn "nvcc not found at $NVCC and not in PATH."
    warn "Install CUDA Toolkit (12.x) or set NVCC=/path/to/nvcc before re-running."
    exit 1
fi
log "nvcc OK ($($NVCC --version 2>/dev/null | tail -1 || nvcc --version | tail -1))"

# ----- 3. detect GPU compute capability for CUDA_ARCH -----
if [ -z "${CUDA_ARCH:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        # "8.6" → "86"; if multiple GPUs, take the first.
        CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
        if [ -n "$CC" ]; then
            CUDA_ARCH="$CC"
        fi
    fi
fi
CUDA_ARCH="${CUDA_ARCH:-86}"   # safe default (RTX 3000/A4500 class)
export CUDA_ARCH
log "CUDA_ARCH=$CUDA_ARCH"

# ----- 4. stComm clone + build + install -----
if [ -d "$STCOMM_PREFIX/lib/cmake/stComm" ] && [ "$FORCE_STCOMM" -eq 0 ]; then
    log "stComm already installed at $STCOMM_PREFIX (use --force-stcomm to rebuild)"
else
    log "fetching stComm into $STCOMM_SRC ($STCOMM_BRANCH)"
    mkdir -p "$(dirname "$STCOMM_SRC")"
    if [ ! -d "$STCOMM_SRC/.git" ]; then
        git clone --branch "$STCOMM_BRANCH" "$STCOMM_REPO_URL" "$STCOMM_SRC"
    else
        git -C "$STCOMM_SRC" fetch origin "$STCOMM_BRANCH"
        git -C "$STCOMM_SRC" checkout "$STCOMM_BRANCH"
        git -C "$STCOMM_SRC" reset --hard "origin/$STCOMM_BRANCH"
    fi

    log "building stComm (CUDA_ARCH=$CUDA_ARCH)"
    pushd "$STCOMM_SRC" >/dev/null
    [ "$FORCE_STCOMM" -eq 1 ] && rm -rf build
    ./build.sh
    log "installing stComm to $STCOMM_PREFIX"
    ./install.sh "$STCOMM_PREFIX"
    popd >/dev/null
fi

# ----- 5. final summary -----
cat <<EOF

[setup] done. Next steps:

    export CMAKE_PREFIX_PATH=$STCOMM_PREFIX:\${CMAKE_PREFIX_PATH:-}
    export LD_LIBRARY_PATH=$STCOMM_PREFIX/lib:\${LD_LIBRARY_PATH:-}

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j

    cd build && ctest --output-on-failure
EOF
