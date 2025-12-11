#!/usr/bin/env bash
set -e

echo "======================================="
echo "   EXAMM AUTO MPI FIX + REBUILD TOOL"
echo "======================================="

# ---- Change this to the EXACT repo path ----
EXAMM_DIR="$HOME/exact"
echo "[INFO] Using EXAMM source directory: $EXAMM_DIR"

if [[ ! -d "$EXAMM_DIR" ]]; then
    echo "[ERROR] EXAMM repo not found at: $EXAMM_DIR"
    echo "Please update EXAMM_DIR in this script."
    exit 1
fi

# --- Detect Homebrew prefix ---
if [[ -d "/opt/homebrew" ]]; then
    BREW_PREFIX="/opt/homebrew"
else
    BREW_PREFIX="/usr/local"
fi

echo "[INFO] Homebrew prefix: $BREW_PREFIX"

# --- Ensure OpenMPI installed ---
if ! brew list open-mpi &>/dev/null; then
    echo "[WARN] Installing OpenMPI..."
    brew install open-mpi
fi

MPI_DIR=$(brew --prefix open-mpi)
MPI_INCLUDE="$MPI_DIR/include"
MPI_LIB="$MPI_DIR/lib/libmpi.dylib"

echo "[INFO] MPI_DIR = $MPI_DIR"

# --- Export environment variables ---
export MPI_HOME="$MPI_DIR"
export PATH="$MPI_DIR/bin:$PATH"
export CPATH="$MPI_INCLUDE:$CPATH"
export LIBRARY_PATH="$MPI_DIR/lib:$LIBRARY_PATH"
export DYLD_LIBRARY_PATH="$MPI_DIR/lib:$DYLD_LIBRARY_PATH"

echo "[INFO] MPI environment set."

# --- Build EXAMM ---
echo "======================================="
echo "   CLEANING + BUILDING EXAMM"
echo "======================================="

cd "$EXAMM_DIR"
rm -rf build
mkdir build
cd build

cmake \
  -DMPI_C_INCLUDE_PATH="$MPI_INCLUDE" \
  -DMPI_C_LIBRARIES="$MPI_LIB" \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpicxx \
  ..

make -j$(sysctl -n hw.ncpu)

echo "======================================="
echo "   🎉 EXAMM BUILD SUCCESSFUL 🎉"
echo "   Executables in: $EXAMM_DIR/build/multithreaded"
echo "======================================="
