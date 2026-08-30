#!/bin/bash
set -euo pipefail

TRSM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TRSM_DIR"
source ./env.sh

CC="${CC:-gcc}"
CFLAGS="${CFLAGS:--O3 -mcpu=native -fopenmp}"
KBLAS_LIB="${KBLAS_LIB:--l:libkblas.so.25.2.1}"

# The contest image provides a versioned KBLAS library whose SONAME is
# libkblas.so.1.  Keep the loader workaround local to this directory.
if [[ ! -e "$TRSM_DIR/libkblas.so.1" && ! -L "$TRSM_DIR/libkblas.so.1" &&
      -f "$KBLAS_DIR/libkblas.so.25.2.1" ]]; then
    ln -s "$KBLAS_DIR/libkblas.so.25.2.1" "$TRSM_DIR/libkblas.so.1"
fi

"$CC" $CFLAGS bench_trsm.c trsm.c -o trsm_test -lm "$KBLAS_LIB"

RUNS="${TRSM_TEST_RUNS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-38}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

echo "===== TRSM tests: runs=$RUNS ====="
numactl -N 1 ./trsm_test 512 19968 "$RUNS"
numactl -N 1 ./trsm_test 2432 17024 "$RUNS"
numactl -N 1 ./trsm_test 17024 512 "$RUNS"
