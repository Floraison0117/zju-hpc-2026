#!/bin/bash
# Build/run environment for TRSM on the 鲲鹏 920F cluster (NSCC-SZ, Donau scheduler).
#
# libkblas v25.2.1 (shenchao_common) is linked against the LLVM OpenMP runtime
# libomp (from BiSheng HPCKit 25.2.1); bench_trsm.c uses GCC libgomp via -fopenmp.
# The two OpenMP runtimes do not nest: the diagonal-block libgomp parallel region
# is always closed before the multithreaded KBLAS cblas_dgemm is issued.
TRSM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KBLAS_DIR=/home/share/shenchao_common/kblas            # kblas.h + libkblas.so.25.2.1
OMP_DIR=/work_ssd/software/HPCKit/25.2.1/compiler/bisheng/lib  # libomp.so
export CPATH="$TRSM_DIR:$KBLAS_DIR:${CPATH:-}"
export LIBRARY_PATH="$TRSM_DIR:$KBLAS_DIR:$OMP_DIR:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$TRSM_DIR:$KBLAS_DIR:$OMP_DIR:${LD_LIBRARY_PATH:-}"
