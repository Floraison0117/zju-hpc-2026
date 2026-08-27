#!/bin/bash
# Decisive batch #7: custom SME kernel at large K (large-NB regime).
set -x
cd ~/trsm_work
source ./env.sh
export OMP_NUM_THREADS=38
export OMP_PROC_BIND=close OMP_PLACES=cores

BENCH=checkpoints/GateB-20260826-16x32/sme_gemm_tiled_bench
ls -la $BENCH 2>/dev/null || BENCH=trsm_sme_candidate_work_20260826/microbench/sme_gemm_tiled_bench
ls -la $BENCH 2>/dev/null

echo "===== 1. tiled bench K sweep (Case3 shape) ====="
for K in 256 512 1024 2048 4096; do
  echo "--- K=$K ---"
  numactl -N 1 $BENCH 16800 512 $K 1
done

echo "===== 2. tiled bench Case2 shape (N=17024) ====="
numactl -N 1 $BENCH 2432 17024 224 1
numactl -N 1 $BENCH 2432 17024 448 1
numactl -N 1 $BENCH 1216 17024 1216 1

echo "===== 3. tiled bench Case1 shape (N=19968) ====="
numactl -N 1 $BENCH 512 19968 256 1
numactl -N 1 $BENCH 512 19968 512 1

echo "===== 4. priv_gemm compute ceiling (rerun) ====="
gcc -O3 -mcpu=native tmp_v16/priv_gemm.c -o tmp_v16/priv_gemm -lm -l:libkblas.so.25.2.1 -fopenmp
numactl -N 1 ./tmp_v16/priv_gemm 2048 3
echo "===== DONE ====="
