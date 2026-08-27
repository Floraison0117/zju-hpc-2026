#!/bin/bash
# Decisive batch #5: real DRAM bandwidth + compute ceiling + NB sweep.
set -x
cd ~/trsm_work
source ./env.sh
export OMP_NUM_THREADS=38
export OMP_PROC_BIND=close OMP_PLACES=cores

echo "===== 1. STREAM (corrected) ====="
gcc -O3 -mcpu=native tmp_v16/stream_bench.c -o tmp_v16/stream_bench -fopenmp
numactl -N 1 ./tmp_v16/stream_bench 1024 0
numactl -N 1 ./tmp_v16/stream_bench 1024 2
numactl -N 1 ./tmp_v16/stream_bench 1024 3
numactl -N 1 ./tmp_v16/stream_bench 512 0

echo "===== 2. Compute ceiling: 38 private GEMMs ====="
gcc -O3 -mcpu=native tmp_v16/priv_gemm.c -o tmp_v16/priv_gemm -lm -l:libkblas.so.25.2.1 -fopenmp
numactl -N 1 ./tmp_v16/priv_gemm 2048 3
numactl -N 1 ./tmp_v16/priv_gemm 1024 3

echo "===== 3. Case3 NB sweep (M-split serial KBLAS) ====="
gcc -O3 -mcpu=native tmp_v16/shape_rate.c -o tmp_v16/shape_rate -lm -l:libkblas.so.25.2.1 -fopenmp
numactl -N 1 ./tmp_v16/shape_rate 17024 512 512 3 m
numactl -N 1 ./tmp_v16/shape_rate 17024 512 1024 3 m
numactl -N 1 ./tmp_v16/shape_rate 17024 512 2048 3 m
numactl -N 1 ./tmp_v16/shape_rate 16000 512 1024 3 m

echo "===== 4. Case2 M-split NB sweep ====="
numactl -N 1 ./tmp_v16/shape_rate 2432 17024 448 3 m
numactl -N 1 ./tmp_v16/shape_rate 2208 17024 448 3 m
numactl -N 1 ./tmp_v16/shape_rate 2432 17024 1216 3 m

echo "===== 5. Case1 M-split vs N-split ====="
numactl -N 1 ./tmp_v16/shape_rate 512 19968 256 3 m
numactl -N 1 ./tmp_v16/shape_rate 256 19968 256 3 m
numactl -N 1 ./tmp_v16/shape_rate 512 19968 128 3 m
echo "===== DONE ====="
