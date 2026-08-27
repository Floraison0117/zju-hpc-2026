#!/bin/bash
# Decisive batch #6: bandwidth pattern sweep + V17 large-NB variants.
set -x
cd ~/trsm_work
source ./env.sh
export OMP_NUM_THREADS=38
export OMP_PROC_BIND=close OMP_PLACES=cores

echo "===== 1. Bandwidth pattern: N sweep at fixed M,K ====="
gcc -O3 -mcpu=native tmp_v16/shape_rate.c -o tmp_v16/shape_rate -lm -l:libkblas.so.25.2.1 -fopenmp
numactl -N 1 ./tmp_v16/shape_rate 16800 512 224 3 m
numactl -N 1 ./tmp_v16/shape_rate 16800 1024 224 3 m
numactl -N 1 ./tmp_v16/shape_rate 16800 2048 224 3 m
numactl -N 1 ./tmp_v16/shape_rate 16800 4096 224 3 m
numactl -N 1 ./tmp_v16/shape_rate 16800 8192 224 3 m

echo "===== 2. Bandwidth pattern: K sweep at fixed M,N ====="
numactl -N 1 ./tmp_v16/shape_rate 16800 512 448 3 m
numactl -N 1 ./tmp_v16/shape_rate 16800 512 896 3 m
numactl -N 1 ./tmp_v16/shape_rate 16800 512 1792 3 m
numactl -N 1 ./tmp_v16/shape_rate 16800 512 3584 3 m

echo "===== 3. M sweep (does slice size matter?) ====="
numactl -N 1 ./tmp_v16/shape_rate 8400 512 224 3 m
numactl -N 1 ./tmp_v16/shape_rate 4200 512 224 3 m
numactl -N 1 ./tmp_v16/shape_rate 2100 512 224 3 m
numactl -N 1 ./tmp_v16/shape_rate 1050 512 224 3 m

echo "===== 4. V17 Case3: blocked-rec-diag NB sweep ====="
for NB in 224 448 896 1792; do
  echo "--- Case3 rec-diag NB=$NB leaf=64 ---"
  gcc -O3 -mcpu=native -DTRSM_CASE3_REC_DIAG -DTRSM_NB3=$NB -DTRSM_BLOCKED_REC_DIAG_LEAF=64 bench_trsm.c tmp_v16/trsm_v17.c -o tmp_v16/t17c3_nb${NB} -lm -l:libkblas.so.25.2.1 -fopenmp
  numactl -N 1 ./tmp_v16/t17c3_nb${NB} 17024 512 5
done

echo "===== 5. V17 Case2: blocked NB sweep ====="
for NB2 in 224 448 608; do
  echo "--- Case2 blocked NB2=$NB2 ---"
  gcc -O3 -mcpu=native -DTRSM_CASE2_BLOCKED -DTRSM_NB2=$NB2 -DTRSM_BLOCKED_REC_DIAG_LEAF=32 bench_trsm.c tmp_v16/trsm_v17.c -o tmp_v16/t17c2_nb${NB2} -lm -l:libkblas.so.25.2.1 -fopenmp
  numactl -N 1 ./tmp_v16/t17c2_nb${NB2} 2432 17024 5
done

echo "===== 6. V17 best combo on all three official cases ====="
numactl -N 1 ./tmp_v16/t17c3_nb896 512 19968 5
numactl -N 1 ./tmp_v16/t17c2_nb448 2432 17024 5
numactl -N 1 ./tmp_v16/t17c3_nb896 17024 512 5
echo "===== DONE ====="
