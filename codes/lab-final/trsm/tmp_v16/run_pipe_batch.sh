#!/bin/bash
# Decisive batch #8: pipelined SME kernel vs original, large-K sweep.
set -x
cd ~/trsm_work
source ./env.sh
export OMP_NUM_THREADS=38
export OMP_PROC_BIND=close OMP_PLACES=cores
ASMFLAGS="-march=armv8.2-a+sve"

mkdir -p tmp_v16/pipebench
gcc -O3 $ASMFLAGS -c tmp_v16/sme_update_16x32_pipe.S -o tmp_v16/pipebench/pipe.o
echo "pipe asm rc=$?"
gcc -O3 $ASMFLAGS -c tmp_v16/sme_update_16x32.S -o tmp_v16/sme_update_16x32.o
echo "orig asm rc=$?"
gcc -O3 -mcpu=native tmp_v16/sme_pipe_bench.c tmp_v16/pipebench/pipe.o tmp_v16/sme_update_16x32.o -o tmp_v16/pipebench/sme_pipe_bench -fopenmp -lm
echo "bench build rc=$?"

echo "===== 1. pipe vs orig, Case3 shape, K sweep ====="
for K in 224 512 1024 2048; do
  echo "--- K=$K ---"
  numactl -N 1 ./tmp_v16/pipebench/sme_pipe_bench 16800 512 $K 1
done

echo "===== 2. pipe vs orig, K=224, M sweep ====="
numactl -N 1 ./tmp_v16/pipebench/sme_pipe_bench 8400 512 224 1
numactl -N 1 ./tmp_v16/pipebench/sme_pipe_bench 4200 512 224 1

echo "===== 3. pipe vs orig, Case2 shape ====="
numactl -N 1 ./tmp_v16/pipebench/sme_pipe_bench 2432 17024 224 1
echo "===== DONE ====="
