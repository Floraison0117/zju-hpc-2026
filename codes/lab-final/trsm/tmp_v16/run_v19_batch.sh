#!/bin/bash
# V19: V9 structure (scalar diag) + SME pipe-kernel M-split far updates.
set -x
cd ~/trsm_work
source ./env.sh
export OMP_NUM_THREADS=38
export OMP_PROC_BIND=close OMP_PLACES=cores
ASMFLAGS="-march=armv8.2-a+sve"

gcc -O3 $ASMFLAGS -c tmp_v16/sme_update_16x32_pipe.S -o tmp_v16/pipebench/pipe.o || exit 1

echo "===== 1. V19 Case3: scalar-diag + SME updates, NB sweep (gather pack) ====="
for NB in 224 448 896; do
  echo "--- NB=$NB gather ---"
  gcc -O3 -mcpu=native -DTRSM_USE_SME_UPDATE -DTRSM_NB3=$NB bench_trsm.c tmp_v16/trsm_v19.c tmp_v16/pipebench/pipe.o \
      -o tmp_v16/t19_nb${NB} -lm -l:libkblas.so.25.2.1 -fopenmp 2>&1 | head -3
  numactl -N 1 ./tmp_v16/t19_nb${NB} 17024 512 5
done

echo "===== 2. V19 Case3 NB sweep (scalar pack) ====="
for NB in 224 448; do
  echo "--- NB=$NB scalar-pack ---"
  gcc -O3 -mcpu=native -DTRSM_USE_SME_UPDATE -DTRSM_SCALAR_PACK -DTRSM_NB3=$NB bench_trsm.c tmp_v16/trsm_v19.c tmp_v16/pipebench/pipe.o \
      -o tmp_v16/t19s_nb${NB} -lm -l:libkblas.so.25.2.1 -fopenmp 2>&1 | head -3
  numactl -N 1 ./tmp_v16/t19s_nb${NB} 17024 512 5
done

echo "===== 3. V19 all three official cases (best NB) ====="
numactl -N 1 ./tmp_v16/t19_nb448 512 19968 5
numactl -N 1 ./tmp_v16/t19_nb448 2432 17024 5
numactl -N 1 ./tmp_v16/t19_nb448 17024 512 5

echo "===== 4. V19 Case2/1 with KBLAS (V9 recursion) sanity ====="
gcc -O3 -mcpu=native bench_trsm.c tmp_v16/trsm_v19.c tmp_v16/pipebench/pipe.o \
    -o tmp_v16/t19_base -lm -l:libkblas.so.25.2.1 -fopenmp
numactl -N 1 ./tmp_v16/t19_base 512 19968 5
numactl -N 1 ./tmp_v16/t19_base 2432 17024 5
numactl -N 1 ./tmp_v16/t19_base 17024 512 5
echo "===== DONE ====="
