#!/bin/bash
# Decisive batch #9: V18 end-to-end TRSM with SME pipe kernel.
set -x
cd ~/trsm_work
source ./env.sh
export OMP_NUM_THREADS=38
export OMP_PROC_BIND=close OMP_PLACES=cores
ASMFLAGS="-march=armv8.2-a+sve"

gcc -O3 $ASMFLAGS -c tmp_v16/sme_update_16x32_pipe.S -o tmp_v16/pipebench/pipe.o || exit 1

build_v18() {
  local name="$1" cf="$2"
  gcc -O3 -mcpu=native $cf bench_trsm.c tmp_v16/trsm_v18.c tmp_v16/pipebench/pipe.o \
      -o tmp_v16/$name -lm -l:libkblas.so.25.2.1 -fopenmp 2>&1 | head -3
}

echo "===== 1. V18 Case3: SME update + rec-diag, NB sweep ====="
for NB in 224 448 896 1792; do
  echo "--- NB=$NB leaf=64 ---"
  build_v18 t18_nb${NB} "-DTRSM_USE_SME_UPDATE -DTRSM_CASE3_REC_DIAG -DTRSM_NB3=$NB -DTRSM_BLOCKED_REC_DIAG_LEAF=64"
  numactl -N 1 ./tmp_v16/t18_nb${NB} 17024 512 5
done

echo "===== 2. V18 best on all three official cases ====="
build_v18 t18_all "-DTRSM_USE_SME_UPDATE -DTRSM_CASE3_REC_DIAG -DTRSM_NB3=896 -DTRSM_BLOCKED_REC_DIAG_LEAF=64"
numactl -N 1 ./tmp_v16/t18_all 512 19968 5
numactl -N 1 ./tmp_v16/t18_all 2432 17024 5
numactl -N 1 ./tmp_v16/t18_all 17024 512 5

echo "===== 3. V18 correctness with different NB/leaf on case3 ====="
build_v18 t18_c2 "-DTRSM_USE_SME_UPDATE -DTRSM_CASE3_REC_DIAG -DTRSM_NB3=448 -DTRSM_BLOCKED_REC_DIAG_LEAF=32"
numactl -N 1 ./tmp_v16/t18_c2 17024 512 5
echo "===== DONE ====="

echo "===== 4. V18 Case2: SME N-slice NB sweep ====="
for NB2 in 224 448 896; do
  echo "--- Case2 nslice NB2=$NB2 ---"
  gcc -O3 -mcpu=native -DTRSM_USE_SME_UPDATE -DTRSM_CASE2_SME_NSLICE -DTRSM_NB2=$NB2 -DTRSM_BLOCKED_REC_DIAG_LEAF=32 bench_trsm.c tmp_v16/trsm_v18.c tmp_v16/pipebench/pipe.o \
      -o tmp_v16/t18c2_ns$NB2 -lm -l:libkblas.so.25.2.1 -fopenmp 2>&1 | head -3
  numactl -N 1 ./tmp_v16/t18c2_ns$NB2 2432 17024 5
done

echo "===== 5. V18 Case1 with SME (N-split recursion + SME updates) ====="
gcc -O3 -mcpu=native -DTRSM_USE_SME_UPDATE bench_trsm.c tmp_v16/trsm_v18.c tmp_v16/pipebench/pipe.o \
    -o tmp_v16/t18c1 -lm -l:libkblas.so.25.2.1 -fopenmp 2>&1 | head -3
numactl -N 1 ./tmp_v16/t18c1 512 19968 5
echo "===== DONE ====="

echo "===== 6. V18 Case3 double-panel variants ====="
for NB in 448 896; do
  echo "--- double-panel NB=$NB leaf=64 ---"
  gcc -O3 -mcpu=native -DTRSM_USE_SME_UPDATE -DTRSM_CASE3_REC_DIAG -DTRSM_DOUBLE_PANEL -DTRSM_NB3=$NB -DTRSM_BLOCKED_REC_DIAG_LEAF=64 bench_trsm.c tmp_v16/trsm_v18.c tmp_v16/pipebench/pipe.o \
      -o tmp_v16/t18dp_nb${NB} -lm -l:libkblas.so.25.2.1 -fopenmp 2>&1 | head -3
  numactl -N 1 ./tmp_v16/t18dp_nb${NB} 17024 512 5
done
echo "===== DONE ====="
