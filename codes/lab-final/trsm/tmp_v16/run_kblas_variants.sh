#!/bin/bash
# Decisive batch #4: KBLAS multithread scaling, variants, env knobs.
set -x
cd ~/trsm_work
source ./env.sh
export OMP_PROC_BIND=close OMP_PLACES=cores

echo "===== A. KBLAS 38-thread scaling on 4096^3 (OMP_NUM_THREADS sweep) ====="
for T in 2 4 8 16 24 32 38; do
  echo "--- T=$T ---"
  OMP_NUM_THREADS=$T numactl -N 1 ./dgemm_bench2 4096 4096 4096 3 1
done

echo "===== B. KBLAS 38-thread with OMP_WAIT_POLICY=PASSIVE ====="
OMP_NUM_THREADS=38 OMP_WAIT_POLICY=PASSIVE numactl -N 1 ./dgemm_bench2 4096 4096 4096 3 1

echo "===== C. KBLAS 38-thread with GOMP_SPINCOUNT + KMP_BLOCKTIME ====="
OMP_NUM_THREADS=38 GOMP_SPINCOUNT=0 KMP_BLOCKTIME=0 numactl -N 1 ./dgemm_bench2 4096 4096 4096 3 1

echo "===== D. HPCKit multi variant ====="
gcc -O3 -mcpu=native dgemm_bench2.c -o dgemm_multi -lm -L/work_ssd/software/HPCKit/25.2.1/kml/bisheng/lib/neon/kblas/multi -l:libkblas.so.25.2.1 -fopenmp
LD_LIBRARY_PATH=/work_ssd/software/HPCKit/25.2.1/kml/bisheng/lib/neon/kblas/multi:$LD_LIBRARY_PATH OMP_NUM_THREADS=38 numactl -N 1 ./dgemm_multi 4096 4096 4096 3 1
echo "--- multi single-thread ---"
LD_LIBRARY_PATH=/work_ssd/software/HPCKit/25.2.1/kml/bisheng/lib/neon/kblas/multi:$LD_LIBRARY_PATH OMP_NUM_THREADS=1 numactl -N 1 ./dgemm_multi 4096 4096 4096 3 1

echo "===== E. HPCKit nolocking variant ====="
gcc -O3 -mcpu=native dgemm_bench2.c -o dgemm_nolock -lm -L/work_ssd/software/HPCKit/25.2.1/kml/bisheng/lib/neon/kblas/nolocking -l:libkblas.so.25.2.1 -fopenmp
LD_LIBRARY_PATH=/work_ssd/software/HPCKit/25.2.1/kml/bisheng/lib/neon/kblas/nolocking:$LD_LIBRARY_PATH OMP_NUM_THREADS=38 numactl -N 1 ./dgemm_nolock 4096 4096 4096 3 1

echo "===== F. HPCKit 25.1.0.SPC001 sve multi variant ====="
gcc -O3 -mcpu=native dgemm_bench2.c -o dgemm_old -lm -L/work_ssd/software/HPCKit/25.1.0.SPC001/kml/bisheng/lib/sve/kblas/multi -l:libkblas.so.25.1.0.SPC001 -fopenmp 2>&1 | head -3
LD_LIBRARY_PATH=/work_ssd/software/HPCKit/25.1.0.SPC001/kml/bisheng/lib/sve/kblas/multi:$LD_LIBRARY_PATH OMP_NUM_THREADS=38 numactl -N 1 ./dgemm_old 4096 4096 4096 3 1

echo "===== G. contest KBLAS 38-thread small-shape (to confirm again) ====="
OMP_NUM_THREADS=38 numactl -N 1 ./dgemm_bench2 16800 512 224 3 1
echo "===== DONE ====="
