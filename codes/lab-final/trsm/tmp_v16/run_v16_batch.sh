#!/bin/bash
# Decisive batch #2: bandwidth, per-shape rates, V16 Case3-recursion variants.
set -x
cd ~/trsm_work
source ./env.sh
export OMP_NUM_THREADS=38
export OMP_PROC_BIND=close OMP_PLACES=cores

echo "===== 0. CPU/cache topology ====="
lscpu | grep -E 'Model name|Architecture|^CPU\(s\)|NUMA node\(s\)|L1d|L2|L3' 
grep -E '^processor|^model name|^flags' /proc/cpuinfo | head -4
echo "--- cache sizes per cpu ---"
lscpu -C | head -12

echo "===== 1. STREAM bandwidth (38 threads, numactl -N 1) ====="
gcc -O3 -mcpu=native stream_bench.c -o stream_bench -fopenmp
numactl -N 1 ./stream_bench 1024 0
numactl -N 1 ./stream_bench 1024 2
numactl -N 1 ./stream_bench 256 0

echo "===== 2. In-situ per-thread KBLAS serial rates ====="
gcc -O3 -mcpu=native shape_rate.c -o shape_rate -lm -l:libkblas.so.25.2.1 -fopenmp
# Case3 recursion level-1 update shape (M-split per thread)
numactl -N 1 ./shape_rate 8512 512 8512 3 m
numactl -N 1 ./shape_rate 4256 512 4256 3 m
numactl -N 1 ./shape_rate 2128 512 2128 3 m
numactl -N 1 ./shape_rate 1064 512 1064 3 m
# Case2 shapes (N-split per thread, current V9 structure)
numactl -N 1 ./shape_rate 1216 448 1216 3 n
numactl -N 1 ./shape_rate 608 448 608 3 n
# Case2 M-split alternative (shared N=17024)
numactl -N 1 ./shape_rate 1216 17024 1216 3 m
numactl -N 1 ./shape_rate 2432 17024 224 3 m
# Case1 shapes (N-split per thread)
numactl -N 1 ./shape_rate 256 525 256 3 n
numactl -N 1 ./shape_rate 128 525 128 3 n

echo "===== 3. V16: Case3 recursion variants on official case3 ====="
for LEAF in 32 64 128 224; do
  echo "--- Case3 recursive leaf=$LEAF ---"
  gcc -O3 -mcpu=native -DTRSM_CASE3_RECURSIVE -DTRSM_LEAF3=$LEAF bench_trsm.c tmp_v16/trsm_v16.c -o trsm_v16_l${LEAF} -lm -l:libkblas.so.25.2.1 -fopenmp 2>/dev/null || \
  gcc -O3 -mcpu=native -DTRSM_CASE3_RECURSIVE -DTRSM_LEAF3=$LEAF bench_trsm.c trsm_v16.c -o trsm_v16_l${LEAF} -lm -l:libkblas.so.25.2.1 -fopenmp
  numactl -N 1 ./trsm_v16_l${LEAF} 17024 512 5
done

echo "===== 4. V16 best on all three official cases ====="
numactl -N 1 ./trsm_v16_l64 512 19968 5
numactl -N 1 ./trsm_v16_l64 2432 17024 5
numactl -N 1 ./trsm_v16_l64 17024 512 5
echo "===== DONE ====="
