#!/bin/bash
# Decisive batch #3: cache hierarchy + STREAM + shape rates (fixed paths).
set -x
cd ~/trsm_work
source ./env.sh
export OMP_NUM_THREADS=38
export OMP_PROC_BIND=close OMP_PLACES=cores

echo "===== 0. Cache hierarchy ====="
for i in 0 1 2 3 4; do
  d=/sys/devices/system/cpu/cpu0/cache/index$i
  [ -d "$d" ] && echo "index$i: $(cat $d/type) size=$(cat $d/size) level=$(cat $d/level) shared=$(cat $d/shared_cpu_list)"
done
echo "--- mem info ---"
grep -E 'MemTotal|MemFree' /proc/meminfo
echo "--- numa ---"
numactl -H 2>/dev/null | head -20
echo "--- freq ---"
cat /sys/devices/system/cpu/cpu40/cpufreq/scaling_cur_freq 2>/dev/null || echo "no cpufreq"

echo "===== 1. STREAM bandwidth ====="
gcc -O3 -mcpu=native tmp_v16/stream_bench.c -o tmp_v16/stream_bench -fopenmp
numactl -N 1 ./tmp_v16/stream_bench 1024 0
numactl -N 1 ./tmp_v16/stream_bench 1024 2
numactl -N 1 ./tmp_v16/stream_bench 256 0
numactl -N 1 ./tmp_v16/stream_bench 256 2

echo "===== 2. In-situ per-thread KBLAS serial rates ====="
gcc -O3 -mcpu=native tmp_v16/shape_rate.c -o tmp_v16/shape_rate -lm -l:libkblas.so.25.2.1 -fopenmp
# Case3 recursion level-1 update shape (M-split per thread, huge K)
numactl -N 1 ./tmp_v16/shape_rate 8512 512 8512 3 m
numactl -N 1 ./tmp_v16/shape_rate 4256 512 4256 3 m
# Case3 blocked NB=224 (V9 baseline shape)
numactl -N 1 ./tmp_v16/shape_rate 16800 512 224 3 m
numactl -N 1 ./tmp_v16/shape_rate 8400 512 224 3 m
numactl -N 1 ./tmp_v16/shape_rate 4200 512 224 3 m
# Case2 shapes (N-split per thread, current V9 structure)
numactl -N 1 ./tmp_v16/shape_rate 1216 448 1216 3 n
numactl -N 1 ./tmp_v16/shape_rate 608 448 608 3 n
# Case2 M-split alternative (shared N=17024)
numactl -N 1 ./tmp_v16/shape_rate 1216 17024 1216 3 m
numactl -N 1 ./tmp_v16/shape_rate 2432 17024 224 3 m
# Case1 shapes (N-split per thread)
numactl -N 1 ./tmp_v16/shape_rate 256 525 256 3 n
numactl -N 1 ./tmp_v16/shape_rate 128 525 128 3 n
echo "===== DONE ====="
