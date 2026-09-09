#!/usr/bin/env bash
set -euo pipefail

gcc -O3 bench_conv.c conv2d.c -o conv2d_test -lm -fopenmp

export OMP_NUM_THREADS=38
export OMP_DYNAMIC=false

numactl -N 1 ./conv2d_test 4096 6144 39 39 1
numactl -N 1 ./conv2d_test 6144 4096 41 41 1
numactl -N 1 ./conv2d_test 4256 6390 55 55 1
numactl -N 1 ./conv2d_test 6390 4256 81 81 1
