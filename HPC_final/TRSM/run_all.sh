#!/bin/bash
# Run the three official TRSM cases with official parameters.
# Must run on a 鲲鹏 920F compute node (numactl -N 1, 38 OpenMP threads).
set -e
cd "$(dirname "$0")"
source ./env.sh
export OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores

echo "===== TRSM official cases (test_runs=5) ====="
echo "--- Case 1: 512 19968  (FLOPs = 5.234491392 GFLOP) ---"
numactl -N 1 ./trsm_test 512 19968 5
echo "--- Case 2: 2432 17024 (FLOPs = 100.690558976 GFLOP) ---"
numactl -N 1 ./trsm_test 2432 17024 5
echo "--- Case 3: 17024 512  (FLOPs = 148.386086912 GFLOP) ---"
numactl -N 1 ./trsm_test 17024 512 5
echo "===== done ====="
