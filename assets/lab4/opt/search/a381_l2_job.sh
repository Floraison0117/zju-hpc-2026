#!/bin/bash
# Iter38 A38-1 fused-z Level-2: 100-step OJ-sim + check.sh FINAL PASS.
set -uo pipefail
CAND=$(cut -d= -f2 ~/cand-a381-path.txt)
ROOT=/home/h3240101033/lab4-gpu
CACHE=/home/h3240101033/lab4-gpu-cand-p26bcd-combined-20260826-103614/twopuncture_cache
EV=$CAND/evidence/a381-l2
mkdir -p "$EV/runroot"; exec > "$EV/job.log" 2>&1
echo "=== START A381-L2 $CAND $(date -u) ==="
nvidia-smi -L 2>/dev/null | head -2

export OMP_NUM_THREADS=8 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ulimit -s unlimited
export AMSS_ENABLE_GPU=ON AMSS_ENABLE_TWOP_GPU=OFF AMSS_ENABLE_TWOP_OMP_TUNE=ON
export AMSS_ENABLE_PACKED_RELAX=ON AMSS_ENABLE_TWOP_COS_TABLE=ON
export AMSS_CUDA_ARCHITECTURES=80 AMSS_OPT="-O3"
export CUDACXX=/usr/local/cuda-13.3/bin/nvcc
export AMSS_BUILD_DIR="$CAND/build-a381"
export AMSS_OUTPUT_ROOT="$EV/runroot"
export AMSS_CACHE_DIR="$CACHE"

cd "$CAND"
grep -E "Final_Evolution_Time|Analysis_Time" AMSS_NCKU_Input.py
./run.sh > "$EV/run.log" 2>&1 || echo "RUN_EXIT=$?"
grep -E "This Program Cost|Total Evolve" "$EV/run.log" | tail -2
grep -E "After Step: (1|50|100) My Rank" "$EV/run.log" | tail -3

echo "=== check.sh ==="
export AMSS_OUTPUT_ROOT
./check.sh > "$EV/check.log" 2>&1 || echo "CHECK_EXIT=$?"
cat "$EV/check.log"
echo "=== DONE A381-L2 $(date -u) ==="
