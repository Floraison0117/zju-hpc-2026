#!/bin/bash
# Iter313233 (P313233 combo) Level-2: 100-step full run + check.sh on the
# combined candidate (P31 global_interp var-batch + P32 sommerfeld compact +
# P33 prolong3 group multi-output).  Base for comparison: deployed 26bcd
# OJ-sim 605.78s (real OJ 601.364s); P33 single L2 = 579.66s.
set -uo pipefail
ROOT=$(ls -dt ~/lab4-gpu-cand-p313233-comb-* 2>/dev/null | head -1)
BASE=/home/h3240101033/lab4-gpu
cd "$ROOT" || { echo "NO CANDIDATE"; exit 1; }
export OMP_NUM_THREADS=8 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ulimit -s unlimited
EV="$ROOT/evidence/p313233-l2"
mkdir -p "$EV"; exec > "$EV/job.log" 2>&1
echo "=== START P313233_L2 $(date -u) ROOT=$ROOT ==="
nvidia-smi -L 2>/dev/null | head -2
export AMSS_ENABLE_GPU=ON AMSS_ENABLE_TWOP_GPU=OFF AMSS_ENABLE_TWOP_OMP_TUNE=ON
export AMSS_ENABLE_PACKED_RELAX=ON AMSS_ENABLE_TWOP_COS_TABLE=ON
export AMSS_CUDA_ARCHITECTURES=80 AMSS_OPT="-O3"
export AMSS_OUTPUT_ROOT="$ROOT" AMSS_CACHE_DIR="$BASE/twopuncture_cache"
if [ -x /usr/local/cuda-13.3/bin/nvcc ]; then export CUDACXX=/usr/local/cuda-13.3/bin/nvcc
elif [ -x /usr/local/cuda/bin/nvcc ]; then export CUDACXX=/usr/local/cuda/bin/nvcc; fi

# verify OJ config
grep -E "Final_Evolution_Time|Analysis_Time" AMSS_NCKU_Input.py | head -2

# clean build (no diagnostic flag)
export AMSS_BUILD_DIR="$ROOT/build-p313233-clean"
rm -rf "$AMSS_BUILD_DIR"
./compile.sh > "$EV/build.log" 2>&1 && echo "BUILD_OK" || { echo "BUILD_FAIL"; grep -iE "error" "$EV/build.log" | head -10; exit 3; }

# 100-step full run (TwoP live, no cache — matches OJ)
export AMSS_BUILD_DIR="$ROOT/build-p313233-clean"
rm -rf "$ROOT/GW250118"
./run.sh > "$EV/run.log" 2>&1 || echo "run exited"
grep -E "Total Evolve Time|This Program Cost|After Step: (1|50|100) " "$EV/run.log" | tail -5

# check.sh (bit-exact vs golden)
OUT="$ROOT/GW250118/AMSS_NCKU_output"
RESULT_DIR="$OUT" ./check.sh "$OUT" "$BASE/golden" > "$EV/check.log" 2>&1 || echo "check exited"
grep -iE "FINAL|PASS|FAIL|RMS|constraint maxima|Trajectory" "$EV/check.log" | tail -8
ls -la "$OUT"/*.dat 2>/dev/null | head
echo "=== DONE P313233_L2 $(date -u) ==="
