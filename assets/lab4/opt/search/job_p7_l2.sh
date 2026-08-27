#!/usr/bin/env bash
# iter16 P7 Level-2 — restrict3 column-fused unroll full 100-step + check.sh
# cand runs in an isolated tree with default (100-step) input; output compared
# against formal golden dir. Reports This Program Cost + FINAL PASS.
set -uo pipefail
BASE=/home/h3240101033/lab4-gpu
CAND=/home/h3240101033/lab4-gpu-cand-p7-20260824-175747
EV="$CAND/evidence/p7-l2"
mkdir -p "$EV"
exec > "$EV/job-l2.log" 2>&1
echo "=== START P7 L2 $(date -u) ==="
nvidia-smi -L 2>/dev/null | head -2

export AMSS_ENABLE_GPU=ON AMSS_ENABLE_TWOP_GPU=OFF AMSS_ENABLE_TWOP_OMP_TUNE=ON
export AMSS_ENABLE_PACKED_RELAX=ON AMSS_ENABLE_TWOP_COS_TABLE=ON
export AMSS_CUDA_ARCHITECTURES=80 AMSS_OPT="-O3" AMSS_MPI_CUDA_AWARE=0

if [ ! -x "$CAND/build/ABEGPU" ]; then
  echo "=== BUILD cand ==="
  ( cd "$CAND" && rm -rf build && AMSS_BUILD_DIR="$CAND/build" JOBS=8 ./compile.sh -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc > "$EV/build-cand.log" 2>&1 ) || echo BUILD_FAIL_CAND
fi
ls -lh "$CAND/build/ABEGPU" 2>/dev/null || echo NO_CAND_BINARY

echo "=== sanity: default input Final_Evolution_Time ==="
grep -E "Final_Evolution_Time|Analysis_Time" "$CAND/AMSS_NCKU_Input.py" | head -4
# ensure the formal input is untouched (A/B job restored it)
grep -E "Final_Evolution_Time|Analysis_Time" "$BASE/AMSS_NCKU_Input.py" | head -4

export OMP_NUM_THREADS=8 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ulimit -s unlimited
mkdir -p "$EV/out/GW250118"
( cd "$CAND" && AMSS_BUILD_DIR="$CAND/build" AMSS_OUTPUT_ROOT="$EV/out" AMSS_CACHE_DIR="$BASE/twopuncture_cache" ./run.sh --twop-cache > "$EV/run-cand.log" 2>&1 || echo RUN_FAIL_CAND )

echo "=== This Program Cost ==="
grep "This Program Cost" "$EV/run-cand.log" | tail -2
echo "=== step timing (last 3) ==="
grep "After Step:" "$EV/run-cand.log" | tail -3

echo "=== CHECK vs formal golden ==="
( cd "$BASE" && ./check.sh "$EV/out/GW250118/AMSS_NCKU_output" "$BASE/golden" > "$EV/check.log" 2>&1 || true )
cat "$EV/check.log"
echo "=== DONE P7 L2 $(date -u) ==="
