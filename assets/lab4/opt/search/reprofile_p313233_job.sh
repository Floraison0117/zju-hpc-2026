#!/bin/bash
# Iter38 reprofile: nsys full 100-step ledger on deployed P313233 state (OJ-sim 540.23s).
#   Purpose: calibrate the new baseline module ledger (rhs dual-kernel current
#   values after 26a/26b/26bcd, global_interp after P31 varbatch, prolong3 after
#   P33 2x2x2 group, sommerfeld after P32 compact), verify three-window share
#   stability, and identify new hot spots for the >90pt push.
#   OJ-sim config: Final=100, Analysis=0.1, Dissipation=0.15, live TwoP (no cache).
set -uo pipefail
ROOT=/home/h3240101033/lab4-gpu
TS=$(date +%Y%m%d-%H%M%S)
EV=$ROOT/evidence/reprofile-p313233-$TS
mkdir -p "$EV/runroot"
exec > "$EV/job.log" 2>&1
echo "=== START REPROFILE $(date -u) TS=$TS ==="
nvidia-smi -L 2>/dev/null | head -2

export OMP_NUM_THREADS=8 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ulimit -s unlimited
export AMSS_ENABLE_GPU=ON AMSS_ENABLE_TWOP_GPU=OFF AMSS_ENABLE_TWOP_OMP_TUNE=ON
export AMSS_ENABLE_PACKED_RELAX=ON AMSS_ENABLE_TWOP_COS_TABLE=ON
export AMSS_CUDA_ARCHITECTURES=80 AMSS_OPT="-O3"
export CUDACXX=/usr/local/cuda-13.3/bin/nvcc
export AMSS_BUILD_DIR="$ROOT/build-reprofile"
export AMSS_OUTPUT_ROOT="$EV/runroot"
export AMSS_CACHE_DIR="$ROOT/twopuncture_cache"
NSYS_BIN=$(command -v nsys)

cd "$ROOT"
rm -rf "$AMSS_BUILD_DIR"
./compile.sh > "$EV/build.log" 2>&1 && echo BUILD_OK || { echo BUILD_FAIL; grep -iE "error" "$EV/build.log" | head -20; exit 3; }
ls -la "$AMSS_BUILD_DIR/ABEGPU" 2>/dev/null

echo "=== nsys profile (full 100-step OJ-sim) ==="
$NSYS_BIN profile -o "$EV/reprofile" --force-overwrite=true --trace=cuda --cuda-memory-usage=false --sample=none \
  ./run.sh > "$EV/nsys-run.log" 2>&1
echo "nsys run rc=$?"
grep -E "This Program Cost|Total Evolve" "$EV/nsys-run.log" | tail -3
grep -E "After Step: (1|50|100) My Rank" "$EV/nsys-run.log" | tail -3

echo "=== nsys stats ==="
$NSYS_BIN stats --report=cuda_gpu_kern_sum --format=csv --output="$EV/stats-kern" "$EV/reprofile.nsys-rep" > "$EV/stats-kern.log" 2>&1; echo "kern rc=$?"
$NSYS_BIN stats --report=cuda_api_sum --format=csv --output="$EV/stats-api" "$EV/reprofile.nsys-rep" > "$EV/stats-api.log" 2>&1; echo "api rc=$?"

echo "=== nsys export sqlite ==="
$NSYS_BIN export -t sqlite -o "$EV/reprofile.sqlite" "$EV/reprofile.nsys-rep" > "$EV/export.log" 2>&1
echo "export rc=$?"
ls -la "$EV/" | head -20
echo "=== DONE REPROFILE $(date -u) ==="
