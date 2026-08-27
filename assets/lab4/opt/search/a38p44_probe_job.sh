#!/bin/bash
# Iter38 A38-P44 L0 probe build: compile the 4x4x4 kernel (uncalled) with
# -DA38P44_PROBE -Xptxas -v; report regs/spill/stack + occupancy projection.
set -uo pipefail
CAND=$(cut -d= -f2 ~/cand-a38p44-path.txt)
EV=$CAND/evidence/p44probe
mkdir -p "$EV"; exec > "$EV/job.log" 2>&1
echo "=== START P44PROBE $CAND $(date -u) ==="

export OMP_NUM_THREADS=8 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ulimit -s unlimited
export AMSS_ENABLE_GPU=ON AMSS_ENABLE_TWOP_GPU=OFF AMSS_ENABLE_TWOP_OMP_TUNE=ON
export AMSS_ENABLE_PACKED_RELAX=ON AMSS_ENABLE_TWOP_COS_TABLE=ON
export AMSS_CUDA_ARCHITECTURES=80 AMSS_OPT="-O3"
export CUDACXX=/usr/local/cuda-13.3/bin/nvcc

cd "$CAND"
export AMSS_BUILD_DIR="$CAND/build-p44probe"
rm -rf "$AMSS_BUILD_DIR"
./compile.sh -DCMAKE_CUDA_FLAGS="-DA38P44_PROBE -Xptxas -v" > "$EV/build.log" 2>&1 \
  && echo BUILD_OK || { echo BUILD_FAIL; grep -iE "error" "$EV/build.log" | head -20; exit 3; }
echo "=== prolong3_multi4_probe_kernel ptxas ==="
grep -A10 "Function.*prolong3_multi4_probe_kernel" "$EV/build.log" | grep -iE "registers|spill|stack" | head -4
echo "=== reference (P33 boundary kernel, same build) ==="
grep -A10 "Function.*prolong3_multi_kernel\b" "$EV/build.log" | grep -iE "registers|spill|stack" | head -3
echo "=== occupancy projection (regs) ==="
grep -A10 "Function.*prolong3_multi4_probe_kernel" "$EV/build.log" | grep -iE "Used [0-9]+ registers" | head -2
echo "=== DONE P44PROBE $(date -u) ==="
