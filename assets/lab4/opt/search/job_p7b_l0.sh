#!/usr/bin/env bash
# iter16 P7b Level-0 — sommerfeld kernels ptxas base probe (deployed state)
# Question: is there P6a-style unroll headroom in sommerfeld kernels?
# (fixed-loop unroll candidates, dependency chains, reg/spill/stack)
set -uo pipefail
BASE=/home/h3240101033/lab4-gpu
CAND=/home/h3240101033/lab4-gpu-cand-p7-20260824-175747
EV="$CAND/evidence/p7b-l0"
mkdir -p "$EV"
exec > "$EV/job-l0.log" 2>&1
echo "=== START P7b L0 $(date -u) ==="

NVCC=/usr/local/cuda-13.3/bin/nvcc
DEFS="-DMPI_CUDA_AWARE=0 -DUSE_GPU -Dfortran3 -Dnewc"
FLAGS="-O3 -Xptxas -v -std=c++14 --generate-code=arch=compute_80,code=[compute_80,sm_80] -rdc=true -lineinfo -maxrregcount=128"
INC_BASE="-I$BASE/src -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include"
INC_CAND="-I$CAND/src -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include"

for pair in "base:$BASE/src/sommerfeld_rout_gpu.cu:$INC_BASE" "cand:$CAND/src/sommerfeld_rout_gpu.cu:$INC_CAND" "fbase:$BASE/src/fmisc_gpu.cu:$INC_BASE"; do
  IFS=: read lbl src inc <<< "$pair"
  echo "=== compile $lbl ==="
  $NVCC -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++-13 $DEFS $inc $FLAGS -x cu -rdc=true -c "$src" -o "$EV/$lbl.o" > "$EV/ptxas-$lbl.log" 2>&1 || { echo "COMPILE_FAIL $lbl"; tail -5 "$EV/ptxas-$lbl.log"; }
done

echo "=== PTXAS sommerfeld kernels ==="
for lbl in base cand; do
  echo "--- $lbl ---"
  grep -A6 "Compiling entry function" "$EV/ptxas-$lbl.log" | grep -E "Compiling|stack|registers" 
done
echo "=== PTXAS fmisc (d_decide3d/d_polin3_1b/polint callers) ==="
grep -A6 "Compiling entry function" "$EV/ptxas-fbase.log" | grep -E "Compiling|stack|registers" | head -20

echo "=== SASS stats sommerfeld (base vs cand must be identical) ==="
for lbl in base cand; do
  /usr/local/cuda-13.3/bin/cuobjdump -sass "$EV/$lbl.o" > "$EV/sass-$lbl.txt" 2>/dev/null
  echo "$lbl sass lines: $(wc -l < "$EV/sass-$lbl.txt")"
done
cmp -s "$EV/sass-base.txt" "$EV/sass-cand.txt" && echo "SASS_IDENTICAL (sommerfeld untouched by P7a)" || echo "SASS_DIFFERS (unexpected!)"
echo "=== DONE P7b L0 $(date -u) ==="
