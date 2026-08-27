#!/usr/bin/env bash
# P6b compile-check — verify the fixed tree (fmisc.h include added) builds clean
set -uo pipefail
CAND=/home/h3240101033/lab4-gpu-cand-p6b-20260824-082335
EV="$CAND/evidence/p6b"
exec > "$EV/compile-check.log" 2>&1
echo "=== START compile-check $(date -u) ==="
NVCC=/usr/local/cuda-13.3/bin/nvcc
DEFS="-DMPI_CUDA_AWARE=0 -DUSE_GPU -Dfortran3 -Dnewc"
FLAGS="-O3 -Xptxas -v -std=c++14 --generate-code=arch=compute_80,code=[compute_80,sm_80] -rdc=true -lineinfo"
INC="-I$CAND/src -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include"
RC=0
for tu in fmisc_gpu.cu prolongrestrict_cell_gpu.cu sommerfeld_rout_gpu.cu derivatives.h; do
  if [ "$tu" = derivatives.h ]; then
    # derivatives.h is included by bssn_rhs_gpu.cu; compile that TU as a proxy
    tu2=bssn_rhs_gpu.cu
  else
    tu2=$tu
  fi
  echo "=== compile $tu2 ==="
  $NVCC -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++-13 $DEFS $INC $FLAGS -x cu -rdc=true -c "$CAND/src/$tu2" -o "$EV/cc-$tu2.o" > "$EV/cc-$tu2.log" 2>&1 || { echo "COMPILE_FAIL $tu2"; RC=1; tail -6 "$EV/cc-$tu2.log"; }
done
echo "RC=$RC"
echo "=== prolong3_kernel ptxas (fixed tree) ==="
grep -A6 "Compiling entry function.*prolong3_kernel" "$EV/cc-prolongrestrict_cell_gpu.cu.log" | grep -E "stack|registers"
echo "=== sommerfeld ptxas (fixed tree) ==="
grep -A6 "Compiling entry function" "$EV/cc-sommerfeld_rout_gpu.cu.log" | grep -E "stack|registers" | head -6
echo "=== DONE $(date -u) ==="
