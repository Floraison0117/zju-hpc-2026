#!/usr/bin/env bash
# iter15 P6b PROBE — forceinline d_symmetry_bd_1b: ptxas + SASS base vs cand
set -uo pipefail
BASE=/home/h3240101033/lab4-gpu
CAND=/home/h3240101033/lab4-gpu-cand-p6b-20260824-082335
EV="$CAND/evidence/p6b"
mkdir -p "$EV"
exec > "$EV/job-l0.log" 2>&1
echo "=== START P6b L0 $(date -u) ==="
nvidia-smi -L 2>/dev/null | head -2

NVCC=/usr/local/cuda-13.3/bin/nvcc
DEFS="-DMPI_CUDA_AWARE=0 -DUSE_GPU -Dfortran3 -Dnewc"
FLAGS="-O3 -Xptxas -v -std=c++14 --generate-code=arch=compute_80,code=[compute_80,sm_80] -rdc=true -lineinfo"
INC_BASE="-I$BASE/src -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include"
INC_CAND="-I$CAND/src -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include"

for pair in "base:$BASE/src/prolongrestrict_cell_gpu.cu:$INC_BASE" "cand:$CAND/src/prolongrestrict_cell_gpu.cu:$INC_CAND"; do
  IFS=: read lbl src inc <<< "$pair"
  echo "=== compile $lbl (prolongrestrict) ==="
  $NVCC -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++-13 $DEFS $inc $FLAGS -x cu -rdc=true -c "$src" -o "$EV/$lbl.o" > "$EV/ptxas-$lbl.log" 2>&1 || { echo "COMPILE_FAIL $lbl"; tail -8 "$EV/ptxas-$lbl.log"; }
done
# fmisc_gpu.cu must compile too (d_symmetry_bd_1b/f_at_1b removed there in cand)
for pair in "base:$BASE/src/fmisc_gpu.cu:$INC_BASE" "cand:$CAND/src/fmisc_gpu.cu:$INC_CAND"; do
  IFS=: read lbl src inc <<< "$pair"
  echo "=== compile $lbl (fmisc_gpu) ==="
  $NVCC -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++-13 $DEFS $inc $FLAGS -x cu -rdc=true -c "$src" -o "$EV/fmisc-$lbl.o" > "$EV/ptxas-fmisc-$lbl.log" 2>&1 || { echo "COMPILE_FAIL fmisc-$lbl"; tail -8 "$EV/ptxas-fmisc-$lbl.log"; }
done
# sommerfeld also calls d_symmetry_bd_1b
for pair in "base:$BASE/src/sommerfeld_rout_gpu.cu:$INC_BASE" "cand:$CAND/src/sommerfeld_rout_gpu.cu:$INC_CAND"; do
  IFS=: read lbl src inc <<< "$pair"
  echo "=== compile $lbl (sommerfeld) ==="
  $NVCC -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++-13 $DEFS $inc $FLAGS -x cu -rdc=true -c "$src" -o "$EV/somm-$lbl.o" > "$EV/ptxas-somm-$lbl.log" 2>&1 || { echo "COMPILE_FAIL somm-$lbl"; tail -8 "$EV/ptxas-somm-$lbl.log"; }
done

echo "=== PTXAS prolong3_kernel ==="
for lbl in base cand; do
  echo "--- $lbl ---"
  grep -A6 "Compiling entry function.*prolong3_kernel" "$EV/ptxas-$lbl.log" | grep -E "stack|registers|Compiling" | head -4
done
echo "=== PTXAS sommerfeld_kernel (impact check) ==="
for lbl in base cand; do
  echo "--- $lbl ---"
  grep -A6 "Compiling entry function" "$EV/ptxas-somm-$lbl.log" | grep -E "stack|registers|Compiling" | head -8
done

echo "=== SASS comparison ==="
for lbl in base cand; do
  /usr/local/cuda-13.3/bin/cuobjdump -sass "$EV/$lbl.o" > "$EV/sass-$lbl.txt" 2>/dev/null || echo "SASS_FAIL $lbl"
done
python3 - "$EV" <<'PY'
import re, sys
from collections import Counter
ev = sys.argv[1]
def kern_sass(path):
    s = open(path).read()
    m = re.search(r'Function : _Z\d*prolong3_kernel.*?(?=\n\s*Function :|\Z)', s, re.S)
    return m.group(0) if m else ''
b = kern_sass(f'{ev}/sass-base.txt')
c = kern_sass(f'{ev}/sass-cand.txt')
def stats(sass, nm):
    ops = re.findall(r'/\*[0-9a-f]+\*/\s+([A-Z0-9.@!]+)', sass)
    cc = Counter(ops)
    print(f'{nm}: bytes={len(sass)} CALL={cc.get("CALL.ABS.NOINC",0)} BSSY={cc.get("BSSY",0)} LDL={cc.get("LDL.64",0)+cc.get("LDL.128",0)} STL={cc.get("STL.64",0)+cc.get("STL.128",0)} LDG={cc.get("LDG.E.64",0)+cc.get("LDG.E.128",0)} DFMA={cc.get("DFMA",0)} total_insns={len(ops)}')
stats(b, 'base prolong3')
stats(c, 'cand prolong3')
print('SASS_IDENTICAL' if b == c else 'SASS_DIFFERS')
PY
echo "=== DONE P6b L0 $(date -u) ==="
