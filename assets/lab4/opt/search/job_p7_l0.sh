#!/usr/bin/env bash
# iter16 P7 Level-0 — restrict3 column-fused unroll ptxas probe: base vs cand
# single-TU apples-to-apples (same CMake flags as deployed build incl.
# -maxrregcount=128), then SASS diff for restrict3_kernel.
set -uo pipefail
BASE=/home/h3240101033/lab4-gpu
CAND=/home/h3240101033/lab4-gpu-cand-p7-20260824-175747
EV="$CAND/evidence/p7-l0"
mkdir -p "$EV"
exec > "$EV/job-l0.log" 2>&1
echo "=== START P7 L0 $(date -u) ==="
nvidia-smi -L 2>/dev/null | head -2

export CUDACXX=/usr/local/cuda-13.3/bin/nvcc
NVCC=/usr/local/cuda-13.3/bin/nvcc
DEFS="-DMPI_CUDA_AWARE=0 -DUSE_GPU -Dfortran3 -Dnewc"
FLAGS="-O3 -Xptxas -v -std=c++14 --generate-code=arch=compute_80,code=[compute_80,sm_80] -rdc=true -lineinfo -maxrregcount=128"
INC_BASE="-I$BASE/src -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include"
INC_CAND="-I$CAND/src -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include"

for pair in "base:$BASE/src/prolongrestrict_cell_gpu.cu:$INC_BASE" "cand:$CAND/src/prolongrestrict_cell_gpu.cu:$INC_CAND"; do
  IFS=: read lbl src inc <<< "$pair"
  echo "=== compile $lbl ==="
  $NVCC -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++-13 $DEFS $inc $FLAGS -x cu -rdc=true -c "$src" -o "$EV/$lbl.o" > "$EV/ptxas-$lbl.log" 2>&1 || { echo "COMPILE_FAIL $lbl"; tail -8 "$EV/ptxas-$lbl.log"; }
done

echo "=== PTXAS restrict3_kernel ==="
for lbl in base cand; do
  echo "--- $lbl ---"
  grep -A8 "Compiling entry function.*restrict3_kernel" "$EV/ptxas-$lbl.log" | head -10
done
echo "=== PTXAS prolong3_kernel (control, must be unchanged) ==="
for lbl in base cand; do
  echo "--- $lbl ---"
  grep -A8 "Compiling entry function.*prolong3_kernel" "$EV/ptxas-$lbl.log" | head -8
done

echo "=== SASS dump (restrict3_kernel) ==="
for lbl in base cand; do
  /usr/local/cuda-13.3/bin/cuobjdump -sass "$EV/$lbl.o" > "$EV/sass-$lbl.txt" 2>/dev/null || echo "SASS_FAIL $lbl"
  echo "$lbl sass lines: $(wc -l < "$EV/sass-$lbl.txt")"
done
if [ -f "$EV/sass-base.txt" ] && [ -f "$EV/sass-cand.txt" ]; then
  echo "=== SASS restrict3_kernel diff ==="
  python3 - "$EV" <<'PY'
import re, sys
from collections import Counter
ev = sys.argv[1]
def kern_sass(path):
    s = open(path).read()
    m = re.search(r'Function : _Z\d*restrict3_kernel.*?(?=\n\s*Function :|\Z)', s, re.S)
    return m.group(0) if m else ''
b = kern_sass(f'{ev}/sass-base.txt')
c = kern_sass(f'{ev}/sass-cand.txt')
print(f'base restrict3 SASS bytes: {len(b)}, cand: {len(c)}')
if b == c:
    print('SASS_IDENTICAL')
else:
    def ops(sass):
        return re.findall(r'/\*[0-9a-f]+\*/\s+([A-Z0-9.@!]+)', sass)
    ob, oc = ops(b), ops(c)
    cb, cc = Counter(ob), Counter(oc)
    diff = {k: (cb[k], cc[k]) for k in set(cb) | set(cc) if cb[k] != cc[k]}
    print('op delta (base->cand):', diff)
    for nm, s in (('base', b), ('cand', c)):
        print(f'{nm}: LDL={len(re.findall(r"LDL", s))} STL={len(re.findall(r"STL", s))} LDG={len(re.findall(r"LDG", s))} FMA={len(re.findall(r"FMA", s))} DADD={len(re.findall(r"DADD", s))} DMUL={len(re.findall(r"DMUL", s))} CALL={len(re.findall(r"CALL", s))}')
PY
fi
echo "=== DONE P7 L0 $(date -u) ==="
