#!/usr/bin/env bash
# iter17 P8 Level-0 — fmisc helper forceinline ptxas probe: base vs cand
# single-TU apples-to-apples (same CMake flags as deployed build incl.
# -maxrregcount=128 -rdc=true), then SASS CALL-count diff for
# sommerfeld_rout_kernel / sommerfeld_routbam_kernel / global_interp_kernel.
set -uo pipefail
BASE=/home/h3240101033/lab4-gpu
CAND="$1"
EV="$CAND/evidence/p8-l0"
mkdir -p "$EV"
exec > "$EV/job-l0.log" 2>&1
echo "=== START P8 L0 $(date -u) ==="
nvidia-smi -L 2>/dev/null | head -2

NVCC=/usr/local/cuda-13.3/bin/nvcc
DEFS="-DMPI_CUDA_AWARE=0 -DUSE_GPU -Dfortran3 -Dnewc"
FLAGS="-O3 -Xptxas -v -std=c++14 --generate-code=arch=compute_80,code=[compute_80,sm_80] -rdc=true -lineinfo -maxrregcount=128"
INC_BASE="-I$BASE/src -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include"
INC_CAND="-I$CAND/src -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include"

for pair in "base:$BASE/src/fmisc_gpu.cu:$INC_BASE" "cand:$CAND/src/fmisc_gpu.cu:$INC_CAND"; do
  IFS=: read lbl src inc <<< "$pair"
  echo "=== compile $lbl (fmisc_gpu) ==="
  $NVCC -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++-13 $DEFS $inc $FLAGS -x cu -rdc=true -c "$src" -o "$EV/fmisc-$lbl.o" > "$EV/ptxas-fmisc-$lbl.log" 2>&1 || { echo "COMPILE_FAIL fmisc-$lbl"; tail -8 "$EV/ptxas-fmisc-$lbl.log"; }
done
for pair in "base:$BASE/src/sommerfeld_rout_gpu.cu:$INC_BASE" "cand:$CAND/src/sommerfeld_rout_gpu.cu:$INC_CAND"; do
  IFS=: read lbl src inc <<< "$pair"
  echo "=== compile $lbl (sommerfeld) ==="
  $NVCC -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++-13 $DEFS $inc $FLAGS -x cu -rdc=true -c "$src" -o "$EV/somm-$lbl.o" > "$EV/ptxas-somm-$lbl.log" 2>&1 || { echo "COMPILE_FAIL somm-$lbl"; tail -8 "$EV/ptxas-somm-$lbl.log"; }
done
# control: prolongrestrict must be byte-identical (no change) -> same ptxas
for pair in "base:$BASE/src/prolongrestrict_cell_gpu.cu:$INC_BASE" "cand:$CAND/src/prolongrestrict_cell_gpu.cu:$INC_CAND"; do
  IFS=: read lbl src inc <<< "$pair"
  echo "=== compile $lbl (prolongrestrict control) ==="
  $NVCC -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++-13 $DEFS $inc $FLAGS -x cu -rdc=true -c "$src" -o "$EV/prol-$lbl.o" > "$EV/ptxas-prol-$lbl.log" 2>&1 || { echo "COMPILE_FAIL prol-$lbl"; tail -8 "$EV/ptxas-prol-$lbl.log"; }
done

echo "=== PTXAS sommerfeld kernels ==="
for lbl in base cand; do
  echo "--- $lbl ---"
  grep -B1 -A4 "registers\|stack frame\|spill" "$EV/ptxas-somm-$lbl.log" | grep -E "Compiling entry|registers|stack frame|spill" | head -12
done
echo "=== PTXAS fmisc kernels (global_interp) ==="
for lbl in base cand; do
  echo "--- $lbl ---"
  grep -B1 -A4 "registers\|stack frame\|spill" "$EV/ptxas-fmisc-$lbl.log" | grep -E "Compiling entry|registers|stack frame|spill" | head -20
done
echo "=== PTXAS prolongrestrict (control, must match) ==="
for lbl in base cand; do
  echo "--- $lbl ---"
  grep -E "Compiling entry function|registers|stack frame|spill" "$EV/ptxas-prol-$lbl.log" | head -10
done

echo "=== SASS CALL comparison ==="
for lbl in base cand; do
  /usr/local/cuda-13.3/bin/cuobjdump -sass "$EV/somm-$lbl.o" > "$EV/sass-somm-$lbl.txt" 2>/dev/null || echo "SASS_FAIL somm-$lbl"
  /usr/local/cuda-13.3/bin/cuobjdump -sass "$EV/fmisc-$lbl.o" > "$EV/sass-fmisc-$lbl.txt" 2>/dev/null || echo "SASS_FAIL fmisc-$lbl"
done
python3 - "$EV" <<'PY'
import re, sys
from collections import Counter
ev = sys.argv[1]
def kern_sass(path, name):
    s = open(path).read()
    pat = re.compile(r'Function : \S*' + re.escape(name) + r'.*?(?=\n\s*Function :|\Z)', re.S)
    m = pat.search(s)
    return m.group(0) if m else ''
def stats(sass, nm):
    ops = re.findall(r'/\*[0-9a-f]+\*/\s+([A-Z0-9.@!]+)', sass)
    cc = Counter(ops)
    print(f'{nm}: bytes={len(sass)} CALL={cc.get("CALL.ABS.NOINC",0)} BSSY={cc.get("BSSY",0)} LDL={cc.get("LDL.64",0)+cc.get("LDL.128",0)} STL={cc.get("STL.64",0)+cc.get("STL.128",0)} total_insns={len(ops)}')
for kern in ["sommerfeld_rout_kernel", "sommerfeld_routbam_kernel"]:
    b = kern_sass(f'{ev}/sass-somm-base.txt', kern)
    c = kern_sass(f'{ev}/sass-somm-cand.txt', kern)
    print(f'--- {kern} ---')
    stats(b, 'base')
    stats(c, 'cand')
    print('SASS_IDENTICAL' if b == c else 'SASS_DIFFERS')
for kern in ["global_interp_kernel", "global_interp_amr_kernel"]:
    b = kern_sass(f'{ev}/sass-fmisc-base.txt', kern)
    c = kern_sass(f'{ev}/sass-fmisc-cand.txt', kern)
    print(f'--- {kern} ---')
    stats(b, 'base')
    stats(c, 'cand')
    print('SASS_IDENTICAL' if b == c else 'SASS_DIFFERS')
# device-function frame listing (do the ABI bodies disappear?)
for lbl in ["base", "cand"]:
    s = open(f'{ev}/sass-fmisc-{lbl}.txt').read()
    fns = re.findall(r'Function : (\S+)', s)
    print(f'fmisc {lbl} functions: {[f for f in fns if "global_interp" in f or "polin" in f or "decide" in f or "polint" in f]}')
PY
echo "=== DONE P8 L0 $(date -u) ==="
