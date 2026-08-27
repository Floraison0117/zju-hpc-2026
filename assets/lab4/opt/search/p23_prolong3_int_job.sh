#!/usr/bin/env bash
# Milestone C round 1: prolong3 interior/boundary split L0 probe + L1 A/B.
# Mirrors mb_intsplit_ab_job.sh (rhs split) for prolong3.
#   L0: single-TU ptxas (base/patched/int) + SASS static counts + opcode mix
#   Build: OFF = formal ~/lab4-gpu (deployed baseline), ON = candidate
#   GATE: int static SASS <= 0.75*base AND int regs <= base regs -> A/B else dead-end
#   L1: 2-step OFF/ON x2 interleaved, .dat IDENTICAL + check.sh + F
#   Bonus: restrict3_kernel L0 static probe (same d_symmetry_bd_1b mechanism)
set -uo pipefail
BASE=/home/h3240101033/lab4-gpu
CAND=$(cat ~/.p23_prolong3_path)
TS=$(date -u +%Y%m%d-%H%M%S)
EV="$CAND/evidence/p23-prolong3-int-$TS"
mkdir -p "$EV"
exec > "$EV/job.log" 2>&1
echo "=== START $(date -u) ==="; hostname
nvidia-smi -L 2>/dev/null | head -2
echo "BASE=$BASE CAND=$CAND EV=$EV"

export AMSS_ENABLE_GPU=ON AMSS_ENABLE_TWOP_GPU=OFF AMSS_ENABLE_TWOP_OMP_TUNE=ON
export AMSS_ENABLE_PACKED_RELAX=ON AMSS_ENABLE_TWOP_COS_TABLE=ON
export AMSS_CUDA_ARCHITECTURES=80 AMSS_OPT="-O3" AMSS_MPI_CUDA_AWARE=0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ulimit -s unlimited

NVCC=/usr/local/cuda-13.3/bin/nvcc
CU=$(ls /usr/local/cuda-13.3/bin/cuobjdump 2>/dev/null || echo /usr/local/cuda/bin/cuobjdump)
FLAGS="-arch=sm_80 -O3 -rdc=true -lineinfo -DUSE_GPU -Dfortran3 -Dnewc -DMPI_CUDA_AWARE=0"

echo "=== L0: single-TU ptxas (apples-to-apples) ==="
( cd "$BASE/src"   && $NVCC $FLAGS -Xptxas -v -c prolongrestrict_cell_gpu.cu     -o "$EV/base.o"     > "$EV/base.nvcc.log"     2>&1 ); echo "base rc=$?"
( cd "$CAND/src"   && $NVCC $FLAGS -Xptxas -v -c prolongrestrict_cell_gpu.cu     -o "$EV/patched.o"  > "$EV/patched.nvcc.log"  2>&1 ); echo "patched rc=$?"
( cd "$CAND/src"   && $NVCC $FLAGS -Xptxas -v -c prolongrestrict_cell_gpu_int.cu -o "$EV/int.o"     > "$EV/int.nvcc.log"      2>&1 ); echo "int rc=$?"
for tag in base patched int; do
  echo "--- $tag ptxas (prolong3_kernel) ---"
  if [ "$tag" = int ]; then FN='prolong3_kernel_int'; else FN='prolong3_kernel'; fi
  grep -A1 "Function properties for .*$FN" "$EV/$tag.nvcc.log" | grep -E "bytes stack frame" | head -2
  grep -oE "Used [0-9]+ registers" "$EV/$tag.nvcc.log" | head -2
  grep -oE "[0-9]+ bytes spill stores, [0-9]+ bytes spill loads" "$EV/$tag.nvcc.log" | head -2
  grep -oE "[0-9]+ bytes stack frame" "$EV/$tag.nvcc.log" | head -2
done

echo "=== L0: SASS static instruction counts (single-TU) ==="
for tag in base patched int; do
  $CU -sass "$EV/$tag.o" > "$EV/$tag.sass" 2>/dev/null
  if [ "$tag" = int ]; then FN='prolong3_kernel_int'; else FN='prolong3_kernel[^_]'; fi
  n=$($CU -sass "$EV/$tag.o" 2>/dev/null | sed -n "/Function : .*$FN/,/Function :/p" | grep -cE "^\s+/\*[0-9a-f]+\*/")
  echo "$tag: $([ "$tag" = int ] && echo prolong3_kernel_int || echo prolong3_kernel) static SASS = $n"
done
echo "--- base prolong3_kernel opcode mix (top 16) ---"
$CU -sass "$EV/base.o" 2>/dev/null | sed -n "/Function : .*prolong3_kernel[^_]/,/Function :/p" | grep -oE "^\s+/\*[0-9a-f]+\*/\s+[A-Z0-9_.@]+" | awk '{print $2}' | sed 's/@.*//' | sort | uniq -c | sort -rn | head -16

echo "=== L0: restrict3 probe (same d_symmetry_bd_1b mechanism) ==="
for tag in base int; do
  if [ "$tag" = int ]; then FN='restrict3_kernel_int'; else FN='restrict3_kernel[^_]'; fi
  n=$($CU -sass "$EV/$tag.o" 2>/dev/null | sed -n "/Function : .*$FN/,/Function :/p" | grep -cE "^\s+/\*[0-9a-f]+\*/")
  echo "$tag: restrict3 static SASS = $n"
  grep -A1 "Function properties for .*$FN" "$EV/$tag.nvcc.log" | grep -E "bytes stack frame" | head -2
  grep -oE "Used [0-9]+ registers" "$EV/$tag.nvcc.log" | head -2
done

echo "=== BUILD base (formal, in-place build dir) and candidate ==="
( cd "$BASE" && AMSS_BUILD_DIR="$BASE/build" JOBS=8 ./compile.sh -DCMAKE_CUDA_COMPILER=$NVCC > "$EV/build-base.log" 2>&1 ) || echo BUILD_FAIL_BASE
ls -lh "$BASE/build/ABEGPU" 2>/dev/null || echo NO_BASE_BINARY
( cd "$CAND" && rm -rf build && AMSS_BUILD_DIR="$CAND/build" JOBS=8 ./compile.sh -DCMAKE_CUDA_COMPILER=$NVCC > "$EV/build-cand.log" 2>&1 ) || echo BUILD_FAIL_CAND
ls -lh "$CAND/build/ABEGPU" 2>/dev/null || echo NO_CAND_BINARY

echo "=== L0: built-binary SASS static counts (cross-check) ==="
for pair in "base:$BASE/build/ABEGPU:prolong3_kernel[^_]" "cand-int:$CAND/build/ABEGPU:prolong3_kernel_int" "cand-bnd:$CAND/build/ABEGPU:prolong3_kernel[^_]"; do
  lbl=${pair%%:*}; rest=${pair#*:}; bin=${rest%%:*}; fn=${rest#*:}
  n=$($CU -sass "$bin" 2>/dev/null | sed -n "/Function : .*$fn/,/Function :/p" | grep -cE "^\s+/\*[0-9a-f]+\*/")
  echo "built:$lbl $([ "$lbl" = cand-int ] && echo prolong3_kernel_int || echo prolong3_kernel) static SASS = $n"
done

echo "=== GATE (int static <= 0.75*base AND int regs <= base regs) ==="
base_count=$($CU -sass "$EV/base.o" 2>/dev/null | sed -n "/Function : .*prolong3_kernel[^_]/,/Function :/p" | grep -cE "^\s+/\*[0-9a-f]+\*/")
int_count=$($CU -sass "$EV/int.o" 2>/dev/null | sed -n "/Function : .*prolong3_kernel_int/,/Function :/p" | grep -cE "^\s+/\*[0-9a-f]+\*/")
base_regs=$(grep -A2 "Function properties for .*prolong3_kernel[^_]" "$EV/base.nvcc.log" | grep -oE "Used [0-9]+ registers" | head -1 | grep -oE "[0-9]+")
int_regs=$(grep -A2 "Function properties for .*prolong3_kernel_int" "$EV/int.nvcc.log" | grep -oE "Used [0-9]+ registers" | head -1 | grep -oE "[0-9]+")
echo "base_count=$base_count int_count=$int_count base_regs=$base_regs int_regs=$int_regs"
thresh=$((base_count * 3 / 4))
GATE=FAIL
if [ -n "$base_count" ] && [ -n "$int_count" ] && [ "$int_count" -le "$thresh" ] && [ -n "$base_regs" ] && [ -n "$int_regs" ] && [ "$int_regs" -le "$base_regs" ]; then
  GATE=PASS; echo "GATE: PASS (int <= $thresh static and regs $int_regs <= $base_regs)"
else
  echo "GATE: FAIL -> dead-end report, skip A/B"
fi

if [ "$GATE" = PASS ]; then
echo "=== PATCH INPUTS Final=2 (backup formal) ==="
cp "$BASE/AMSS_NCKU_Input.py" "$EV/input.base.backup.py"
for t in "$BASE" "$CAND"; do
  python3 - "$t/AMSS_NCKU_Input.py" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
s = re.sub(r'Final_Evolution_Time\s*=.*', 'Final_Evolution_Time = 2.0', s)
s = re.sub(r'Analysis_Time\s*=.*', 'Analysis_Time = 0.1', s)
open(p, 'w').write(s)
print("patched", p)
PY
done
for lbl in off1 on1 off2 on2; do mkdir -p "$EV/$lbl/GW250118"; done

echo "=== A/B runs (OFF/ON x2 interleaved, same node, shared twop cache) ==="
run_ab() {
  local label=$1 tree=$2
  ( cd "$tree" && AMSS_BUILD_DIR="$tree/build" AMSS_OUTPUT_ROOT="$EV/$label" AMSS_CACHE_DIR="$BASE/twopuncture_cache" ./run.sh --twop-cache > "$EV/run-$label.log" 2>&1 || echo "RUN_FAIL $label" )
  echo "--- $label steps ---"
  grep "After Step:" "$EV/run-$label.log" | tail -2
}
echo "=== A/B OFF1 ==="; run_ab off1 "$BASE"
echo "=== A/B ON1  ==="; run_ab on1  "$CAND"
echo "=== A/B OFF2 ==="; run_ab off2 "$BASE"
echo "=== A/B ON2  ==="; run_ab on2  "$CAND"

echo "=== RESTORE FORMAL INPUT ==="
cp "$EV/input.base.backup.py" "$BASE/AMSS_NCKU_Input.py"
sha256sum "$BASE/AMSS_NCKU_Input.py"

echo "=== .dat IDENTICAL (OFF vs ON, comments stripped) ==="
for pair in "off1 on1" "off2 on2"; do
  set -- $pair; OFF=$1; ON=$2
  echo "--- $OFF vs $ON ---"
  for f in bssn_BH.dat bssn_constraint.dat bssn_psi4.dat bssn_ADMQs.dat; do
    a="$EV/$OFF/GW250118/AMSS_NCKU_output/binary_output/$f"
    b="$EV/$ON/GW250118/AMSS_NCKU_output/binary_output/$f"
    if [ -f "$a" ] && [ -f "$b" ]; then
      ha=$(grep -v "^#" "$a" | sha256sum | cut -d' ' -f1)
      hb=$(grep -v "^#" "$b" | sha256sum | cut -d' ' -f1)
      [ "$ha" = "$hb" ] && echo "$f: IDENTICAL $ha" || echo "$f: DIFFER ($ha vs $hb)"
    else
      echo "$f: MISSING ($a / $b)"
    fi
  done
done

echo "=== check.sh ON vs OFF (golden = OFF output) ==="
for n in 1 2; do
  RES="$EV/on$n/GW250118/AMSS_NCKU_output"; GOL="$EV/off$n/GW250118/AMSS_NCKU_output"
  ( cd "$BASE" && RESULT_DIR="$RES" ./check.sh "$RES" "$GOL" > "$EV/check-on$n.log" 2>&1 || true )
  echo "--- on$n vs off$n ---"
  grep -E "Trajectory RMS|Trajectory:|Constraints|FINAL" "$EV/check-on$n.log" | tail -6
done

echo "=== F (per-step) ==="
python3 - "$EV" <<'PY'
import re, statistics, sys
ev = sys.argv[1]
def steps(label):
    v = []
    for ln in open(f"{ev}/run-{label}.log"):
        m = re.search(r'After Step: (\d+) My Rank: 0 takes ([\d.]+) seconds', ln)
        if m: v.append(float(m.group(2)))
    return v
off = steps("off1") + steps("off2")
on  = steps("on1") + steps("on2")
if not off or not on:
    print(f"OFF steps={off} ON steps={on} (insufficient data)"); sys.exit(0)
print(f"OFF steps: {off}")
print(f"ON  steps: {on}")
print(f"OFF mean={statistics.mean(off):.4f} median={statistics.median(off):.4f} s/step")
print(f"ON  mean={statistics.mean(on):.4f} median={statistics.median(on):.4f} s/step")
print(f"F(mean)={statistics.mean(off)/statistics.mean(on):.4f}  F(median)={statistics.median(off)/statistics.median(on):.4f}")
PY
else
echo "=== GATE FAIL: no A/B (dead-end path) ==="
fi
echo "=== DONE $(date -u) ==="
