#!/usr/bin/env bash
# Milestone B L1 decisive test: RHS interior/boundary split A/B (OFF/ON x2, 2-step)
# plus L0 ptxas/SASS for rhs_kernel_int. Runs entirely in one lab4g10 job.
set -uo pipefail
BASE=/home/h3240101033/lab4-gpu
CAND=$(cat ~/.mb_intsplit_path)
TS=$(date -u +%Y%m%d-%H%M%S)
EV="$CAND/evidence/mb-intsplit-ab-$TS"
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
( cd "$BASE/src"   && $NVCC $FLAGS -maxrregcount=128 -Xptxas -v -c bssn_rhs_gpu.cu     -o "$EV/base_lb2.o"     > "$EV/base_lb2.nvcc.log"     2>&1 ); echo "base_lb2 rc=$?"
( cd "$CAND/src"   && $NVCC $FLAGS -maxrregcount=128 -Xptxas -v -c bssn_rhs_gpu.cu     -o "$EV/patched_lb2.o" > "$EV/patched_lb2.nvcc.log" 2>&1 ); echo "patched_lb2 rc=$?"
( cd "$CAND/src"   && $NVCC $FLAGS -maxrregcount=128 -Xptxas -v -c bssn_rhs_gpu_int.cu -o "$EV/int_lb2.o"    > "$EV/int_lb2.nvcc.log"    2>&1 ); echo "int_lb2 rc=$?"
for tag in base_lb2 patched_lb2 int_lb2; do
  echo "--- $tag ptxas (rhs_kernel) ---"
  grep -A1 "Function properties for .*rhs_kernel" "$EV/$tag.nvcc.log" | grep -E "bytes stack frame" | head -2
  grep -oE "Used [0-9]+ registers" "$EV/$tag.nvcc.log" | head -2
  grep -oE "[0-9]+ bytes spill stores, [0-9]+ bytes spill loads" "$EV/$tag.nvcc.log" | head -2
  grep -oE "[0-9]+ bytes stack frame" "$EV/$tag.nvcc.log" | head -2
done

echo "=== L0: SASS static instruction counts ==="
for tag in base_lb2 patched_lb2 int_lb2; do
  $CU -sass "$EV/$tag.o" > "$EV/$tag.sass" 2>/dev/null
  if [ "$tag" = int_lb2 ]; then FN='rhs_kernel_int'; else FN='rhs_kernel[^_]'; fi
  n=$($CU -sass "$EV/$tag.o" 2>/dev/null | sed -n "/Function : .*$FN/,/Function :/p" | grep -cE "^\s+/\*[0-9a-f]+\*/")
  echo "$tag: $([ "$tag" = int_lb2 ] && echo rhs_kernel_int || echo rhs_kernel) static SASS = $n"
done
echo "--- int_lb2 opcode mix (top 14) ---"
$CU -sass "$EV/int_lb2.o" 2>/dev/null | sed -n "/Function : .*rhs_kernel_int/,/Function :/p" | grep -oE "^\s+/\*[0-9a-f]+\*/\s+[A-Z0-9_.@]+" | awk '{print $2}' | sed 's/@.*//' | sort | uniq -c | sort -rn | head -14

echo "=== BUILD candidate ==="
( cd "$CAND" && rm -rf build && AMSS_BUILD_DIR="$CAND/build" JOBS=8 ./compile.sh -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc > "$EV/build-cand.log" 2>&1 ) || echo BUILD_FAIL_CAND
ls -lh "$CAND/build/ABEGPU" 2>/dev/null || echo NO_CAND_BINARY
grep -E "ptxas info|Used [0-9]+ registers" "$EV/build-cand.log" | grep -iE "rhs_kernel|register" | head -6

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
echo "=== DONE $(date -u) ==="
