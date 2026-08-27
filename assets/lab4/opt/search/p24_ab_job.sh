#!/usr/bin/env bash
# Milestone C round 2 A/B: one kernel per job (30-min wall-clock discipline:
# 1 build + 1 A/B only). OFF = formal binary, ON = candidate binary,
# 2-step OFF/ON x2 interleaved, shared twop cache, .dat IDENTICAL + check.sh.
# Usage: p24_ab_job.sh <r3|gi|som>
set -uo pipefail
K="${1:-}"
case "$K" in
  r3)  CAND=$(cat ~/.p24_r3_path);  KERN='restrict3_kernel';     INT='restrict3_kernel_int';     TU='prolongrestrict_cell_gpu';     REGEX='restrict3_kernel[^_]' ;;
  gi)  CAND=$(cat ~/.p24_gi_path);  KERN='global_interp_kernel'; INT='global_interp_kernel_int'; TU='fmisc_gpu';                   REGEX='global_interp_kernel[^_]' ;;
  som) CAND=$(cat ~/.p24_som_path); KERN='sommerfeld_rout_kernel'; INT='sommerfeld_rout_kernel_int'; TU='sommerfeld_rout_gpu';   REGEX='sommerfeld_rout_kernel[^_]' ;;
  *) echo "usage: $0 <r3|gi|som>"; exit 2 ;;
esac
BASE=/home/h3240101033/lab4-gpu
TS=$(date -u +%Y%m%d-%H%M%S)
EV="$CAND/evidence/p24-${K}-ab-$TS"
mkdir -p "$EV"
exec > "$EV/job.log" 2>&1
echo "=== START $(date -u) ==="; hostname
nvidia-smi -L 2>/dev/null | head -2
echo "BASE=$BASE CAND=$CAND K=$K EV=$EV"
HAD_CACHE=0; [ -e "$BASE/twopuncture_cache" ] && HAD_CACHE=1

export AMSS_ENABLE_GPU=ON AMSS_ENABLE_TWOP_GPU=OFF AMSS_ENABLE_TWOP_OMP_TUNE=ON
export AMSS_ENABLE_PACKED_RELAX=ON AMSS_ENABLE_TWOP_COS_TABLE=ON
export AMSS_CUDA_ARCHITECTURES=80 AMSS_OPT="-O3" AMSS_MPI_CUDA_AWARE=0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ulimit -s unlimited
NVCC=/usr/local/cuda-13.3/bin/nvcc
CU=$(ls /usr/local/cuda-13.3/bin/cuobjdump 2>/dev/null || echo /usr/local/cuda/bin/cuobjdump)

echo "=== candidate hashes ==="
sha256sum "$CAND"/src/${TU}.cu "$CAND"/src/${TU}_int.cu 2>/dev/null | cut -c1-16 | xargs -n2 echo "  "

echo "=== BUILD candidate ==="
( cd "$CAND" && rm -rf build && AMSS_BUILD_DIR="$CAND/build" JOBS=8 ./compile.sh -DCMAKE_CUDA_COMPILER=$NVCC > "$EV/build-cand.log" 2>&1 ) || echo BUILD_FAIL_CAND
ls -lh "$CAND/build/ABEGPU" 2>/dev/null || { echo NO_CAND_BINARY; exit 1; }

echo "=== L0 cross-check on built binaries ==="
for pair in "base:$BASE/build/ABEGPU:$REGEX" "cand-bnd:$CAND/build/ABEGPU:$REGEX" "cand-int:$CAND/build/ABEGPU:$INT"; do
  lbl=${pair%%:*}; rest=${pair#*:}; bin=${rest%%:*}; fn=${rest#*:}
  n=$($CU -sass "$bin" 2>/dev/null | sed -n "/Function : .*$fn/,/Function :/p" | grep -cE "^\s+/\*[0-9a-f]+\*/")
  echo "built:$lbl $fn static SASS = $n"
done

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
if [ "$HAD_CACHE" = 0 ]; then rm -rf "$BASE/twopuncture_cache"; fi

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
