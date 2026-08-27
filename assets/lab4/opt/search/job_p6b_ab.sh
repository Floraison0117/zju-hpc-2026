#!/usr/bin/env bash
# iter15 P6 Level-1 — prolong3 6x6 unroll A/B (OFF/ON x2, 2-step) + bit-exact RMS
# base = formal tree (build/ABEGPU is fresh deployed binary, verified in job-l0)
set -uo pipefail
BASE=/home/h3240101033/lab4-gpu
CAND=/home/h3240101033/lab4-gpu-cand-p6b-20260824-082335
EV="$CAND/evidence/p6b"
mkdir -p "$EV"
exec > "$EV/job-ab.log" 2>&1
echo "=== START P6 A/B $(date -u) ==="
nvidia-smi -L 2>/dev/null | head -2

export AMSS_ENABLE_GPU=ON AMSS_ENABLE_TWOP_GPU=OFF AMSS_ENABLE_TWOP_OMP_TUNE=ON
export AMSS_ENABLE_PACKED_RELAX=ON AMSS_ENABLE_TWOP_COS_TABLE=ON
export AMSS_CUDA_ARCHITECTURES=80 AMSS_OPT="-O3" AMSS_MPI_CUDA_AWARE=0

echo "=== verify formal binary == base (ptxas probe cross-check in job-l0) ==="
ls -lh "$BASE/build/ABEGPU" || echo NO_BASE_BINARY

echo "=== BUILD cand ==="
( cd "$CAND" && rm -rf build && AMSS_BUILD_DIR="$CAND/build" JOBS=8 ./compile.sh -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc > "$EV/build-cand.log" 2>&1 ) || echo BUILD_FAIL_CAND
ls -lh "$CAND/build/ABEGPU" 2>/dev/null || echo NO_CAND_BINARY

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

export OMP_NUM_THREADS=8 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ulimit -s unlimited
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
grep -E "Final_Evolution_Time|Analysis_Time" "$BASE/AMSS_NCKU_Input.py"

echo "=== RMS ON vs OFF (golden=OFF output) ==="
for n in 1 2; do
  RES="$EV/on$n/GW250118/AMSS_NCKU_output"; GOL="$EV/off$n/GW250118/AMSS_NCKU_output"
  ( cd "$BASE" && RESULT_DIR="$RES" ./check.sh "$RES" "$GOL" > "$EV/check-on$n.log" 2>&1 || true )
  echo "--- on$n vs off$n ---"
  grep -E "Trajectory RMS|Trajectory:|Constraints:|FINAL" "$EV/check-on$n.log" | tail -6
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
t_off = statistics.mean(off); t_on = statistics.mean(on)
print(f"OFF steps: {off}  mean={t_off:.4f} s/step")
print(f"ON  steps: {on}  mean={t_on:.4f} s/step")
print(f"F = {t_off/t_on:.4f}")
PY
echo "=== DONE P6 A/B $(date -u) ==="
