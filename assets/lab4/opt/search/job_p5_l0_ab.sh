#!/usr/bin/env bash
# iter14 P5 — fdderivs fh float probe (single variable: 61 h_* double->float)
# Level-0 ptxas (base/cand x lb2/nolb) + dead-end gate + Level-1 2-step A/B (OFF/ON x2)
set -uo pipefail
BASE=/home/h3240101033/lab4-gpu
CAND=/home/h3240101033/lab4-gpu-cand-p5-20260824-073425
EV="$CAND/evidence/p5"
mkdir -p "$EV"
exec > "$EV/job.log" 2>&1
echo "=== START P5 $(date -u) ==="
nvidia-smi -L 2>/dev/null | head -2

export CUDACXX=/usr/local/cuda-13.3/bin/nvcc
NVCC="$CUDACXX"
DEFS="-DMPI_CUDA_AWARE=0 -DUSE_GPU -Dfortran3 -Dnewc"
FLAGS="-O3 -Xptxas -v -std=c++14 --generate-code=arch=compute_80,code=[compute_80,sm_80] -rdc=true -lineinfo"
INC_BASE="-I$BASE/src -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include"
INC_CAND="-I$CAND/src -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include"
mkdir -p /tmp/p5-l0
cp "$BASE/src/bssn_rhs_gpu.cu" /tmp/p5-l0/base.cu
cp "$CAND/src/bssn_rhs_gpu.cu" /tmp/p5-l0/cand.cu
for pair in "base-lb2:/tmp/p5-l0/base.cu:$INC_BASE" "cand-lb2:/tmp/p5-l0/cand.cu:$INC_CAND" "base-nolb:/tmp/p5-l0/base.cu:$INC_BASE" "cand-nolb:/tmp/p5-l0/cand.cu:$INC_CAND"; do
  IFS=: read lbl src inc <<< "$pair"
  out=/tmp/p5-l0/$lbl.o
  if [ "$lbl" = "base-nolb" ] || [ "$lbl" = "cand-nolb" ]; then
    sed "s/__global__ __launch_bounds__(256, 2) void rhs_kernel/__global__ void rhs_kernel/" "$src" > /tmp/p5-l0/$lbl.cu
    src=/tmp/p5-l0/$lbl.cu
  fi
  echo "=== compile $lbl ==="
  $NVCC -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++-13 $DEFS $inc $FLAGS -x cu -rdc=true -c "$src" -o "$out" > "$EV/ptxas-$lbl.log" 2>&1 || { echo "COMPILE_FAIL $lbl"; tail -5 "$EV/ptxas-$lbl.log"; }
done

parse_regs()  { grep -A8 "Compiling entry function.*rhs_kernel" "$1" | grep -oE "Used [0-9]+ registers" | head -1 | grep -oE "[0-9]+"; }
parse_spill(){ grep -A8 "Compiling entry function.*rhs_kernel" "$1" | grep -oE "[0-9]+ bytes spill stores" | head -1 | grep -oE "^[0-9]+"; }
B_REGS=$(parse_regs "$EV/ptxas-base-lb2.log"); B_SPILL=$(parse_spill "$EV/ptxas-base-lb2.log")
N_REGS=$(parse_regs "$EV/ptxas-base-nolb.log"); N_SPILL=$(parse_spill "$EV/ptxas-base-nolb.log")
C_REGS=$(parse_regs "$EV/ptxas-cand-lb2.log"); C_SPILL=$(parse_spill "$EV/ptxas-cand-lb2.log")
CN_REGS=$(parse_regs "$EV/ptxas-cand-nolb.log"); CN_SPILL=$(parse_spill "$EV/ptxas-cand-nolb.log")
echo "=== P5 PTXAS SUMMARY ==="
echo "base-lb2  : regs=$B_REGS spill_stores=$B_SPILL"
echo "base-nolb : regs=$N_REGS spill_stores=$N_SPILL"
echo "cand-lb2  : regs=$C_REGS spill_stores=$C_SPILL"
echo "cand-nolb : regs=$CN_REGS spill_stores=$CN_SPILL"

GATE=DEAD_END
if [ -n "$CN_REGS" ] && [ "$CN_REGS" -lt 255 ]; then GATE=PASS; fi
if [ -n "$C_SPILL" ] && [ -n "$B_SPILL" ] && [ "$C_SPILL" -lt $((B_SPILL * 99 / 100)) ]; then GATE=PASS; fi
echo "GATE=$GATE"
if [ "$GATE" != PASS ]; then
  echo "P5_GATE_DEAD_END (natural regs still 255 AND lb2 spill stores unchanged)"
  exit 0
fi
echo "P5_GATE_PASS -> full build + Level-1 A/B"

export AMSS_ENABLE_GPU=ON AMSS_ENABLE_TWOP_GPU=OFF AMSS_ENABLE_TWOP_OMP_TUNE=ON
export AMSS_ENABLE_PACKED_RELAX=ON AMSS_ENABLE_TWOP_COS_TABLE=ON
export AMSS_CUDA_ARCHITECTURES=80 AMSS_OPT="-O3" AMSS_MPI_CUDA_AWARE=0

echo "=== BUILD base ==="
( cd "$BASE" && rm -rf build && ./compile.sh -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc > "$EV/build-base.log" 2>&1 || echo BUILD_FAIL_BASE )
ls -lh "$BASE/build/ABEGPU" 2>/dev/null
echo "=== BUILD cand ==="
( cd "$CAND" && rm -rf build && ./compile.sh -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc > "$EV/build-cand.log" 2>&1 || echo BUILD_FAIL_CAND )
ls -lh "$CAND/build/ABEGPU" 2>/dev/null

echo "=== PATCH INPUTS Final=2 ==="
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

export OMP_NUM_THREADS=8 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ulimit -s unlimited
run_ab() {
  local label=$1 tree=$2
  ( cd "$tree" && AMSS_OUTPUT_ROOT="$EV/$label" AMSS_CACHE_DIR="$BASE/twopuncture_cache" ./run.sh --twop-cache > "$EV/run-$label.log" 2>&1 || echo "RUN_FAIL $label" )
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

echo "=== RMS ON vs OFF (golden=OFF output) ==="
RES="$EV/on1/GW250118/AMSS_NCKU_output"; GOL="$EV/off1/GW250118/AMSS_NCKU_output"
( cd "$BASE" && RESULT_DIR="$RES" ./check.sh "$RES" "$GOL" > "$EV/check-on1.log" 2>&1 || true )
grep -E "Trajectory RMS|Trajectory:|Constraints:|FINAL" "$EV/check-on1.log" | tail -6
RES="$EV/on2/GW250118/AMSS_NCKU_output"; GOL="$EV/off2/GW250118/AMSS_NCKU_output"
( cd "$BASE" && RESULT_DIR="$RES" ./check.sh "$RES" "$GOL" > "$EV/check-on2.log" 2>&1 || true )
grep -E "Trajectory RMS|Trajectory:|Constraints:|FINAL" "$EV/check-on2.log" | tail -6

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
t_off = statistics.mean(off); t_on = statistics.mean(on)
print(f"OFF steps: {off}  mean={t_off:.4f} s/step")
print(f"ON  steps: {on}  mean={t_on:.4f} s/step")
print(f"F = {t_off/t_on:.4f}")
PY
echo "=== DONE P5 $(date -u) ==="
