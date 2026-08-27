#!/bin/bash
# P-RK4: RK4 cross-variable batching (launch-structure lever).
#   Mechanism: 912,576 per-variable RK4 launches (24 vars x 38,024 sites,
#   19.76s / 3.2% module) -> one launch per (block, lev, substep) covering
#   all 24 variables (grid.y = var_idx), bit-exact by construction
#   (identical elementwise arithmetic on disjoint per-var arrays).
#   Level-0: build -Xptxas -v, grep rungekutta4_batch_kernel.
#   Level-1: 2-step interleaved A/B, analysis OFF (RK4 runs every substep).
#   base = deployed 26bcd binary.
set -uo pipefail
ROOT=$(ls -dt ~/lab4-gpu-cand-rk4batch-* 2>/dev/null | head -1)
BASEBUILD=/home/h3240101033/lab4-gpu-cand-p26bcd-combined-20260826-103614/build-p26bcd
CACHE=/home/h3240101033/lab4-gpu-cand-p26bcd-combined-20260826-103614/twopuncture_cache
cd "$ROOT" || { echo "NO CANDIDATE"; exit 1; }
export OMP_NUM_THREADS=8 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ulimit -s unlimited
EV="$ROOT/evidence/rk4batch"
mkdir -p "$EV"; exec > "$EV/job.log" 2>&1
echo "=== START RK4BATCH $(date -u) ROOT=$ROOT ==="
nvidia-smi -L 2>/dev/null | head -2
ls -lh "$BASEBUILD/ABEGPU" || { echo NO_BASE_BUILD; exit 2; }
export AMSS_ENABLE_GPU=ON AMSS_ENABLE_TWOP_GPU=OFF AMSS_ENABLE_TWOP_OMP_TUNE=ON
export AMSS_ENABLE_PACKED_RELAX=ON AMSS_ENABLE_TWOP_COS_TABLE=ON
export AMSS_CUDA_ARCHITECTURES=80 AMSS_OPT="-O3"
export AMSS_CACHE_DIR="$CACHE"
if [ -x /usr/local/cuda-13.3/bin/nvcc ]; then export CUDACXX=/usr/local/cuda-13.3/bin/nvcc
elif [ -x /usr/local/cuda/bin/nvcc ]; then export CUDACXX=/usr/local/cuda/bin/nvcc; fi

# --- Level-0: candidate build with ptxas -v ---
export AMSS_BUILD_DIR="$ROOT/build-rk4batch"
rm -rf "$AMSS_BUILD_DIR"
./compile.sh -DCMAKE_CUDA_FLAGS="-Xptxas -v" > "$EV/build-rk4batch.log" 2>&1 \
  && echo "BUILD_RK4BATCH_OK" || { echo "BUILD_RK4BATCH_FAIL"; grep -iE "error" "$EV/build-rk4batch.log" | head -20; exit 3; }
echo "=== batch kernel ptxas (candidate) ==="
grep -A4 "Function.*rungekutta4_batch_kernel" "$EV/build-rk4batch.log" | grep -iE "registers|spill|stack" | head -4

# --- 2-step config, analysis OFF ---
python3 - <<'PY'
import re
s=open('AMSS_NCKU_Input.py').read()
s=re.sub(r'Final_Evolution_Time\s*=.*','Final_Evolution_Time = 2.0',s)
s=re.sub(r'Analysis_Time\s*=.*','Analysis_Time = 1000.0',s)
open('AMSS_NCKU_Input.py','w').write(s)
PY

# --- Level-1: 4-round interleaved A/B ---
run_v() {
  local v="$1" b="$2"
  export AMSS_BUILD_DIR="$b" AMSS_OUTPUT_ROOT="$ROOT" AMSS_CACHE_DIR="$CACHE"
  rm -rf "$ROOT/GW250118_$v" "$ROOT/GW250118"
  ./run.sh --twop-cache > "$EV/run-$v.log" 2>&1 || echo "run $v exited"
  cp -r "$ROOT/GW250118/AMSS_NCKU_output" "$ROOT/GW250118_$v" 2>/dev/null
  echo "--- $v ---"; grep -E "After Step: [12] My Rank" "$EV/run-$v.log" | tail -2
}
run_v base1 "$BASEBUILD"
run_v cand1 "$ROOT/build-rk4batch"
run_v cand2 "$ROOT/build-rk4batch"
run_v base2 "$BASEBUILD"

# --- Bit-exact check ---
echo "=== BIT-EXACT ==="
BITEXACT=1
for pair in "base1 cand1" "base2 cand2"; do
  set -- $pair; a="$ROOT/GW250118_$1"; b="$ROOT/GW250118_$2"
  for fn in bssn_BH.dat bssn_psi4.dat bssn_ADMQs.dat bssn_constraint.dat; do
    if [ -f "$a/$fn" ] && [ -f "$b/$fn" ]; then
      if diff <(tail -n+2 "$a/$fn") <(tail -n+2 "$b/$fn") >/dev/null; then echo "$1-$2 $fn IDENTICAL"; else echo "$1-$2 $fn DIFFER"; BITEXACT=0; fi
    else echo "$1-$2 $fn missing"; BITEXACT=0; fi
  done
done

# --- F from Step2 medians ---
S2_BASE=$(grep -h "After Step: 2 My Rank" "$EV/run-base1.log" "$EV/run-base2.log" | awk '{print $(NF-1)}' | sort -n | awk '{a[NR]=$1} END{print (NR%2? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2)}')
S2_CAND=$(grep -h "After Step: 2 My Rank" "$EV/run-cand1.log" "$EV/run-cand2.log" | awk '{print $(NF-1)}' | sort -n | awk '{a[NR]=$1} END{print (NR%2? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2)}')
echo "=== F = base_median/cand_median ==="
echo "Step2 base median: $S2_BASE  Step2 cand median: $S2_CAND"
F=$(python3 -c "print('%.4f' % ($S2_BASE/$S2_CAND))" 2>/dev/null || echo "NA")
echo "F(speedup) = $F"
GATE=$(python3 -c "import sys; sys.exit(0 if (float('$F')>1.0) else 1)" 2>/dev/null; echo $?)
if [ "$BITEXACT" -ne 1 ] || [ "$GATE" != "0" ]; then
  echo "=== GATE FAILED (bit-exact=$BITEXACT F=$F): no Level-2 ==="
else
  echo "=== GATE PASSED (bit-exact=1 F=$F): Level-2 candidate ==="
fi

# --- restore OJ config ---
python3 - <<'PY'
import re
s=open('AMSS_NCKU_Input.py').read()
s=re.sub(r'Final_Evolution_Time\s*=.*','Final_Evolution_Time = 100.0',s)
s=re.sub(r'Analysis_Time\s*=.*','Analysis_Time = 0.1',s)
open('AMSS_NCKU_Input.py','w').write(s)
PY
echo "=== DONE RK4BATCH $(date -u) ==="
