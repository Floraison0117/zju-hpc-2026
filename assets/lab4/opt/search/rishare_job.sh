#!/bin/bash
# P-RISHARE: rhs_interior cross-call field read sharing (ptxas-gated probe).
#   Mechanism: hoist betax/betay/betaz center reads once per thread in
#   rhs_kernel_int, thread through 24 d_lopsided_point calls (72 loads -> 3).
#   GATE (task-specified): ptxas spill rise >5% vs deployed baseline -> dead
#   without A/B.  If gate passes (spill <= +5%), run L1 A/B.
set -uo pipefail
ROOT=$(ls -dt ~/lab4-gpu-cand-ri-share-* 2>/dev/null | head -1)
BASEBUILD=/home/h3240101033/lab4-gpu-cand-p26bcd-combined-20260826-103614/build-p26bcd
CACHE=/home/h3240101033/lab4-gpu-cand-p26bcd-combined-20260826-103614/twopuncture_cache
cd "$ROOT" || { echo "NO CANDIDATE"; exit 1; }
export OMP_NUM_THREADS=8 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ulimit -s unlimited
EV="$ROOT/evidence/rishare"
mkdir -p "$EV"; exec > "$EV/job.log" 2>&1
echo "=== START RISHARE $(date -u) ROOT=$ROOT ==="
nvidia-smi -L 2>/dev/null | head -2
ls -lh "$BASEBUILD/ABEGPU" || { echo NO_BASE_BUILD; exit 2; }
export AMSS_ENABLE_GPU=ON AMSS_ENABLE_TWOP_GPU=OFF AMSS_ENABLE_TWOP_OMP_TUNE=ON
export AMSS_ENABLE_PACKED_RELAX=ON AMSS_ENABLE_TWOP_COS_TABLE=ON
export AMSS_CUDA_ARCHITECTURES=80 AMSS_OPT="-O3"
export AMSS_CACHE_DIR="$CACHE"
if [ -x /usr/local/cuda-13.3/bin/nvcc ]; then export CUDACXX=/usr/local/cuda-13.3/bin/nvcc
elif [ -x /usr/local/cuda/bin/nvcc ]; then export CUDACXX=/usr/local/cuda/bin/nvcc; fi

# --- Level-0: candidate build with ptxas -v ---
export AMSS_BUILD_DIR="$ROOT/build-rishare"
rm -rf "$AMSS_BUILD_DIR"
./compile.sh -DCMAKE_CUDA_FLAGS="-Xptxas -v" > "$EV/build-rishare.log" 2>&1 \
  && echo "BUILD_RISHARE_OK" || { echo "BUILD_RISHARE_FAIL"; grep -iE "error" "$EV/build-rishare.log" | head -20; exit 3; }

# --- ptxas gate: compare rhs_kernel_int spill vs baseline numbers ---
# Baseline (deployed 26bcd, iter26a): rhs_kernel_int 128 regs / 712B stack /
# 2164B spill stores / 3328B spill loads.
echo "=== rhs_kernel_int ptxas (candidate) ==="
grep -A4 "Function.*rhs_kernel_int" "$EV/build-rishare.log" | grep -iE "registers|spill|stack" | head -4
SP_ST=$(grep -A4 "Function.*rhs_kernel_int" "$EV/build-rishare.log" | grep -iE "spill stores" | grep -oE "[0-9]+ bytes spill stores" | grep -oE "[0-9]+" | head -1)
SP_LD=$(grep -A4 "Function.*rhs_kernel_int" "$EV/build-rishare.log" | grep -iE "spill loads" | grep -oE "[0-9]+ bytes spill loads" | grep -oE "[0-9]+" | head -1)
echo "candidate rhs_kernel_int spill stores: ${SP_ST:-NA}  spill loads: ${SP_LD:-NA}"
echo "baseline (26bcd): spill stores 2164 / spill loads 3328"
# gate: rise >5% in either component -> dead
python3 - <<PY
st = ${SP_ST:-9999}; ld = ${SP_LD:-9999}
base_st, base_ld = 2164, 3328
r_st = (st - base_st) / base_st * 100
r_ld = (ld - base_ld) / base_ld * 100
print(f"spill stores delta: {r_st:+.1f}%  spill loads delta: {r_ld:+.1f}%")
if r_st > 5.0 or r_ld > 5.0:
    print("GATE: FAIL (spill rise >5%) -> dead, no A/B")
else:
    print("GATE: PASS (spill <=5%) -> proceed to L1 A/B")
PY

# --- 2-step config, analysis OFF ---
python3 - <<'PY'
import re
s=open('AMSS_NCKU_Input.py').read()
s=re.sub(r'Final_Evolution_Time\s*=.*','Final_Evolution_Time = 2.0',s)
s=re.sub(r'Analysis_Time\s*=.*','Analysis_Time = 1000.0',s)
open('AMSS_NCKU_Input.py','w').write(s)
PY

# --- Level-1: 4-round interleaved A/B (only if gate passed) ---
GATE_OK=$(grep -c "GATE: PASS" "$EV/job.log")
if [ "$GATE_OK" -eq 1 ]; then
  run_v() {
    local v="$1" b="$2"
    export AMSS_BUILD_DIR="$b" AMSS_OUTPUT_ROOT="$ROOT" AMSS_CACHE_DIR="$CACHE"
    rm -rf "$ROOT/GW250118_$v" "$ROOT/GW250118"
    ./run.sh --twop-cache > "$EV/run-$v.log" 2>&1 || echo "run $v exited"
    cp -r "$ROOT/GW250118/AMSS_NCKU_output" "$ROOT/GW250118_$v" 2>/dev/null
    echo "--- $v ---"; grep -E "After Step: [12] My Rank" "$EV/run-$v.log" | tail -2
  }
  run_v base1 "$BASEBUILD"
  run_v cand1 "$ROOT/build-rishare"
  run_v cand2 "$ROOT/build-rishare"
  run_v base2 "$BASEBUILD"
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
  S2_BASE=$(grep -h "After Step: 2 My Rank" "$EV/run-base1.log" "$EV/run-base2.log" | awk '{print $(NF-1)}' | sort -n | awk '{a[NR]=$1} END{print (NR%2? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2)}')
  S2_CAND=$(grep -h "After Step: 2 My Rank" "$EV/run-cand1.log" "$EV/run-cand2.log" | awk '{print $(NF-1)}' | sort -n | awk '{a[NR]=$1} END{print (NR%2? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2)}')
  echo "Step2 base median: $S2_BASE  Step2 cand median: $S2_CAND"
  F=$(python3 -c "print('%.4f' % ($S2_BASE/$S2_CAND))" 2>/dev/null || echo "NA")
  echo "F(speedup) = $F"
else
  echo "=== GATE FAILED at ptxas: skipping A/B ==="
fi

# --- restore OJ config ---
python3 - <<'PY'
import re
s=open('AMSS_NCKU_Input.py').read()
s=re.sub(r'Final_Evolution_Time\s*=.*','Final_Evolution_Time = 100.0',s)
s=re.sub(r'Analysis_Time\s*=.*','Analysis_Time = 0.1',s)
open('AMSS_NCKU_Input.py','w').write(s)
PY
echo "=== DONE RISHARE $(date -u) ==="
