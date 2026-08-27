#!/bin/bash
# Iter38 A38-2 sommerfeld fused-z L0+L1 A/B (stacked on A38-1).
set -uo pipefail
CAND=$(cut -d= -f2 ~/cand-a382-path.txt)
A381=$(cut -d= -f2 ~/cand-a381-path.txt)
BASEBUILD=$A381/build-a381
CACHE=/home/h3240101033/lab4-gpu-cand-p26bcd-combined-20260826-103614/twopuncture_cache
EV=$CAND/evidence/a382
mkdir -p "$EV"; exec > "$EV/job.log" 2>&1
echo "=== START A382 $CAND $(date -u) ==="
nvidia-smi -L 2>/dev/null | head -2

export OMP_NUM_THREADS=8 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ulimit -s unlimited
export AMSS_ENABLE_GPU=ON AMSS_ENABLE_TWOP_GPU=OFF AMSS_ENABLE_TWOP_OMP_TUNE=ON
export AMSS_ENABLE_PACKED_RELAX=ON AMSS_ENABLE_TWOP_COS_TABLE=ON
export AMSS_CUDA_ARCHITECTURES=80 AMSS_OPT="-O3"
export CUDACXX=/usr/local/cuda-13.3/bin/nvcc
export AMSS_CACHE_DIR="$CACHE"

cd "$CAND"
export AMSS_BUILD_DIR="$CAND/build-a382"
rm -rf "$AMSS_BUILD_DIR"
./compile.sh -DCMAKE_CUDA_FLAGS="-Xptxas -v" > "$EV/build-a382.log" 2>&1 \
  && echo BUILD_OK || { echo BUILD_FAIL; grep -iE "error" "$EV/build-a382.log" | head -20; exit 3; }
echo "=== sommerfeld ptxas (candidate A38-2) ==="
for K in sommerfeld_rout_compact_kernel sommerfeld_routbam_kernel; do
  echo "-- $K"; grep -A8 "Function.*$K" "$EV/build-a382.log" | grep -iE "registers|spill|stack" | head -3
done

python3 - <<'PY'
import re
s=open('AMSS_NCKU_Input.py').read()
s=re.sub(r'Final_Evolution_Time\s*=.*','Final_Evolution_Time = 2.0',s)
s=re.sub(r'Analysis_Time\s*=.*','Analysis_Time = 0.1',s)
open('AMSS_NCKU_Input.py','w').write(s)
PY

run_v() {
  local v="$1" b="$2"
  export AMSS_BUILD_DIR="$b" AMSS_OUTPUT_ROOT="$CAND" AMSS_CACHE_DIR="$CACHE"
  rm -rf "$CAND/GW250118_$v" "$CAND/GW250118"
  ./run.sh --twop-cache > "$EV/run-$v.log" 2>&1 || echo "run $v exited"
  cp -r "$CAND/GW250118/AMSS_NCKU_output" "$CAND/GW250118_$v" 2>/dev/null
  echo "--- $v ---"; grep -E "After Step: [12] My Rank" "$EV/run-$v.log" | tail -2
}
run_v base1 "$BASEBUILD"
run_v cand1 "$CAND/build-a382"
run_v cand2 "$CAND/build-a382"
run_v base2 "$BASEBUILD"

echo "=== BIT-EXACT ==="
BITEXACT=1
for pair in "base1 cand1" "base2 cand2"; do
  set -- $pair; a="$CAND/GW250118_$1"; b="$CAND/GW250118_$2"
  for fn in bssn_BH.dat bssn_psi4.dat bssn_ADMQs.dat bssn_constraint.dat; do
    if [ -f "$a/$fn" ] && [ -f "$b/$fn" ]; then
      if diff <(tail -n+2 "$a/$fn") <(tail -n+2 "$b/$fn") >/dev/null; then echo "$1-$2 $fn IDENTICAL"; else echo "$1-$2 $fn DIFFER"; BITEXACT=0; fi
    else echo "$1-$2 $fn missing"; BITEXACT=0; fi
  done
done

S2_BASE=$(grep -h "After Step: 2 My Rank" "$EV/run-base1.log" "$EV/run-base2.log" | awk '{print $(NF-1)}' | sort -n | awk '{a[NR]=$1} END{print (NR%2? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2)}')
S2_CAND=$(grep -h "After Step: 2 My Rank" "$EV/run-cand1.log" "$EV/run-cand2.log" | awk '{print $(NF-1)}' | sort -n | awk '{a[NR]=$1} END{print (NR%2? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2)}')
echo "=== F = base_median/cand_median ==="
echo "Step2 base median: $S2_BASE  Step2 cand median: $S2_CAND"
F=$(python3 -c "print('%.4f' % ($S2_BASE/$S2_CAND))" 2>/dev/null || echo "NA")
echo "F(speedup) = $F"
python3 -c "import sys; sys.exit(0 if float('$F')>1.0 else 1)" 2>/dev/null && GATE=1 || GATE=0
if [ "$BITEXACT" -ne 1 ] || [ "$GATE" -ne 1 ]; then
  echo "=== GATE FAILED (bit-exact=$BITEXACT F=$F): no Level-2 ==="
else
  echo "=== GATE PASSED (bit-exact=1 F=$F): Level-2 candidate ==="
fi

python3 - <<'PY'
import re
s=open('AMSS_NCKU_Input.py').read()
s=re.sub(r'Final_Evolution_Time\s*=.*','Final_Evolution_Time = 100.0',s)
s=re.sub(r'Analysis_Time\s*=.*','Analysis_Time = 0.1',s)
open('AMSS_NCKU_Input.py','w').write(s)
PY
echo "=== DONE A382 $(date -u) ==="
