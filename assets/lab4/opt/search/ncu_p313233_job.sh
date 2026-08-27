#!/bin/bash
# Iter38 ncu micro-profile on deployed P313233 state (v3: no --kill; 2-step run
# completes normally so the .ncu-rep reports flush cleanly).
set -uo pipefail
ROOT=/home/h3240101033/lab4-gpu
TS=$(date +%Y%m%d-%H%M%S)
EV=$ROOT/evidence/ncu-p313233-$TS
mkdir -p "$EV"
exec > "$EV/job.log" 2>&1
echo "=== START NCU $(date -u) TS=$TS ==="
nvidia-smi -L 2>/dev/null | head -2

export OMP_NUM_THREADS=8 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ulimit -s unlimited
export AMSS_ENABLE_GPU=ON AMSS_ENABLE_TWOP_GPU=OFF AMSS_ENABLE_TWOP_OMP_TUNE=ON
export AMSS_ENABLE_PACKED_RELAX=ON AMSS_ENABLE_TWOP_COS_TABLE=ON
export AMSS_CUDA_ARCHITECTURES=80 AMSS_OPT="-O3"
export AMSS_BUILD_DIR="$ROOT/build-reprofile"
export AMSS_CACHE_DIR="/home/h3240101033/lab4-gpu-cand-p26bcd-combined-20260826-103614/twopuncture_cache"
cd "$ROOT"
cp AMSS_NCKU_Input.py "$EV/Input.backup.py"
python3 - <<'PY'
import re
s=open('AMSS_NCKU_Input.py').read()
s=re.sub(r'Final_Evolution_Time\s*=.*','Final_Evolution_Time = 2.0',s)
s=re.sub(r'Analysis_Time\s*=.*','Analysis_Time = 0.1',s)
open('AMSS_NCKU_Input.py','w').write(s)
PY

NCU="ncu --clock-control none --section SpeedOfLight --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats --section Occupancy"
run_k() {
  local K="$1" SKIP="$2" CNT="$3"
  echo "=== $K skip=$SKIP count=$CNT ==="
  export AMSS_OUTPUT_ROOT="$EV/runroot-$K"
  rm -rf "$AMSS_OUTPUT_ROOT"; mkdir -p "$AMSS_OUTPUT_ROOT"
  timeout 600 $NCU --kernel-name-base demangled -k "regex:$K" --launch-skip "$SKIP" --launch-count "$CNT" \
    --target-processes all -o "$EV/$K" \
    ./run.sh --twop-cache > "$EV/ncu-$K.log" 2>&1
  echo "ncu rc=$? (0 expected)"
  grep -E "This Program Cost|Total Evolve" "$EV/ncu-$K.log" | tail -1
}
run_k global_interp_multi_kernel 0 12
run_k rhs_kernel_facepure 0 2
run_k rhs_kernel_facez 0 2
run_k rhs_kernel_int 0 1
run_k restrict3_kernel 0 2
run_k rungekutta4_rout_kernel 0 2

cp "$EV/Input.backup.py" AMSS_NCKU_Input.py
ls -la "$EV/" | grep ncu-rep
echo "=== DONE NCU $(date -u) ==="
