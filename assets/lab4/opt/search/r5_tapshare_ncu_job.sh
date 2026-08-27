#!/bin/bash
# R5 tap-sharing premise check: ncu on global_interp_multi_kernel (MassPAng
# path) in the DEPLOYED A381 state. Question: after A38-1 fused-z (ya[216]
# local array eliminated, stack 2352->624B), is the analysis kernel still
# L2-pipe bound? If L2 is no longer saturated -> tap-sharing load-reduction
# premise is weakened -> -10~25s estimate must be revised.
set -uo pipefail
BASE=/home/h3240101033/lab4-gpu
EV=$BASE/evidence/r5-tapshare-ncu
mkdir -p "$EV"; exec > "$EV/job.log" 2>&1
echo "=== START R5-TAPSHARE-NCU $(date -u) ==="
nvidia-smi -L 2>/dev/null | head -2

export OMP_NUM_THREADS=8 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ulimit -s unlimited
export AMSS_ENABLE_GPU=ON AMSS_ENABLE_TWOP_GPU=OFF AMSS_ENABLE_TWOP_OMP_TUNE=ON
export AMSS_ENABLE_PACKED_RELAX=ON AMSS_ENABLE_TWOP_COS_TABLE=ON
export AMSS_CUDA_ARCHITECTURES=80 AMSS_OPT="-O3"
export CUDACXX=/usr/local/cuda-13.3/bin/nvcc
export AMSS_BUILD_DIR="$BASE/build-r5ncu"
export AMSS_OUTPUT_ROOT="$BASE"

# verify deployed state hash
sha256sum "$BASE/src/fmisc.h" "$BASE/src/fmisc_gpu.cu" | head -2

# build (clean)
cd "$BASE"
rm -rf "$AMSS_BUILD_DIR"
./compile.sh > "$EV/build.log" 2>&1 && echo BUILD_OK || { echo BUILD_FAIL; grep -iE "error" "$EV/build.log" | head -10; exit 3; }

# set input to 2 steps
python3 - <<'PY'
import re
s=open('AMSS_NCKU_Input.py').read()
s=re.sub(r'Final_Evolution_Time\s*=.*','Final_Evolution_Time = 2.0',s)
s=re.sub(r'Analysis_Time\s*=.*','Analysis_Time = 0.1',s)
open('AMSS_NCKU_Input.py','w').write(s)
PY

# 2-step run under ncu, capture global_interp_multi_kernel metrics
rm -rf "$BASE/GW250118" "$BASE/GW250118_ncu"
export AMSS_BUILD_DIR="$BASE/build-r5ncu"
ncu --kernel-name-base demangled --kernel-name regex:global_interp --launch-skip 30 --launch-count 3 \
    --section SpeedOfLight --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis \
    --section SchedulerStats --section WarpStateStats \
    --csv ./run.sh --twop-cache > "$EV/ncu.csv" 2>"$EV/ncu.err" || echo "ncu exited $?"
echo "=== ncu stderr tail ==="; tail -5 "$EV/ncu.err"
echo "=== key metrics ==="
python3 - <<'PY'
import csv, glob
try:
    rows = list(csv.reader(open("$EV/ncu.csv")))
except Exception as e:
    print("csv parse fail", e); raise SystemExit
# find metric rows for global_interp_multi_kernel
for r in rows:
    name = r[1] if len(r)>1 else ""
    if "global_interp" in name or "Name" in r:
        print(",".join(r[:4]))
PY
echo "=== restore input ==="
python3 - <<'PY'
import re
s=open('AMSS_NCKU_Input.py').read()
s=re.sub(r'Final_Evolution_Time\s*=.*','Final_Evolution_Time = 100.0',s)
open('AMSS_NCKU_Input.py','w').write(s)
PY
echo "=== DONE R5-TAPSHARE-NCU $(date -u) ==="
