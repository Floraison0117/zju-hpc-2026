#!/bin/bash
# R5 tap-sharing premise check v2: ncu on global_interp_multi_kernel.
# Fixed: launch-skip 5 / count 2, --target-processes all, -o report file.
set -uo pipefail
BASE=/home/h3240101033/lab4-gpu
EV=$BASE/evidence/r5-tapshare-ncu2
mkdir -p "$EV"; exec > "$EV/job.log" 2>&1
echo "=== START R5-TAPSHARE-NCU2 $(date -u) ==="
nvidia-smi -L 2>/dev/null | head -2

export OMP_NUM_THREADS=8 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ulimit -s unlimited
export AMSS_ENABLE_GPU=ON AMSS_ENABLE_TWOP_GPU=OFF AMSS_ENABLE_TWOP_OMP_TUNE=ON
export AMSS_ENABLE_PACKED_RELAX=ON AMSS_ENABLE_TWOP_COS_TABLE=ON
export AMSS_CUDA_ARCHITECTURES=80 AMSS_OPT="-O3"
export CUDACXX=/usr/local/cuda-13.3/bin/nvcc
export AMSS_BUILD_DIR="$BASE/build-r5ncu"
export AMSS_OUTPUT_ROOT="$BASE"

cd "$BASE"
[ -d "$AMSS_BUILD_DIR" ] || { ./compile.sh > "$EV/build.log" 2>&1 && echo BUILD_OK || { echo BUILD_FAIL; grep -iE "error" "$EV/build.log"|head; exit 3; }; }

python3 - <<'PY'
import re
s=open('AMSS_NCKU_Input.py').read()
s=re.sub(r'Final_Evolution_Time\s*=.*','Final_Evolution_Time = 2.0',s)
s=re.sub(r'Analysis_Time\s*=.*','Analysis_Time = 0.1',s)
open('AMSS_NCKU_Input.py','w').write(s)
PY

rm -rf "$BASE/GW250118"
export AMSS_BUILD_DIR="$BASE/build-r5ncu"
ncu --clock-control none --target-processes all --kernel-name regex:global_interp_multi \
    --launch-skip 5 --launch-count 2 \
    --section SpeedOfLight --section LaunchStats --section MemoryWorkloadAnalysis \
    --section SchedulerStats --section WarpStateStats \
    -o "$EV/report" ./run.sh --twop-cache > "$EV/ncu.stdout" 2>&1 || echo "ncu exited $?"
echo "=== ncu stdout tail ==="; tail -8 "$EV/ncu.stdout"
if [ -f "$EV/report.ncu-rep" ]; then
  ncu --import "$EV/report.ncu-rep" --csv > "$EV/metrics.csv" 2>/dev/null && echo IMPORT_OK || echo IMPORT_FAIL
  echo "=== key metrics ==="
  grep -iE "global_interp_multi|lts__throughput|l1tex__throughput|stall|Duration|elapsed" "$EV/metrics.csv" | head -15
else
  echo "NO REPORT FILE"; ls -la "$EV/"
fi
python3 - <<'PY'
import re
s=open('AMSS_NCKU_Input.py').read()
s=re.sub(r'Final_Evolution_Time\s*=.*','Final_Evolution_Time = 100.0',s)
open('AMSS_NCKU_Input.py','w').write(s)
PY
echo "=== DONE R5-TAPSHARE-NCU2 $(date -u) ==="
