#!/bin/bash
# P313233 deploy: snapshot formal, install the combined candidate (P31+P32+P33),
# rebuild, verify with 100-step OJ-sim + check.sh, then clean the formal into
# the submission package (9 items). Modeled on p26bcd_deploy.sh.
set -uo pipefail
BASE=/home/h3240101033/lab4-gpu
CAND=~/lab4-gpu-cand-p313233-comb-20260826-171357
cd "$BASE" || exit 1
export OMP_NUM_THREADS=8 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ulimit -s unlimited
EV="$BASE/evidence/p313233-deploy-$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$EV"; exec > "$EV/job.log" 2>&1
echo "=== START P313233_DEPLOY $(date -u) CAND=$CAND ==="
nvidia-smi -L 2>/dev/null | head -2
export AMSS_ENABLE_GPU=ON AMSS_ENABLE_TWOP_GPU=OFF AMSS_ENABLE_TWOP_OMP_TUNE=ON
export AMSS_ENABLE_PACKED_RELAX=ON AMSS_ENABLE_TWOP_COS_TABLE=ON
export AMSS_CUDA_ARCHITECTURES=80 AMSS_OPT="-O3"
export AMSS_OUTPUT_ROOT="$BASE"
if [ -x /usr/local/cuda-13.3/bin/nvcc ]; then export CUDACXX=/usr/local/cuda-13.3/bin/nvcc
elif [ -x /usr/local/cuda/bin/nvcc ]; then export CUDACXX=/usr/local/cuda/bin/nvcc; fi

# 0. hash guard: formal must match the 26bcd deployed baseline
for f in bssn_rhs_gpu.cu derivatives.h gpu_manager.cu prolongrestrict_cell_gpu.cu; do
  h=$(sha256sum src/$f | awk '{print $1}')
  echo "formal $f $h"
done
FB=$(sha256sum src/bssn_rhs_gpu.cu | awk '{print $1}')
case "$FB" in 9baee005*) ;; *) echo "FORMAL DRIFTED"; exit 3;; esac

# 1. snapshot
SNAP="$BASE-snapshot-pre-p313233-$(date +%Y%m%d-%H%M%S)"
cp -r "$BASE" "$SNAP" 2>/dev/null && echo "snapshot -> $SNAP" || { echo "SNAPSHOT_FAIL"; exit 5; }

# 2. install the 9 changed files from the candidate
for f in fmisc_gpu.cu fmisc.h MPatch_gpu.cu Parallel_GPU.cpp sommerfeld_rout_gpu.cu \
         sommerfeld_rout.h bssn_step_gpu.C prolongrestrict_cell_gpu.cu prolongrestrict_cell_gpu_int.cu; do
  if [ -f "$CAND/src/$f" ]; then cp "$CAND/src/$f" src/$f; echo "installed src/$f $(sha256sum src/$f | awk '{print $1}')"
  elif [ -f "$CAND/$f" ]; then cp "$CAND/$f" $f; echo "installed $f $(sha256sum $f | awk '{print $1}')"; fi
done

# 3. rebuild
export AMSS_BUILD_DIR="$BASE/build"
rm -rf "$AMSS_BUILD_DIR"
./compile.sh > "$EV/build.log" 2>&1 && echo "BUILD_OK" || { echo "BUILD_FAIL"; grep -iE "error" "$EV/build.log" | head -10; exit 6; }

# 4. 100-step OJ-sim (TwoP live)
rm -rf GW250118 AMSS_NCKU_output Ansorg.psid twopuncture_cache
./run.sh > "$EV/run.log" 2>&1 || echo "run exited"
grep -E "Total Evolve Time|This Program Cost|After Step: (1|50|100) " "$EV/run.log" | tail -5

# 5. check.sh
OUT="$BASE/GW250118/AMSS_NCKU_output"
RESULT_DIR="$OUT" ./check.sh "$OUT" "$BASE/golden" > "$EV/check.log" 2>&1 || echo "check exited"
grep -iE "FINAL|PASS|FAIL|RMS|constraint maxima|Trajectory" "$EV/check.log" | tail -8

# 6. clean the formal into the submission package (9 items)
echo "=== cleaning submission package ==="
rm -rf GW250118 AMSS_NCKU_output Ansorg.psid twopuncture_cache build evidence __pycache__ \
       *.tar.gz .remote_work 2>/dev/null
ls -la "$BASE" | head -14
echo "=== DONE P313233_DEPLOY $(date -u) ==="
