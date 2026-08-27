#!/bin/bash
# A381 deploy: snapshot formal, install fmisc.h + fmisc_gpu.cu from candidate,
# rebuild, 100-step OJ-sim + check.sh. Logs to ~/a381-deploy-<ts>/ (outside formal).
set -uo pipefail
BASE=/home/h3240101033/lab4-gpu
CAND=~/lab4-gpu-cand-a381-fusedz-20260827-054736
cd "$BASE" || exit 1
export OMP_NUM_THREADS=8 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ulimit -s unlimited
EV=~/a381-deploy-$(date -u +%Y%m%d-%H%M%S)
mkdir -p "$EV"; exec > "$EV/job.log" 2>&1
echo "=== START A381_DEPLOY $(date -u) ==="
nvidia-smi -L 2>/dev/null | head -2
export AMSS_ENABLE_GPU=ON AMSS_ENABLE_TWOP_GPU=OFF AMSS_ENABLE_TWOP_OMP_TUNE=ON
export AMSS_ENABLE_PACKED_RELAX=ON AMSS_ENABLE_TWOP_COS_TABLE=ON
export AMSS_CUDA_ARCHITECTURES=80 AMSS_OPT="-O3"
export AMSS_OUTPUT_ROOT="$BASE"
if [ -x /usr/local/cuda-13.3/bin/nvcc ]; then export CUDACXX=/usr/local/cuda-13.3/bin/nvcc
elif [ -x /usr/local/cuda/bin/nvcc ]; then export CUDACXX=/usr/local/cuda/bin/nvcc; fi

# hash guard: formal must be P313233 baseline
FB=$(sha256sum src/fmisc.h | awk '{print $1}')
echo "formal fmisc.h $FB"
case "$FB" in abec6936*) ;; *) echo "FORMAL DRIFTED"; exit 3;; esac

# snapshot
SNAP="$BASE-snapshot-pre-a381-$(date +%Y%m%d-%H%M%S)"
cp -r "$BASE" "$SNAP" 2>/dev/null && echo "snapshot -> $SNAP" || { echo "SNAPSHOT_FAIL"; exit 5; }

# install 2 files
for f in fmisc.h fmisc_gpu.cu; do
  cp "$CAND/src/$f" src/$f && echo "installed $f $(sha256sum src/$f | awk '{print $1}')"
done

# rebuild
export AMSS_BUILD_DIR="$BASE/build"
rm -rf "$AMSS_BUILD_DIR"
./compile.sh > "$EV/build.log" 2>&1 && echo "BUILD_OK" || { echo "BUILD_FAIL"; grep -iE "error" "$EV/build.log" | head -10; exit 6; }

# 100-step OJ-sim (TwoP live)
rm -rf GW250118 AMSS_NCKU_output Ansorg.psid twopuncture_cache
./run.sh > "$EV/run.log" 2>&1 || echo "run exited"
grep -E "Total Evolve Time|This Program Cost|After Step: (1|50|100) " "$EV/run.log" | tail -5

# check.sh
OUT="$BASE/GW250118/AMSS_NCKU_output"
RESULT_DIR="$OUT" ./check.sh "$OUT" "$BASE/golden" > "$EV/check.log" 2>&1 || echo "check exited"
grep -iE "FINAL|PASS|FAIL|RMS|constraint maxima|Trajectory" "$EV/check.log" | tail -8

# clean scratch but keep evidence dir empty for packaging (main agent cleans final)
rm -rf GW250118 AMSS_NCKU_output Ansorg.psid twopuncture_cache build __pycache__ build-reprofile
echo "=== all logs in $EV ==="
ls -la "$EV"
echo "=== DONE A381_DEPLOY $(date -u) ==="
