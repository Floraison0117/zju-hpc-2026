#!/bin/bash
# P313233 deploy part 2: formal already has the 9 candidate files installed and
# snapshot exists. Just rebuild + 100-step OJ-sim + check.sh + clean.
set -uo pipefail
BASE=/home/h3240101033/lab4-gpu
cd "$BASE" || exit 1
export OMP_NUM_THREADS=8 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ulimit -s unlimited
EV="$BASE/evidence/p313233-deploy2-$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$EV"; exec > "$EV/job.log" 2>&1
echo "=== START P313233_DEPLOY2 $(date -u) ==="
nvidia-smi -L 2>/dev/null | head -2
export AMSS_ENABLE_GPU=ON AMSS_ENABLE_TWOP_GPU=OFF AMSS_ENABLE_TWOP_OMP_TUNE=ON
export AMSS_ENABLE_PACKED_RELAX=ON AMSS_ENABLE_TWOP_COS_TABLE=ON
export AMSS_CUDA_ARCHITECTURES=80 AMSS_OPT="-O3"
export AMSS_OUTPUT_ROOT="$BASE"
if [ -x /usr/local/cuda-13.3/bin/nvcc ]; then export CUDACXX=/usr/local/cuda-13.3/bin/nvcc
elif [ -x /usr/local/cuda/bin/nvcc ]; then export CUDACXX=/usr/local/cuda/bin/nvcc; fi

# verify the 9 installed files match candidate hashes
echo "=== installed file hashes ==="
sha256sum src/fmisc_gpu.cu src/fmisc.h src/MPatch_gpu.cu src/Parallel_GPU.cpp \
          src/sommerfeld_rout_gpu.cu src/sommerfeld_rout.h src/bssn_step_gpu.C \
          src/prolongrestrict_cell_gpu.cu src/prolongrestrict_cell_gpu_int.cu | awk '{print substr($1,1,8), $2}'

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

# clean the formal into the submission package (9 items)
echo "=== cleaning submission package ==="
rm -rf GW250118 AMSS_NCKU_output Ansorg.psid twopuncture_cache build evidence __pycache__ \
       *.tar.gz .remote_work 2>/dev/null
ls -la "$BASE" | head -14
echo "=== DONE P313233_DEPLOY2 $(date -u) ==="
