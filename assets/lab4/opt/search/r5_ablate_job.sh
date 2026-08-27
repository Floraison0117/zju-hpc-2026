#!/bin/bash
# R5 L0 probe: Ricci Step-4 ablation on thin interior kernel.
# Compare base vs ablate at lb2 + nolb. Diagnostic: is Ricci the reg peak?
set -uo pipefail
BASE=/home/h3240101033/lab4-gpu
CAND=$(cut -d= -f2 ~/cand-r5-path.txt 2>/dev/null)
EV=$CAND/evidence/r5-l0
mkdir -p "$EV"; exec > "$EV/job4.log" 2>&1
echo "=== START R5-ABLATE $(date -u) ==="
export OMP_NUM_THREADS=8 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ulimit -s unlimited
export CUDACXX=/usr/local/cuda-13.3/bin/nvcc
NVCC=$CUDACXX
DEFS="-DMPI_CUDA_AWARE=0 -DUSE_GPU -Dfortran3 -Dnewc"
FLAGS="-O3 -Xptxas -v -std=c++14 --generate-code=arch=compute_80,code=[compute_80,sm_80] -rdc=true -lineinfo"
INC_B="-I$BASE/src -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include"
INC_C="-I$CAND/src -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include"

mkdir -p /tmp/r5-ab
cp "$BASE/src/bssn_rhs_gpu_int.cu" /tmp/r5-ab/base.cu
cp "$CAND/src/bssn_rhs_gpu_int_ablate.cu" /tmp/r5-ab/ablate.cu

compile_one() { # $1=label $2=src $3=inc $4=nolb
  local lbl=$1 src=$2 inc=$3 nolb=$4
  if [ "$nolb" = "1" ]; then
    sed "s/__global__ __launch_bounds__(256, 2) void rhs_kernel_int_ablate/__global__ void rhs_kernel_int_ablate/" "$src" > /tmp/r5-ab/${lbl}.cu
    src=/tmp/r5-ab/${lbl}.cu
  fi
  $NVCC -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++-13 $DEFS $inc $FLAGS -x cu -rdc=true -c "$src" -o /tmp/r5-ab/$lbl.o > "$EV/ptxas-$lbl.log" 2>&1 \
    || { echo "COMPILE_FAIL $lbl"; tail -6 "$EV/ptxas-$lbl.log"; return 1; }
  echo "-- $lbl:"
  grep -A8 "Compiling entry function .*rhs_kernel_int_ablate" "$EV/ptxas-$lbl.log" | grep -oE "Used [0-9]+ registers|[0-9]+ bytes spill stores|[0-9]+ bytes spill loads|[0-9]+ bytes stack frame" | head -5
}
echo "=== ablate-lb2 (128 regs) ==="; compile_one ablate-lb2 /tmp/r5-ab/ablate.cu "$INC_C" 0
echo "=== ablate-nolb (natural) ==="; compile_one ablate-nolb /tmp/r5-ab/ablate.cu "$INC_C" 1
echo "=== reference: base-lb2 ==="
grep -A8 "Compiling entry function .*rhs_kernel_int " "$EV/ptxas-base-lb2.log" | grep -oE "Used [0-9]+ registers|[0-9]+ bytes spill stores|[0-9]+ bytes spill loads" | head -3
echo "=== reference: base-nolb (job3) ==="
grep -A8 "Compiling entry function .*rhs_kernel_int " "$EV/ptxas-base-nolb.log" | grep -oE "Used [0-9]+ registers|[0-9]+ bytes spill stores|[0-9]+ bytes spill loads" | head -3
echo "=== DONE R5-ABLATE $(date -u) ==="
