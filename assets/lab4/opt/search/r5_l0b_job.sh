#!/bin/bash
# R5 L0 probe part 2: candidate at lb3(80 regs)/lb4(64 regs) vs base.
# Question: after smem-staging reduces natural-reg spill -79%, does higher
# occupancy (37.5%/50%) become viable? (base-lb3 was 13240B spill = dead)
set -uo pipefail
BASE=/home/h3240101033/lab4-gpu
CAND=$(cut -d= -f2 ~/cand-r5-path.txt 2>/dev/null)
EV=$CAND/evidence/r5-l0
mkdir -p "$EV"; exec > "$EV/job2.log" 2>&1
echo "=== START R5-L0b $(date -u) ==="
export OMP_NUM_THREADS=8 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ulimit -s unlimited
export CUDACXX=/usr/local/cuda-13.3/bin/nvcc
NVCC=$CUDACXX
DEFS="-DMPI_CUDA_AWARE=0 -DUSE_GPU -Dfortran3 -Dnewc"
FLAGS="-O3 -Xptxas -v -std=c++14 --generate-code=arch=compute_80,code=[compute_80,sm_80] -rdc=true -lineinfo"
INC_B="-I$BASE/src -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include"
INC_C="-I$CAND/src -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include"

mkdir -p /tmp/r5-l0b
cp "$BASE/src/bssn_rhs_gpu_int.cu" /tmp/r5-l0b/base.cu
cp "$CAND/src/bssn_rhs_gpu_int_zr.cu" /tmp/r5-l0b/cand.cu

compile_lb() { # $1=label $2=src $3=inc $4=minblocks $5=kernel-regex
  local lbl=$1 src=$2 inc=$3 n=$4 kre=$5
  sed "s/__launch_bounds__(256, 2)/__launch_bounds__(256, $n)/" "$src" > /tmp/r5-l0b/$lbl.cu
  $NVCC -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++-13 $DEFS $inc $FLAGS -x cu -rdc=true -c /tmp/r5-l0b/$lbl.cu -o /tmp/r5-l0b/$lbl.o > "$EV/ptxas-$lbl.log" 2>&1 \
    || { echo "COMPILE_FAIL $lbl"; tail -5 "$EV/ptxas-$lbl.log"; return 1; }
  echo "-- $lbl:"
  grep -A8 "Compiling entry function.*$kre" "$EV/ptxas-$lbl.log" | grep -oE "Used [0-9]+ registers|[0-9]+ bytes spill stores|[0-9]+ bytes spill loads|[0-9]+ bytes stack frame|[0-9]+ bytes smem" | head -6
  return 0
}
echo "=== base-lb3 (80 regs) ==="
compile_lb base-lb3 /tmp/r5-l0b/base.cu "$INC_B" 3 'rhs_kernel_int\b'
echo "=== cand-lb3 (80 regs) ==="
compile_lb cand-lb3 /tmp/r5-l0b/cand.cu "$INC_C" 3 'rhs_kernel_int_zr'
echo "=== cand-lb4 (64 regs) ==="
compile_lb cand-lb4 /tmp/r5-l0b/cand.cu "$INC_C" 4 'rhs_kernel_int_zr'
echo "=== base-lb4 (64 regs) ==="
compile_lb base-lb4 /tmp/r5-l0b/base.cu "$INC_B" 4 'rhs_kernel_int\b'
echo "=== DONE R5-L0b $(date -u) ==="
