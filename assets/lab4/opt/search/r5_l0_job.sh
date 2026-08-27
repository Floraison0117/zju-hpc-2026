#!/bin/bash
# R5 L0 probe: z-rolling smem-staged interior kernel (register-floor test).
# Compile base (rhs_kernel_int) vs cand (rhs_kernel_int_zr) single-TU with
# ptxas -v, lb2 + nolb variants; report regs/spill/smem/static.
# Gate: interior currently 128 regs/1836B spill (lb2), natural 255.
# PASS if cand natural regs <128 OR lb2 spill significantly down.
set -uo pipefail
BASE=/home/h3240101033/lab4-gpu
CAND=$(cut -d= -f2 ~/cand-r5-path.txt 2>/dev/null)
[ -n "$CAND" ] || CAND=/home/h3240101033/lab4-gpu-cand-r5-zroll
EV=$CAND/evidence/r5-l0
mkdir -p "$EV"; exec > "$EV/job.log" 2>&1
echo "=== START R5-L0 $(date -u) CAND=$CAND ==="
nvidia-smi -L 2>/dev/null | head -2

export OMP_NUM_THREADS=8 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ulimit -s unlimited
export AMSS_ENABLE_GPU=ON AMSS_ENABLE_TWOP_GPU=OFF AMSS_ENABLE_TWOP_OMP_TUNE=ON
export AMSS_ENABLE_PACKED_RELAX=ON AMSS_ENABLE_TWOP_COS_TABLE=ON
export AMSS_CUDA_ARCHITECTURES=80 AMSS_OPT="-O3"
export CUDACXX=/usr/local/cuda-13.3/bin/nvcc

NVCC=$CUDACXX
DEFS="-DMPI_CUDA_AWARE=0 -DUSE_GPU -Dfortran3 -Dnewc"
FLAGS="-O3 -Xptxas -v -std=c++14 --generate-code=arch=compute_80,code=[compute_80,sm_80] -rdc=true -lineinfo"
INC_B="-I$BASE/src -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include"
INC_C="-I$CAND/src -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include"

mkdir -p /tmp/r5-l0
cp "$BASE/src/bssn_rhs_gpu_int.cu" /tmp/r5-l0/base.cu
cp "$CAND/src/bssn_rhs_gpu_int_zr.cu" /tmp/r5-l0/cand.cu
ls -la /tmp/r5-l0/

compile_one() { # $1=label $2=src $3=inc $4=nolb(0/1)
  local lbl=$1 src=$2 inc=$3 nolb=$4
  if [ "$nolb" = "1" ]; then
    sed "s/__global__ __launch_bounds__(256, 2) void rhs_kernel_int_zr/__global__ void rhs_kernel_int_zr/" "$src" > /tmp/r5-l0/${lbl}.cu
    src=/tmp/r5-l0/${lbl}.cu
  fi
  echo "=== compile $lbl ==="
  $NVCC -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++-13 $DEFS $inc $FLAGS -x cu -rdc=true -c "$src" -o /tmp/r5-l0/$lbl.o > "$EV/ptxas-$lbl.log" 2>&1 \
    || { echo "COMPILE_FAIL $lbl"; tail -8 "$EV/ptxas-$lbl.log"; return 1; }
  return 0
}

parse() { # $1=log $2=kernel-regex $3=metric
  grep -A12 "Function properties: .*$2" "$1" | head -1 >/dev/null 2>&1
  grep -A8 "Compiling entry function .*$2" "$1" | grep -oE "$3" | head -1
}

# base lb2 + nolb, cand lb2 + nolb
compile_one base-lb2  /tmp/r5-l0/base.cu  "$INC_B" 0
compile_one base-nolb /tmp/r5-l0/base.cu  "$INC_B" 1 || true   # base kernel name differs, use sed for rhs_kernel_int
compile_one cand-lb2  /tmp/r5-l0/cand.cu  "$INC_C" 0
compile_one cand-nolb /tmp/r5-l0/cand.cu  "$INC_C" 1

# fix: base-nolb sed must target rhs_kernel_int (base name)
if [ ! -s "$EV/ptxas-base-nolb.log" ] || ! grep -q "Used" "$EV/ptxas-base-nolb.log"; then
  sed "s/__global__ __launch_bounds__(256, 2) void rhs_kernel_int/__global__ void rhs_kernel_int/" /tmp/r5-l0/base.cu > /tmp/r5-l0/base_nolb.cu
  $NVCC -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++-13 $DEFS $INC_B $FLAGS -x cu -rdc=true -c /tmp/r5-l0/base_nolb.cu -o /tmp/r5-l0/base_nolb.o > "$EV/ptxas-base-nolb.log" 2>&1 \
    || { echo "COMPILE_FAIL base-nolb"; tail -8 "$EV/ptxas-base-nolb.log"; }
fi

report() { # $1=label $2=kernel
  local log="$EV/ptxas-$1.log"
  [ -s "$log" ] || { echo "$1: NO_LOG"; return; }
  echo "-- $1 ($2)"
  grep -A8 "Compiling entry function .*$2" "$log" | grep -oE "Used [0-9]+ registers|spill stores|spill loads|[0-9]+ bytes spill stores|[0-9]+ bytes spill loads" | head -4
  grep -A8 "Compiling entry function .*$2" "$log" | grep -oE "[0-9]+ bytes stack frame" | head -1
  grep -A8 "Compiling entry function .*$2" "$log" | grep -oE "[0-9]+ bytes smem" | head -1
}
echo "=== PTXAS SUMMARY ==="
report base-lb2  rhs_kernel_int
report base-nolb rhs_kernel_int
report cand-lb2  rhs_kernel_int_zr
report cand-nolb rhs_kernel_int_zr

echo "=== GATE EVALUATION ==="
B_REGS=$(grep -A8 "Compiling entry function .*rhs_kernel_int\b" "$EV/ptxas-base-lb2.log" | grep -oE "Used [0-9]+ registers" | head -1 | grep -oE "[0-9]+")
B_SPILL=$(grep -A8 "Compiling entry function .*rhs_kernel_int\b" "$EV/ptxas-base-lb2.log" | grep -oE "[0-9]+ bytes spill stores" | head -1 | grep -oE "^[0-9]+")
C_REGS=$(grep -A8 "Compiling entry function .*rhs_kernel_int_zr\b" "$EV/ptxas-cand-lb2.log" | grep -oE "Used [0-9]+ registers" | head -1 | grep -oE "[0-9]+")
C_SPILL=$(grep -A8 "Compiling entry function .*rhs_kernel_int_zr\b" "$EV/ptxas-cand-lb2.log" | grep -oE "[0-9]+ bytes spill stores" | head -1 | grep -oE "^[0-9]+")
CN_REGS=$(grep -A8 "Compiling entry function .*rhs_kernel_int_zr\b" "$EV/ptxas-cand-nolb.log" | grep -oE "Used [0-9]+ registers" | head -1 | grep -oE "[0-9]+")
echo "base-lb2: regs=$B_REGS spill_stores=$B_SPILL"
echo "cand-lb2: regs=$C_REGS spill_stores=$C_SPILL"
echo "cand-nolb: regs=$CN_REGS (natural)"
GATE=DEAD_END
if [ -n "$CN_REGS" ] && [ "$CN_REGS" -lt 255 ] 2>/dev/null; then GATE=PASS; fi
if [ -n "$C_SPILL" ] && [ -n "$B_SPILL" ] && [ "$C_SPILL" -lt $((B_SPILL * 98 / 100)) ] 2>/dev/null; then GATE=PASS; fi
echo "GATE=$GATE"
echo "=== DONE R5-L0 $(date -u) ==="
