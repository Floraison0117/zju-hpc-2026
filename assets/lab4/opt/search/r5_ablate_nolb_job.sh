#!/bin/bash
# R5: measure ablate kernel at natural regs (nolb) - fix sed kernel name.
set -uo pipefail
CAND=$(cut -d= -f2 ~/cand-r5-path.txt 2>/dev/null)
EV=$CAND/evidence/r5-l0
mkdir -p "$EV"; exec > "$EV/job5.log" 2>&1
echo "=== START R5-ABLATE-NOLB $(date -u) ==="
export OMP_NUM_THREADS=8 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export CUDACXX=/usr/local/cuda-13.3/bin/nvcc
NVCC=$CUDACXX
DEFS="-DMPI_CUDA_AWARE=0 -DUSE_GPU -Dfortran3 -Dnewc"
FLAGS="-O3 -Xptxas -v -std=c++14 --generate-code=arch=compute_80,code=[compute_80,sm_80] -rdc=true -lineinfo"
INC_C="-I$CAND/src -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include"
mkdir -p /tmp/r5-ab2
cp $CAND/src/bssn_rhs_gpu_int_ablate.cu /tmp/r5-ab2/ablate.cu
sed "s/__global__ __launch_bounds__(256, 2) void rhs_kernel_int/__global__ void rhs_kernel_int/" /tmp/r5-ab2/ablate.cu > /tmp/r5-ab2/ablate_nolb.cu
$NVCC -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++-13 $DEFS $INC_C $FLAGS -x cu -rdc=true -c /tmp/r5-ab2/ablate_nolb.cu -o /tmp/r5-ab2/ablate_nolb.o > $EV/ptxas-ablate-nolb2.log 2>&1 || { echo COMPILE_FAIL; tail -6 $EV/ptxas-ablate-nolb2.log; exit 1; }
echo "=== ablate-nolb2 (natural) ==="
grep -A4 "Function properties" $EV/ptxas-ablate-nolb2.log | grep -oE "[0-9]+ bytes stack frame|[0-9]+ bytes spill stores|[0-9]+ bytes spill loads" | head -3
grep "Used" $EV/ptxas-ablate-nolb2.log | head -1
echo "=== reference base-nolb ==="
grep -A4 "Function properties" $EV/ptxas-base-nolb.log | grep -oE "[0-9]+ bytes stack frame|[0-9]+ bytes spill stores|[0-9]+ bytes spill loads" | head -3
grep "Used" $EV/ptxas-base-nolb.log | head -1
echo "=== DONE $(date -u) ==="
