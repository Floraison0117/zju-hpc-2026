#!/usr/bin/env bash
# Job B (v2): fine scale — Nsight Compute on _fused_dequant_gemm_kernel_v2.
# Previous run failed: --kernel-name does EXACT match, so "fused_dequant_gemm" did
# not match "_fused_dequant_gemm_kernel_v2". Fix: use regex: prefix.
# Confirms v2 kernel is executing (H3) and gives compute util / memory BW /
# occupancy / stall reasons for the 7.2 ms/launch decode hotspot.
# Partition lab5 (H800 MIG 1g.10gb), 30m wall.
set -u
cd /home/h3240101033/lab5-fix-dev
export PYTHONPATH=/home/h3240101033/lab5/src
PY=/opt/lab5-venv/bin/python
MODEL=/home/h3240101033/HPC101/src/lab5/results/gemma-4-12b-gptq-cholesky-w4a16
SMALL=/home/h3240101033/lab05_final/datasets/performance_small.jsonl
OUT=/home/h3240101033/lab5-fix-dev/profile-out
mkdir -p "$OUT"
SCRIPT=/home/h3240101033/lab5-fix-dev/profile_multiscale.py
KNAME="regex:_fused_dequant_gemm_kernel_v2"

echo "##### sanity + clear __pycache__ + py_compile #####"
$PY -c "import torch; print('dev',torch.cuda.get_device_name(0),'free',torch.cuda.mem_get_info()[0]/1024**3,'GiB')"
find /home/h3240101033/lab5/src -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
$PY -m py_compile "$SCRIPT" && echo "py_compile OK" || { echo "py_compile FAIL"; exit 1; }

echo ""
echo "########## SCALE 3 (FINE): ncu --set full on _fused_dequant_gemm_kernel_v2 ##########"
echo "##### kernel-name filter: $KNAME  (exact-name match failed before) #####"
# --launch-count 2: profile first 2 matching launches (replay mode collects all metrics).
# --kernel-name-base function (default) matches against the CUDA function name.
# --clock-control none keeps profiling fast; relative metrics still valid.
ncu --set full \
  --kernel-name "$KNAME" \
  --kernel-name-base function \
  --launch-skip 0 \
  --launch-count 2 \
  --target-processes all \
  --replay-mode kernel \
  --clock-control none \
  -o "$OUT/fine.ncu-rep" \
  --force-overwrite \
  "$PY" "$SCRIPT" --scale fine --model "$MODEL" --small "$SMALL" --out-dir "$OUT" --n-decode 4 \
  2>&1 | tee "$OUT/fine_ncu_stdout.txt"

echo ""
echo "##### also emit a text summary (per-kernel) so the report has readable metrics #####"
ncu --import "$OUT/fine.ncu-rep" --print-summary per-kernel 2>&1 | tee "$OUT/fine_ncu_summary.txt" | head -60

echo ""
echo "##### detailed metrics for the first profiled launch (key sections) #####"
ncu --import "$OUT/fine.ncu-rep" --page details \
  --section ComputeWorkloadAnalysis MemoryWorkloadAnalysis Occupancy SchedulerStats WarpStateStats SourceCounters 2>&1 | \
  tee "$OUT/fine_ncu_details.txt" | head -120

echo ""
echo "########## artifacts ##########"
ls -la "$OUT"
echo "##### DONE #####"
