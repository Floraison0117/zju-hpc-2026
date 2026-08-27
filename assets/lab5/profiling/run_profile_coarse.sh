#!/usr/bin/env bash
# Coarse-only re-run (after GenerationOutput.metrics field fix).
# Partition lab5 (H800 MIG 1g.10gb), 30m wall.
set -u
cd /home/h3240101033/lab5-fix-dev
export PYTHONPATH=/home/h3240101033/lab5/src
PY=/opt/lab5-venv/bin/python
MODEL=/home/h3240101033/HPC101/src/lab5/results/gemma-4-12b-gptq-cholesky-w4a16
SMALL=/home/h3240101033/lab05_final/datasets/performance_small.jsonl
PUBLIC=/home/h3240101033/lab05_final/datasets/performance_public.jsonl
OUT=/home/h3240101033/lab5-fix-dev/profile-out
mkdir -p "$OUT"
SCRIPT=/home/h3240101033/lab5-fix-dev/profile_multiscale.py

echo "##### sanity + clear __pycache__ + py_compile #####"
$PY -c "import torch; print('dev',torch.cuda.get_device_name(0),'free',torch.cuda.mem_get_info()[0]/1024**3,'GiB')"
find /home/h3240101033/lab5/src -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
$PY -m py_compile "$SCRIPT" && echo "py_compile OK" || { echo "py_compile FAIL"; exit 1; }

echo ""
echo "########## SCALE 1 (COARSE): full BS2 public run, phase breakdown ##########"
$PY "$SCRIPT" --scale coarse --model "$MODEL" --small "$SMALL" --public "$PUBLIC" --out-dir "$OUT" 2>&1

echo ""
echo "########## artifacts ##########"
ls -la "$OUT"/coarse_phase_breakdown.json 2>&1
echo "##### DONE #####"
