#!/usr/bin/env bash
# Job A: multi-scale profiling (coarse + medium) for Lab5 task2 hotspot.
# Partition lab5 (H800 MIG 1g.10gb), 30m wall. Run against the DEPLOYED ~/lab5/src
# (authoritative OJ submission code, SHA-verified); script + outputs live in dev.
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

echo "##### sanity: python deps + gpu #####"
$PY -c "import torch,triton,transformers; print('torch',torch.__version__,'triton',triton.__version__); print('dev',torch.cuda.get_device_name(0),'free',torch.cuda.mem_get_info()[0]/1024**3,'GiB')"

echo "##### Level 0: clear __pycache__ (avoid stale bytecode) #####"
find /home/h3240101033/lab5/src -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
echo "##### Level 0: py_compile check #####"
$PY -m py_compile "$SCRIPT" && echo "py_compile OK" || { echo "py_compile FAIL"; exit 1; }

echo ""
echo "########## SCALE 1 (COARSE): full BS2 public run ##########"
$PY "$SCRIPT" --scale coarse --model "$MODEL" --small "$SMALL" --public "$PUBLIC" --out-dir "$OUT" 2>&1

echo ""
echo "########## SCALE 2a (MEDIUM): torch.profiler key_averages + chrome trace ##########"
$PY "$SCRIPT" --scale medium --model "$MODEL" --small "$SMALL" --out-dir "$OUT" --n-decode 12 2>&1

echo ""
echo "########## SCALE 2b (MEDIUM): nsys timeline (copy-stream vs compute-stream overlap) ##########"
nsys profile \
  -o "$OUT/medium.nsys-rep" \
  -t cuda,nvtx,osrt \
  --stats=true \
  --force-overwrite=true \
  "$PY" "$SCRIPT" --scale medium-nsys --model "$MODEL" --small "$SMALL" --out-dir "$OUT" --n-decode 20 2>&1

echo ""
echo "########## artifacts ##########"
ls -la "$OUT"
echo "##### DONE #####"
