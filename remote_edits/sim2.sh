#!/bin/bash
set -e
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/env.sh"
export PYTHONPATH="$ROOT:$ROOT/checker:${PYTHONPATH:-}"
if ! python3 -c "import custom_ops_lib" >/dev/null 2>&1; then bash checker/build.sh; fi
source "$ROOT/env.sh"
KERNEL_NAME=$(python3 - <<'PY'
import json, glob, os
f = sorted(glob.glob(os.path.expanduser('~/custom_opp/vendors/customize/op_impl/ai_core/tbe/kernel/ascend910b/fused_add_rms_norm/FusedAddRmsNorm_*.json')))[0]
print([e['kernelName'] for e in json.load(open(f))['kernelList']][0])
PY
)
export SIM_B="${SIM_B:-560}" SIM_H="${SIM_H:-1024}"
OUT="op_sim2"
rm -rf "$ROOT/$OUT"
echo "[scratch] simulating $SIM_B x $SIM_H"
timeout 900 msprof op simulator \
  --soc-version=Ascend910B4 \
  --kernel-name="$KERNEL_NAME" \
  --launch-count=1 \
  --aic-metrics=PipeUtilization,ResourceConflictRatio \
  --output="$ROOT/$OUT" \
  python3 scratch/sim_shape.py || echo "[scratch] simulator exit $?"
echo "=== sim2 done ==="
