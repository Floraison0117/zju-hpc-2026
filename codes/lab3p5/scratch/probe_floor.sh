#!/bin/bash
# scratch: run the floor ladder probes in ONE job.
# Usage: hpc submit -p lab3p5 bash scratch/probe_floor.sh
set -e
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/env.sh"
export PYTHONPATH="$ROOT:$ROOT/checker:${PYTHONPATH:-}"
if ! python3 -c "import custom_ops_lib" >/dev/null 2>&1; then bash checker/build.sh; source "$ROOT/env.sh"; fi
KERNEL_NAME=$(python3 - <<'PY'
import json,glob,os
f=sorted(glob.glob(os.path.expanduser('~/custom_opp/vendors/customize/op_impl/ai_core/tbe/kernel/ascend910b/fused_add_rms_norm/FusedAddRmsNorm_*.json')))[0]
print([e['kernelName'] for e in json.load(open(f))['kernelList']][0])
PY
)
echo "[probe] kernel: $KERNEL_NAME"
for P in -100 -101 -102 -105 -109; do
    OUT="prof_probe_$P"
    rm -rf "$ROOT/$OUT"
    LAB35_PROBE="$P" msprof op --kernel-name="$KERNEL_NAME" --warm-up=10 --launch-count=1 \
        --output="$ROOT/$OUT" python3 checker/test_op.py --profile 2 >/dev/null 2>&1 || true
    US=$(python3 checker/get_time.py "$ROOT/$OUT" 2>/dev/null || echo "ERR")
    echo "PROBE $P : $US us"
done
