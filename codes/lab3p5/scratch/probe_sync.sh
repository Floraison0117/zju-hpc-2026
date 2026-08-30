#!/bin/bash
# scratch: sync-cost probes (P4 event pair / P5 scalar GM reads / P6 framework
# tiling / P99 selector-only baseline), 5 samples each.
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

for P in 99 4 5 6; do
    vals=""
    for i in 1 2 3 4 5; do
        OUT="prof_probe_run"
        rm -rf "$ROOT/$OUT"
        LAB35_PROBE="$P" msprof op --kernel-name="$KERNEL_NAME" --warm-up=10 --launch-count=1 \
            --output="$ROOT/$OUT" python3 checker/test_op.py --profile 2 >/dev/null 2>&1 || true
        v=$(python3 checker/get_time.py "$ROOT/$OUT" 2>/dev/null || echo ERR)
        vals="$vals $v"
    done
    echo "PROBE $P samples:$vals"
done
