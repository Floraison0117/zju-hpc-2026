#!/bin/bash
# scratch: invocation A/B/C comparison — isolate why probe-style msprof and
# checker/profile.sh report different Task Durations for the same binary.
set -e
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/env.sh"
export PYTHONPATH="$ROOT:$ROOT/checker"
KERNEL_NAME=$(python3 - <<'PY'
import json, glob, os
f = sorted(glob.glob(os.path.expanduser('~/custom_opp/vendors/customize/op_impl/ai_core/tbe/kernel/ascend910b/fused_add_rms_norm/FusedAddRmsNorm_*.json')))[0]
print([e['kernelName'] for e in json.load(open(f))['kernelList']][0])
PY
)
echo "=== A: probe-style, LAB35_PROBE=96 (pre-init return) ==="
for i in 1 2 3; do
    rm -rf prof_a
    LAB35_PROBE=96 msprof op --kernel-name="$KERNEL_NAME" --warm-up=10 --launch-count=1 \
        --output="$ROOT/prof_a" python3 checker/test_op.py --profile 2 >/dev/null 2>&1 || echo msprofA_err
    python3 checker/get_time.py prof_a 2>/dev/null || echo A_ERR
done
echo "=== B: probe-style, no env var (real kernel) ==="
for i in 1 2 3; do
    rm -rf prof_b
    msprof op --kernel-name="$KERNEL_NAME" --warm-up=10 --launch-count=1 \
        --output="$ROOT/prof_b" python3 checker/test_op.py --profile 2 >/dev/null 2>&1 || echo msprofB_err
    python3 checker/get_time.py prof_b 2>/dev/null || echo B_ERR
done
echo "=== C: profile.sh style (real kernel) ==="
for i in 1 2 3; do
    rm -rf prof_c
    LANG=ascendc timeout 180 msprof op --warm-up=10 --kernel-name="$KERNEL_NAME" --launch-count=1 \
        --output="$ROOT/prof_c" /usr/local/python3.11.14/bin/python3 checker/test_op.py --profile 2 >/dev/null 2>&1 || echo msprofC_err
    python3 checker/get_time.py prof_c 2>/dev/null || echo C_ERR
done
